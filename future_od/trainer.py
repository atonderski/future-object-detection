import os

import wandb

import torch
import torch.distributed as distrib

from future_od.utils.distributed import EXIT, gather_distrib_od_map_stuffs, reduce_distrib_loss
from future_od.utils.od_map import aggregate_mean_average_precision
from future_od.utils.recursive_functions import recursive_detach_cpu, recursive_to
from future_od.utils.stats import AverageMeter
from future_od.utils.visualization import visualize, visualize_wandb
from future_od.utils.wandb import WandBConfig


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        lr_sched,
        train_loader,
        val_loaders,
        checkpoint_path,
        visualization_path,
        save_name,
        device,
        print_interval,
        visualization_epochs,
        visualization_iterations,
        category_dict,
        checkpoint_epochs=None,
        gradient_clip_value=None,
        distributed=False,
        is_master=False,
        wandb_config=WandBConfig(),
        max_norm=0.0,
    ):
        self._model = model
        self._model_no_ddp = (
            model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        )
        self._optimizer = optimizer
        self._lr_sched = lr_sched

        self._train_loader = train_loader
        self._val_loaders = val_loaders
        if isinstance(self._val_loaders, list):
            self._val_loaders = {f"val{idx}": ldr for idx, ldr in enumerate(self._val_loaders)}
        assert (
            len(self._train_loader) and min(len(loader) for loader in self._val_loaders) > 0
        ), "All loaders must be non-empty"

        self._gradient_clip_value = gradient_clip_value
        assert gradient_clip_value is None or isinstance(gradient_clip_value, (int, float))

        self._save_checkpoints = bool(checkpoint_epochs)
        self._checkpoint_path = checkpoint_path
        self._visualization_path = visualization_path
        self._save_name = save_name
        self._device = device

        self._print_interval = print_interval
        self._visualization_epochs = visualization_epochs
        self._visualization_iterations = visualization_iterations
        self._category_dict = category_dict
        self._distributed = distributed
        self._is_master = is_master

        # Initialize statistics variables @todo should we also add some KPI s.a. mIoU?
        self._stats = {}
        modes = ["train"] + list(self._val_loaders.keys())
        for mode in modes:
            for key in self._model_no_ddp.get_stat_idfs():
                self._stats[f"{mode} {key} loss"] = AverageMeter()

        self._epoch = 0
        self._training_iterations = 0
        self._wandb_config = wandb_config
        self._max_norm = max_norm

    def train(self, max_epochs):
        self._setup_wandb(tags=["training"])
        print(f"Training epochs {self._epoch + 1} to {max_epochs}. Moving model to {self._device}.")
        if not self._distributed:
            self._model.to(self._device)
        for epoch in range(self._epoch + 1, max_epochs + 1):
            self._epoch = epoch
            if self._distributed:
                self._train_loader.sampler.set_epoch(
                    epoch
                )  # Needed for correct shuffling of dataset
            print(f"Starting epoch {epoch} with lr={self._lr_sched.get_last_lr()}")
            self._train_epoch()
            self._lr_sched.step()
            if EXIT.is_set():
                return
            if self._save_checkpoints:
                print("Saving Checkpoint")
                self.save_checkpoint(is_final=(epoch == max_epochs))

        print(f"Finished training!")

    def eval(self):
        self._setup_wandb(tags=["eval"])
        print(f"Running eval. Moving model to {self._device}.")
        if not self._distributed:
            self._model.to(self._device)
        self._run_eval()

    def _setup_wandb(self, tags=None):
        if self._is_master and self._wandb_config.enabled:
            conf = self._wandb_config
            wandb.init(
                project=conf.project,
                entity=conf.entity,
                config=conf.hyperparams,
                name=conf.name,
                notes=conf.notes,
                resume="must" if conf.resume_id else None,
                id=conf.resume_id,
                tags=tags,
            )
            if self._wandb_config.watch_model:
                wandb.watch(self._model_no_ddp)

    def _run_eval(self):
        self._model.train(False)
        with torch.no_grad():
            for loader_name, data_loader in self._val_loaders.items():
                if hasattr(self._model_no_ddp._model, "drop_mode"):
                    self._model_no_ddp._model.drop_mode = f"{loader_name}"
                self._run_epoch(mode=f"{loader_name}", data_loader=data_loader)

    def _train_epoch(self):
        """Do one epoch of training and validation."""
        self._model.train(True)
        if hasattr(self._model_no_ddp._model, "drop_mode"):
            self._model_no_ddp._model.drop_mode = "train"
        self._run_epoch(mode="train", data_loader=self._train_loader)

        self._run_eval()

        # Update all stat values
        for stat_value in self._stats.values():
            if isinstance(stat_value, AverageMeter):
                stat_value.new_epoch()

    def _run_epoch(self, mode, data_loader):
        """We expect to do Video Semantic Segmentation (VSS), Video 2D Object Detection (VOD),
        and Video Instance Segmentation (VIS).
        """
        log_to_wandb = self._is_master and self._wandb_config.enabled
        num_iterations = len(data_loader)
        od_map_stuff_lst = [[], [], [], []]
        hardest_data, hardest_output, highest_loss = None, None, -1e10
        for i, data in enumerate(data_loader):
            if EXIT.is_set():
                return
            cpu_data = data
            data = recursive_to(data, self._device)
            if hasattr(self._model, "supervisor"):
                data = self._model.supervisor.augment_data(data, mode)

            visualize_this_iteration = (
                i in self._visualization_iterations
                and self._epoch in self._visualization_epochs
                and self._is_master
            )

            if mode == "train":
                self._optimizer.zero_grad()
            model_output, state, loss, stats, od_map_stuffs = self._model(
                data=data,
                visualize=visualize_this_iteration,
                epoch=self._epoch,
                distributed=self._distributed,
            )
            if mode == "train":
                loss.backward()
                if self._epoch == 1 and i == 0:
                    print("\nFirst iteration. Checking whether all layers gradients.")
                    for name, param in self._model.named_parameters():
                        if param.grad is None:
                            print(name, "got no gradient")
                if self._max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_norm)
                self._optimizer.step()
                self._training_iterations += 1

            if self._is_master and loss > highest_loss:
                highest_loss = loss.detach().cpu()
                hardest_data = cpu_data
                hardest_output = recursive_detach_cpu(model_output)

            # Save some stats
            if self._distributed:
                stats = reduce_distrib_loss(stats, average=True)
            stats = {key: loss.detach().to("cpu").item() for key, loss in stats.items()}
            self.save_stats(stats, model_output, data, mode)

            # Only save stats from up to 10k images to avoid memory issues
            world_size = 1 if not self._distributed else distrib.get_world_size()
            if i * data_loader.batch_size * world_size < 10000:
                if self._distributed:
                    od_map_stuffs = gather_distrib_od_map_stuffs(od_map_stuffs)
                else:
                    od_map_stuffs = [[elem] for elem in od_map_stuffs]
                for idx in range(4):
                    for elem in od_map_stuffs[idx]:
                        od_map_stuff_lst[idx].append(elem.detach().cpu())

            if visualize_this_iteration:
                self.visualize_batch(data, model_output, mode, log_to_wandb)
            if (i + 1) % self._print_interval == 0:
                loss_str = [(self._stats[f"{mode} {key} loss"].avg, key) for key in stats.keys()]
                loss_str = [f"{val:.5f} ({name})" for val, name in loss_str]
                loss_str = "  ".join(loss_str)
                print(f"[{mode}: {self._epoch}, {i + 1:4d}/{num_iterations}] Loss: {loss_str}.")

            # Release all memory
            stats_keys = list(stats.keys())
            del model_output, state, data, od_map_stuffs, stats, loss

        # end for
        loss_items = [(self._stats[f"{mode} {key} loss"].avg, key) for key in stats_keys]
        loss_str = [f"{val:.5f} ({name})" for val, name in loss_items]
        loss_str = "  ".join(loss_str)
        print(f"[{mode}: {self._epoch}] Loss: {loss_str}")

        ap = aggregate_mean_average_precision(
            torch.cat(od_map_stuff_lst[0], dim=2),
            torch.cat(od_map_stuff_lst[1], dim=2),
            torch.cat(od_map_stuff_lst[2], dim=2),
            torch.stack(od_map_stuff_lst[3], dim=2),
            self._device,
        )
        print(
            f"AP50 for epoch is:",
            " ".join([f"{elem:.3f}" for elem in ap["all"][0, :, 0]]),
        )
        print(
            f"MAP for epoch is:",
            " ".join([f"{elem:.3f}" for elem in ap["threshavg"][:, 0]]),
        )
        print(
            f"MAP for small objects is:",
            " ".join([f"{elem:.3f}" for elem in ap["threshavg"][:, 1]]),
        )
        print(
            f"MAP for medium objects is:",
            " ".join([f"{elem:.3f}" for elem in ap["threshavg"][:, 2]]),
        )
        print(
            f"MAP for large objects is:",
            " ".join([f"{elem:.3f}" for elem in ap["threshavg"][:, 3]]),
        )

        if log_to_wandb:
            wandb_log = {
                "epoch": self._epoch,
                "iteration": self._training_iterations,
            }
            for style in ["classavg", "generic"]:
                for size_idx, size in enumerate(["", "-small", "-medium", "-large"]):
                    wandb_log[f"{mode}-{style}/ap{size}"] = ap[f"{style} threshavg"][size_idx]
                    wandb_log[f"{mode}-{style}/ap50{size}"] = ap[f"{style}"][0, size_idx]
                    wandb_log[f"{mode}-{style}/ap70{size}"] = ap[f"{style}"][4, size_idx]
            for class_idx, class_name in enumerate(self._category_dict.values()):
                wandb_log[f"{mode}-class/ap_{class_name}"] = ap[f"threshavg"][class_idx, 0]
                wandb_log[f"{mode}-class/ap50_{class_name}"] = ap[f"all"][0, class_idx, 0]
                wandb_log[f"{mode}-class/ap70_{class_name}"] = ap[f"all"][4, class_idx, 0]
            for val, name in loss_items:
                wandb_log[f"{mode}-losses/{name}"] = val
            wandb.log(wandb_log)

            if (self._epoch in self._visualization_epochs) and self._is_master:
                self.visualize_batch(
                    hardest_data, hardest_output, mode, log_to_wandb, prefix="hardest_"
                )

    def save_checkpoint(self, is_final: bool = False):
        """Saves a checkpoint of the network and other variables."""
        if not self._is_master:
            return
        state = {
            "epoch": self._epoch,
            "net_type": type(self._model_no_ddp).__name__,
            "net": self._model_no_ddp.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "lr_schedule": self._lr_sched.state_dict(),
            "stats": self._stats,
            "device": self._device,
        }
        file_path = "{}/{}.pth.tar".format(self._checkpoint_path, self._save_name)
        torch.save(state, file_path)
        if is_final:
            file_path = "{}/{}_final.pth.tar".format(self._checkpoint_path, self._save_name)
            torch.save({"net": state["net"]}, file_path)

    def load_checkpoint(self, checkpoint: str = None, load_only_net=False):
        """Loads a network checkpoint file."""
        print(f"Loading checkpoint: {str(checkpoint)}")
        if checkpoint is None:  # Load most recent checkpoint
            checkpoint_path = "{}/{}.pth.tar".format(self._checkpoint_path, self._save_name)
        elif isinstance(checkpoint, str):  # checkpoint is the epoch file path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError("Checkpoint must be string or None")
        if not os.path.isfile(checkpoint_path):
            print(
                f"WARNING: Attempted to load checkpoint at epoch {checkpoint}, but it does not"
                + " exist. Continuing without loading. If runfile is correctly set up, there will"
                + " be an upcoming training stage that will begin from scratch."
            )
            return
        checkpoint_dict = torch.load(checkpoint_path)
        assert (
            type(self._model_no_ddp).__name__ == checkpoint_dict["net_type"]
        ), "Network is not of correct type"
        self._model_no_ddp.load_state_dict(checkpoint_dict["net"])
        if not load_only_net:
            self._epoch = checkpoint_dict["epoch"]
            self._optimizer.load_state_dict(checkpoint_dict["optimizer"])
            self._stats = checkpoint_dict["stats"]
            self._device = checkpoint_dict["device"]
            self._lr_sched.load_state_dict(checkpoint_dict["lr_schedule"])
        print("Loaded: {}".format(checkpoint_path))

    def save_stats(self, stats, model_output, data, mode):
        for name, loss in stats.items():
            self._stats[f"{mode} {name} loss"].update(loss, 1)

    def visualize_batch(self, data, model_output, mode, log_to_wandb, prefix=""):
        B, L_out, T, M, C = model_output["class_scores"].size()
        L_in = data["video"].size()[1]
        assert L_in == L_out or L_out == 1
        BACKGROUND_CLASS = C  # Max index + 1
        video = data["video"].cpu()  # (B, L, 3, H, W)
        pred_class_scores = (
            model_output["class_scores"][:, :, 0, :, :].cpu().detach()
        )  # (B, L, M, C)
        pred_boxes = model_output["boxes"][:, :, 0, :, :].cpu().detach()  # (B, L, M, 4)
        anno_classes = data["classes"].cpu()  # (B, N)
        anno_boxes = data["boxes"].cpu()  # (B, N, 4)
        anno_active = data["active"].cpu()  # (B, N)
        anno_frame_ids = data["annotated_frame_idx"].cpu()  # (B,)
        anno_classes[anno_active == 0] = BACKGROUND_CLASS
        ignore_boxes = data["ignore_boxes"].cpu()  # (B, N, 4)
        model_moods = model_output["moods"]

        wandb_images = []
        for b in range(B):
            fpath = os.path.join(self._visualization_path, f"{prefix}{mode}_b{b}_anno.png")
            visualize(
                video[b, anno_frame_ids[b]],
                anno_classes[b],
                anno_boxes[b],
                fpath,
                BACKGROUND_CLASS,
            )
            for l in range(L_in):
                # Handle -1 index
                has_anno = l == anno_frame_ids[b]
                # Usually all frames have detections, otherwise only the annotated frame
                has_det = L_in == L_out or has_anno
                pred_cls, pred_box, anno_cls, anno_box, ignores = (
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                if has_anno:
                    anno_cls, anno_box, ignores = (
                        anno_classes[b],
                        anno_boxes[b],
                        ignore_boxes[b],
                    )
                if has_det:
                    # if there is only 1 det we assume it's for the annotated frame
                    l_det = l if L_out == L_in else 0
                    pred_cls, pred_box = (
                        pred_class_scores[b, l_det],
                        pred_boxes[b, l_det],
                    )

                # Deprecated
                # fpath = os.path.join(self._visualization_path, f"{prefix}{mode}_b{b}_pred_l{l}.png")
                # visualize(video[b, l], pred_cls, pred_box, fpath, BACKGROUND_CLASS)

                if not (log_to_wandb and b < self._wandb_config.num_images):
                    continue
                wandb_image = visualize_wandb(
                    image=video[b, l],
                    pred_scores=pred_cls,
                    pred_boxes=pred_box,
                    background_class=BACKGROUND_CLASS,
                    category_dict=self._category_dict,
                    anno_classes=anno_cls,
                    anno_boxes=anno_box,
                    ignore_boxes=ignores,
                    model_mood=model_moods[b][l],
                )
                wandb_images.append(wandb_image)

        if wandb_images:
            wandb.log(
                {
                    f"visualization/{prefix}{mode}_bounding_boxes": wandb_images,
                    "epoch": self._epoch,
                }
            )
