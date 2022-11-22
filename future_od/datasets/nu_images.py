import os
import random
from collections import defaultdict

from nuimages import NuImages

import torch
import torch.utils.data
import torchvision as tv

from future_od.datasets.transforms import ImageRemap, JointCenterCrop, JointCompose, JointResize
from future_od.datasets.utils import concat_quaternion, construct_box_targets, inverse_quaternion


ORIGINAL_IMSIZE = (900, 1600)

ANNOTATED_FRAME = 6  # 6 before (0-5), 6 after (7-12)

CATEGORY_DICT = {
    0: "Vehicle",
    1: "Truck",
    2: "Trailer",
    3: "Pedestrian",
    4: "Bus",  # not VulnerableVehicle
    5: "Motorcyclist",
    6: "Bicyclist",
    7: "ConstructionVehicle",  # not OtherVulnerableVehicle
}
IGNORE_CATEGORY = len(CATEGORY_DICT)
DISCARD_CATEGORIES = {
    "a86329ee68a0411fb426dcad3b21452f",  # "flat.driveable_surface",
    "653f7efbb9514ce7b81d44070d6208c1",  # "movable_object.barrier",
    "063c5e7f638343d3a7230bc3641caf97",  # "movable_object.debris",
    "d772e4bae20f493f98e15a76518b31d7",  # "movable_object.pushable_pullable",
    "85abebdccd4d46c7be428af5a6173947",  # "movable_object.trafficcone",
}
CATEGORY_MAP = {
    "63a94dfa99bb47529567cd90d3b58384": IGNORE_CATEGORY,  # "animal",
    # "a86329ee68a0411fb426dcad3b21452f": IGNORE_CATEGORY,  # "flat.driveable_surface",
    "1fa93b757fc74fb197cdd60001ad8abf": 3,  # "human.pedestrian.adult",
    "b1c6de4c57f14a5383d9f963fbdcb5cb": 3,  # "human.pedestrian.child",
    "909f1237d34a49d6bdd27c2fe4581d79": 3,  # "human.pedestrian.construction_worker",
    "403fede16c88426885dd73366f16c34a": IGNORE_CATEGORY,  # "human.pedestrian.personal_mobility",
    "e3c7da112cd9475a9a10d45015424815": 3,  # "human.pedestrian.police_officer",
    "6a5888777ca14867a8aee3fe539b56c4": IGNORE_CATEGORY,  # "human.pedestrian.stroller",
    "b2d7c6c701254928a9e4d6aac9446d79": IGNORE_CATEGORY,  # "human.pedestrian.wheelchair",
    # "653f7efbb9514ce7b81d44070d6208c1": 9,  # "movable_object.barrier",
    # "063c5e7f638343d3a7230bc3641caf97": IGNORE_CATEGORY,  # "movable_object.debris",
    # "d772e4bae20f493f98e15a76518b31d7": IGNORE_CATEGORY,  # "movable_object.pushable_pullable",
    # "85abebdccd4d46c7be428af5a6173947": 8,  # "movable_object.trafficcone",
    "0a30519ee16a4619b4f4acfe2d78fb55": IGNORE_CATEGORY,  # "static_object.bicycle_rack",
    "fc95c87b806f48f8a1faea2dcc2222a4": 6,  # "vehicle.bicycle",
    "003edbfb9ca849ee8a7496e9af3025d4": 4,  # "vehicle.bus.bendy",
    "fedb11688db84088883945752e480c2c": 4,  # "vehicle.bus.rigid",
    "fd69059b62a3469fbaef25340c0eab7f": 0,  # "vehicle.car",
    "5b3cd6f2bca64b83aa3d0008df87d0e4": 7,  # "vehicle.construction",
    "7754874e6d0247f9855ae19a4028bf0e": 0,  # "vehicle.ego",
    "732cce86872640628788ff1bb81006d4": IGNORE_CATEGORY,  # "vehicle.emergency.ambulance",
    "7b2ff083a64e4d53809ae5d9be563504": IGNORE_CATEGORY,  # "vehicle.emergency.police",
    "dfd26f200ade4d24b540184e16050022": 5,  # "vehicle.motorcycle",
    "90d0f6f8e7c749149b1b6c3a029841a8": 2,  # "vehicle.trailer",
    "6021b5187b924d64be64a702e5570edf": 1,  # "vehicle.truck",
}


class NuImagesDataset(torch.utils.data.Dataset):
    """ """

    def __init__(
        self,
        root_path,
        split,
        night=False,
        front_camera_only=False,
        max_num_objects=256,
        frames=(ANNOTATED_FRAME,),
        joint_transform=None,
        image_transform=None,
        max_frame_random_offset: int = 0,
        frame_offset_sampler=None,
        annotated_frame_idx_override=None,
    ):
        self.root_path = root_path
        self.split = split
        self.max_num_objects = max_num_objects
        self.frames = frames
        self.image_transform = image_transform or tv.transforms.Compose(
            [
                ImageRemap(),
                tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.joint_transform = joint_transform or JointCompose(
            [
                JointResize(size=(256, 962)),
                JointCenterCrop(size=(256, 960)),
            ]
        )
        self.max_frame_random_offset = max_frame_random_offset
        self.frame_offset_sampler = frame_offset_sampler
        self.annotated_frame_idx_override = annotated_frame_idx_override
        assert split in ("mini", "train", "val", "test")

        self.nuimages = NuImages(version="v1.0-" + split, dataroot=root_path)
        self.object_anns_dict = defaultdict(list)
        self.samples = []
        self._init_data(night, front_camera_only)

    def _init_data(self, night: bool, front_camera_only: bool):
        skip_counter = 0
        for o in self.nuimages.object_ann:
            if o["category_token"] not in DISCARD_CATEGORIES:
                self.object_anns_dict[o["sample_data_token"]].append(o)

        # pre-compute lookup maps to avoid tons of list traversal
        sensors = {s["token"]: s for s in self.nuimages.sensor}
        cs_to_s = {cs["token"]: cs["sensor_token"] for cs in self.nuimages.calibrated_sensor}
        log_to_file = {log["token"]: log["logfile"] for log in self.nuimages.log}

        for sample in self.nuimages.sample:
            skip_counter += 1
            # Check night condition
            if night:
                logfile = log_to_file[sample["log_token"]]
                hour = int(logfile.split("-")[4])
                if 6 < hour < 18:
                    continue

            # Check camera condition
            if front_camera_only:
                sample_data = self.nuimages.get("sample_data", sample["key_camera_token"])
                sensor = sensors[cs_to_s[sample_data["calibrated_sensor_token"]]]
                if sensor["channel"] != "CAM_FRONT":
                    continue

            # Check that we have exactly 6 frames to each side
            sd_tokens = self.nuimages.get_sample_content(sample["token"])
            if len(sd_tokens) != 13 or sd_tokens[6] != sample["key_camera_token"]:
                continue

            # All checks passed, add the sample (and counteract the skip_counter increase)
            skip_counter -= 1
            self.samples.append((sample, sd_tokens))

        if skip_counter:
            print(f"skipped {skip_counter} samples")

    def __len__(self):
        return len(self.samples)

    def _read_images(self, sample_data_tokens):
        if self.frame_offset_sampler is None:
            random_offset = random.randint(0, self.max_frame_random_offset)
        else:
            random_offset = self.frame_offset_sampler()
        frames = [frame + random_offset for frame in self.frames]
        filenames = [
            self.nuimages.get("sample_data", sample_data_tokens[frame_idx])["filename"]
            for frame_idx in frames
        ]
        images = [
            tv.io.read_image(os.path.join(self.root_path, filename)) for filename in filenames
        ]
        images = torch.stack(images, dim=0)
        images = self.image_transform(images)
        annotated_frame_idx = (
            self.annotated_frame_idx_override
            if self.annotated_frame_idx_override is not None
            else frames.index(ANNOTATED_FRAME)
        )
        return images, annotated_frame_idx, frames

    def _get_meta(self, sample):
        return "none", -1.0  #  weather icon, solar angle

    def _get_imu(self, sample_data_tokens, frame_ids):
        """
        Returns:
            Tensor: (L, 3) containing translation relative to first frame translation (l=0)
            Tensor: (L, 3) containing acceleration
            Tensor: (L, 4) containing rotation relative to first frame rotation (l=0)
            Tensor: (L, 3) containing rotation rate
            Tensor: (L, 3) containing speed in driving direction
        """
        translation = torch.empty((len(frame_ids), 3), dtype=torch.float)
        acceleration = torch.empty((len(frame_ids), 3), dtype=torch.float)
        rotation = torch.empty((len(frame_ids), 4), dtype=torch.float)
        rotation_rate = torch.empty((len(frame_ids), 3), dtype=torch.float)
        speed = torch.empty((len(frame_ids), 1), dtype=torch.float)
        for l, frame_idx in enumerate(frame_ids):
            sample_data = self.nuimages.get("sample_data", sample_data_tokens[frame_idx])
            ego_pose = self.nuimages.get("ego_pose", sample_data["ego_pose_token"])
            translation[l] = torch.tensor(ego_pose["translation"])
            acceleration[l] = torch.tensor(ego_pose["acceleration"])
            rotation[l] = torch.tensor(ego_pose["rotation"])
            rotation_rate[l] = torch.tensor(ego_pose["rotation_rate"])
            speed[l] = torch.tensor(ego_pose["speed"])
        translation = translation - translation[0:1]
        rotation = concat_quaternion(
            rotation, inverse_quaternion(rotation[0:1, :].expand_as(rotation))
        )
        return translation, acceleration, rotation, rotation_rate, speed

    def _get_object_boxes(self, object_annos):
        box_list = [torch.tensor(obj["bbox"]) for obj in object_annos]
        boxes = torch.stack(box_list) if box_list else torch.zeros((0, 4))
        return boxes

    def _get_object_classes(self, object_annos):
        classes = [CATEGORY_MAP[obj["category_token"]] for obj in object_annos]
        return torch.tensor(classes, dtype=torch.int64)

    def _get_od_anno(self, sample_data_token):
        annos = self.object_anns_dict[sample_data_token]
        boxes = self._get_object_boxes(annos)
        classes = self._get_object_classes(annos)
        return boxes, classes

    def __getitem__(self, idx):
        """We use a dense representation of objects. The box- and class tensors have a constant
        size of Nmax objects. If there are fewer than Nmax objects in an image, then we mark the
        unused slots as inactive using the active-tensor.
        Returns:
            dict: {
                video (Tensor)                  : of size (L, 3, H, W)
                boxes (Tensor)                  : (Nmax, 4) tensor, using xyxy representation in image coordinates
                classes (LongTensor)            : of size (Nmax,)
                active (LongTensor)             : of size (Nmax,) where 1 is object and 0 is a nonactive object slot
                annotated_frame_idx (LongTensor): of size (1,) marking which of the L frames is annotated
                ignore_boxes (Tensor)           : of size (Nmax, 4) as (x,y,x,y) in image coordinates
                weather (str)                   : @todo not yet used, seems to be missing sometimes?
                sun_elevation (float)           : @todo not yet used, seems to be missing sometimes?
                translation (Tensor) (L, 3)     : relative first returned timepoint, m
                acceleration (Tensor) (L, 3)    : m/s^2
                rotation (Tensor) (L, 4)        : relative first returned timepoint, quaternion
                rotation_rate (Tensor) (L, 3)   : gyro-like output
                idf (str)                       : Identifier for this sample
            }
        """
        sample, sample_data_tokens = self.samples[idx]
        video, annotated_frame_idx, frame_ids = self._read_images(sample_data_tokens)
        weather, sun_elevation = self._get_meta(sample)
        imu = self._get_imu(sample_data_tokens, frame_ids)
        annos = self.object_anns_dict[sample["key_camera_token"]]
        boxes = self._get_object_boxes(annos)
        classes = self._get_object_classes(annos)
        video, boxes, classes = self.joint_transform(video, boxes, classes)
        boxes, classes, ignore_boxes, active = construct_box_targets(
            boxes,
            classes,
            max_num_objects=self.max_num_objects,
            ignore_categories={IGNORE_CATEGORY},
        )
        idf = f"{idx}"

        output = {
            "video": video,
            "boxes": boxes,
            "classes": classes,
            "active": active,
            "annotated_frame_idx": torch.tensor(annotated_frame_idx, dtype=torch.int64),
            "ignore_boxes": ignore_boxes,
            "weather": weather,
            "sun_elevation": sun_elevation,
            "translation": imu[0],
            "acceleration": imu[1],
            "rotation": imu[2],
            "rotation_rate": imu[3],
            "speed": imu[4],
            "idf": idf,
        }
        return output
