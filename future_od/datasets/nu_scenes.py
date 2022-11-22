import json
import os
from collections import defaultdict
from typing import List, Optional

import numpy as np
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes

import torch
import torch.utils.data
import torchvision as tv

from future_od.datasets.transforms import ImageRemap, JointCenterCrop, JointCompose, JointResize
from future_od.datasets.utils import concat_quaternion, construct_box_targets, inverse_quaternion


ORIGINAL_IMSIZE = (900, 1600)
FRONT_CAMERA = "CAM_FRONT"
ALL_CAMERAS = (
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
)
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
    "flat.driveable_surface",
    "movable_object.barrier",
    "movable_object.debris",
    "movable_object.pushable_pullable",
    "movable_object.trafficcone",
}
CATEGORY_MAP = {
    "animal": IGNORE_CATEGORY,
    "human.pedestrian.adult": 3,
    "human.pedestrian.child": 3,
    "human.pedestrian.construction_worker": 3,
    "human.pedestrian.personal_mobility": IGNORE_CATEGORY,
    "human.pedestrian.police_officer": 3,
    "human.pedestrian.stroller": IGNORE_CATEGORY,
    "human.pedestrian.wheelchair": IGNORE_CATEGORY,
    "static_object.bicycle_rack": IGNORE_CATEGORY,
    "vehicle.bicycle": 6,
    "vehicle.bus.bendy": 4,
    "vehicle.bus.rigid": 4,
    "vehicle.car": 0,
    "vehicle.construction": 7,
    "vehicle.ego": 0,
    "vehicle.emergency.ambulance": IGNORE_CATEGORY,
    "vehicle.emergency.police": IGNORE_CATEGORY,
    "vehicle.motorcycle": 5,
    "vehicle.trailer": 2,
    "vehicle.truck": 1,
}
SPLIT_TO_VERSION = {
    "train": "v1.0-trainval",
    "val": "v1.0-trainval",
    "mini_train": "v1.0-mini",
    "mini_val": "v1.0-mini",
    "test": "v1.0-test",
}


class NuScenesDataset(torch.utils.data.Dataset):
    """NuScenes dataset."""

    def __init__(
        self,
        root_path,
        split,
        night=False,
        front_camera_only=False,
        max_num_objects=256,
        frame_offsets=(0,),
        joint_transform=None,
        image_transform=None,
        annotated_frame_idx_override=None,
        filter_offsets=None,
    ):
        self.root_path = root_path
        self.max_num_objects = max_num_objects
        self.frame_offsets = frame_offsets
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
        self.annotated_frame_idx_override = annotated_frame_idx_override
        split = split.replace("-", "_")
        assert split in SPLIT_TO_VERSION, f"split must be one of {SPLIT_TO_VERSION.keys()}"

        self.nuscenes = NuScenes(version=SPLIT_TO_VERSION[split], dataroot=root_path)
        self.nusc_can = NuScenesCanBus(dataroot=root_path)
        self.object_anns_dict = defaultdict(list)
        self.samples = []
        self.imus = {}
        self._init_data(split, night, front_camera_only, filter_offsets)

    def _init_data(
        self,
        split: str,
        night: bool,
        front_camera_only: bool,
        filter_offsets: Optional[List[float]],
    ):
        # Ensure offsets are ordered
        numeric_offsets = tuple([off for off in self.frame_offsets if not isinstance(off, str)])
        assert numeric_offsets == tuple(sorted(numeric_offsets)), "Offsets must be ordered"

        # Filter out samples belonging to the desired split
        print(f"Filtering out frames belonging to the {split} split")
        split_scenes = create_splits_scenes()[split]
        # Remove scenes that don't have can bus data
        split_scenes = {s for s in split_scenes if int(s[-4:]) not in self.nusc_can.can_blacklist}
        split_samples = []
        for sample in self.nuscenes.sample:
            scene_record = self.nuscenes.get("scene", sample["scene_token"])
            if scene_record["name"] in split_scenes:
                split_samples.append(sample)

        with open(
            os.path.join(self.nuscenes.dataroot, self.nuscenes.version, "image_annotations.json")
        ) as file:
            annotations_2d = json.load(file)
        for o in annotations_2d:
            if o["category_name"] not in DISCARD_CATEGORIES:
                self.object_anns_dict[o["sample_data_token"]].append(o)

        skip_counter = 0
        cameras = [FRONT_CAMERA] if front_camera_only else ALL_CAMERAS
        for sample in split_samples:
            skip_counter += len(cameras)

            # Check night condition
            if night:
                scene = self.nuscenes.get("scene", sample["scene_token"])
                logfile = self.nuscenes.get("log", scene["log_token"])["logfile"]
                hour = int(logfile.split("-")[4])
                if 6 < hour < 18:
                    continue

            for camera in cameras:
                sample_data = self.nuscenes.get("sample_data", sample["data"][camera])
                # Check offset filter
                if filter_offsets is not None:
                    matches = self._get_surrounding_data(sample_data, filter_offsets)
                    if len(matches) != len(filter_offsets):
                        continue
                # Get surrounding frames
                sample_datas = self._get_surrounding_data(sample_data, self.frame_offsets)
                if not len(sample_datas) >= len(self.frame_offsets):
                    continue
                # All checks passed, add the sample (and counteract the skip_counter increase)
                self.samples.append(sample_datas)
                skip_counter -= 1

        self._init_imu_for_samples(split_scenes)

        if skip_counter:
            print(f"skipped {skip_counter} samples")

    def _get_surrounding_data(self, sample_data, offsets):
        frames = {0.0: sample_data}

        # Backwards pass
        curr_data = sample_data
        prev_offsets = [
            off for off in reversed(offsets) if off != "next" and (off == "prev" or off < 0)
        ]
        while prev_offsets and curr_data["prev"]:
            curr_data = self.nuscenes.get("sample_data", curr_data["prev"])
            curr_diff = round((curr_data["timestamp"] - sample_data["timestamp"]) / 1e6, 2)
            if not isinstance(prev_offsets[0], str) and curr_diff < prev_offsets[0]:
                break
            if curr_diff == prev_offsets[0] or prev_offsets[0] == "prev":
                frames[curr_diff] = curr_data
                prev_offsets.pop(0)

        # Forward pass
        curr_data = sample_data
        next_offsets = [off for off in offsets if off != "prev" and (off == "next" or off > 0)]
        while next_offsets and curr_data["next"]:
            curr_data = self.nuscenes.get("sample_data", curr_data["next"])
            curr_diff = round((curr_data["timestamp"] - sample_data["timestamp"]) / 1e6, 2)
            if not isinstance(next_offsets[0], str) and curr_diff > next_offsets[0]:
                break
            if curr_diff == next_offsets[0] or next_offsets[0] == "next":
                frames[curr_diff] = curr_data
                next_offsets.pop(0)

        # Sort the dict
        return {k: v for k, v in sorted(frames.items())}

    def _init_imu_for_samples(self, split_scenes):
        """Here we go through all samples and find closest canbus pose for each."""
        scene_poses = {}
        scene_utimes = {}
        for scene_name in split_scenes:
            pose_msg = self.nusc_can.get_messages(scene_name=scene_name, message_name="pose")
            scene_poses[scene_name] = pose_msg
            scene_utimes[scene_name] = np.array([pose["utime"] for pose in pose_msg])

        for sample_datas in self.samples:
            sample = self.nuscenes.get("sample", next(iter(sample_datas.values()))["sample_token"])
            scene_name = self.nuscenes.get("scene", sample["scene_token"])["name"]
            for _, sample_data in sorted(sample_datas.items()):
                closes_idx = np.argmin(np.abs(scene_utimes[scene_name] - sample_data["timestamp"]))
                canbus_pose = scene_poses[scene_name][closes_idx]
                ego_pose = self.nuscenes.get("ego_pose", sample_data["ego_pose_token"])
                self.imus[sample_data["token"]] = {**canbus_pose, **ego_pose}

    def __len__(self):
        return len(self.samples)

    def _read_images(self, all_sample_datas):
        # TODO: figure out which images to read based on the provided offsets
        filenames = [sample_data["filename"] for sample_data in all_sample_datas.values()]
        images = [
            tv.io.read_image(os.path.join(self.root_path, filename)) for filename in filenames
        ]
        images = torch.stack(images, dim=0)
        images = self.image_transform(images)
        annotated_frame_idx = (
            self.annotated_frame_idx_override
            if self.annotated_frame_idx_override is not None
            else self.frame_offsets.index(0.0)
        )
        return images, annotated_frame_idx

    def _get_meta(self, sample):
        return "none", -1.0  #  weather icon, solar angle

    def _get_imu(self, all_sample_data):
        """
        Returns:
            Tensor: (L, 3) containing translation relative to first frame translation (l=0)
            Tensor: (L, 3) containing acceleration
            Tensor: (L, 4) containing rotation relative to first frame rotation (l=0)
            Tensor: (L, 3) containing rotation rate
            Tensor: (L, 1) containing speed in driving direction
        """
        n_samples = len(all_sample_data)
        translation = torch.empty((n_samples, 3), dtype=torch.float)
        acceleration = torch.empty((n_samples, 3), dtype=torch.float)
        rotation = torch.empty((n_samples, 4), dtype=torch.float)
        rotation_rate = torch.empty((n_samples, 3), dtype=torch.float)
        speed = torch.empty((n_samples, 1), dtype=torch.float)
        for l, (offset, sample_data) in enumerate(all_sample_data.items()):
            imu = self.imus[sample_data["token"]]
            translation[l] = torch.tensor(imu["translation"])
            acceleration[l] = torch.tensor(imu["accel"])
            rotation[l] = torch.tensor(imu["rotation"])
            rotation_rate[l] = torch.tensor(imu["rotation_rate"])
            speed[l] = torch.tensor(imu["vel"][0])
        translation = translation - translation[0:1]
        rotation = concat_quaternion(
            rotation, inverse_quaternion(rotation[0:1, :].expand_as(rotation))
        )
        return translation, acceleration, rotation, rotation_rate, speed

    def _get_object_boxes(self, object_annos):
        box_list = [torch.tensor(obj["bbox_corners"]) for obj in object_annos]
        boxes = torch.stack(box_list) if box_list else torch.zeros((0, 4))
        return boxes

    def _get_object_classes(self, object_annos):
        classes = [CATEGORY_MAP[obj["category_name"]] for obj in object_annos]
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
                idf (str)                       : Identifier for this sample
            }
        """
        all_sample_data = self.samples[idx]
        keyframe_sample_data = all_sample_data[0]
        if 0 not in self.frame_offsets:
            # 0 is always added so here we remove it if necessary
            all_sample_data.pop(0)
        video, annotated_frame_idx = self._read_images(all_sample_data)
        weather, sun_elevation = self._get_meta(keyframe_sample_data)
        imu = self._get_imu(all_sample_data)
        annos = self.object_anns_dict[keyframe_sample_data["token"]]
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
            "temporal_offsets": torch.tensor(list(all_sample_data.keys()), dtype=torch.float32),
            "idf": idf,
        }
        return output
