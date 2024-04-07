from .kitti_dataset_1215 import KITTIDataset
from .kitti_dataset_3d import KITTI3DDataset
from .sceneflow_dataset import SceneFlowDatset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "kitti3d": KITTI3DDataset
}
