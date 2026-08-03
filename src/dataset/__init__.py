from copy import deepcopy
from typing import Dict
from torch.utils.data import Dataset
from .vimeo90k_dataset import Vimeo90kDataset
from .imagenet_dataset import ImageNetDataset
# from src.utils.registry import DATASET_REGISTRY
from src.utils.logger import log_dict_items


# dataset_list = ['Vimeo90k', 'ImageNet']

# def cvt_dataset_name(dataset_name: str) -> str:
#     _dataset_name = dataset_name.lower()
#     for name in dataset_list:
#         if name.lower() == _dataset_name:
#             return name
#     raise ValueError(f'Invalid dataset_name: "{dataset_name}".')

# def build_dataset(dataset_opt: Dict, is_train: bool=True) -> Dataset:
#     """
#     Args:
#         dataset_opt (Dict):
#             must have key: `name`
#             `name`: Vimeo90k, ImageNet, UVG
#         is_train (bool, optional): Defaults to True.
#     Returns:
#         Dataset:
#     """
#     dataset_opt = deepcopy(dataset_opt)
#     dataset_name = cvt_dataset_name(dataset_opt.pop('name')) # ex) OpenImage
#     # dataset_type = dataset_opt.pop('type') # ex) ImageDataset
#     # registry_key = dataset_name + dataset_type
#     registry_key = dataset_name + "Dataset"

#     dataset_opt['is_train'] = is_train

#     dataset = DATASET_REGISTRY.get(registry_key)(**dataset_opt)
#     log_dict_items(dataset_opt, level='DEBUG', indent=True) 
#     log_dict_items({'len(dataset)': len(dataset)}, level='INFO', indent=True)
#     assert len(dataset) > 0, 'len(dataset) should be > 0.'
#     return dataset
