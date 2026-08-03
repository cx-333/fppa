# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501
from typing import Any, Optional, Callable, Dict, Tuple, List

import inspect
import os
import os.path as osp

from src.utils.misc import Color as Clr
from src.utils.logger import get_root_logger


class Registry:
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map: Dict[str, Dict] = {}

    def _do_register(self, name: str, obj: Any, filename: str) -> None:
        assert name not in self._obj_map, (
            f"An object named '{name}' was already registered "
            f"in '{self._name}' registry!"
        )
        self._obj_map[name] = {"obj": obj, "filename": filename}

    def register(self) -> Callable:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """

        def deco(func_or_class: Any) -> Any:
            filename = osp.basename(inspect.stack()[1].filename)
            name = func_or_class.__name__
            self._do_register(name, func_or_class, filename)
            return func_or_class

        return deco

    def get(self, class_name: str, display_name: Optional[str] = None) -> Any:
        ret = self._obj_map.get(class_name)
        if ret is None:
            raise KeyError(
                f"No object named '{class_name}' found in '{self._name}' registry!"
            )
        obj, filename = ret["obj"], ret["filename"]
        logger = get_root_logger()
        display_name = self._name if display_name is None else display_name
        logger.info(
            f"{display_name} [{Clr.BLUE}{class_name}{Clr.RESET}] (from {Clr.GREEN}{filename}{Clr.RESET}) is built"
        )
        return obj

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def keys(self) -> List:
        return list(self._obj_map.keys())


TRAINER_REGISTRY = Registry("trainer")              # trainer
OPTIMIZER_REGISTRY = Registry("optimizer")          # optimizer
SCHEDULER_REGISTRY = Registry("scheduler")          # scheduler

MODEL_REGISTRY = Registry("comp_model")             # model
ENCODER_REGISTRY = Registry("encoder")              # encoder
DECODER_REGISTRY = Registry("decoder")              # decoder
MOTIONNET_REGISTRY = Registry("motion_net")             # motion estimation
RECON_GENE_REGISTRY = Registry("recon_gene")            # recon generation 
HYPERENCODER_REGISTRY = Registry("hyperencoder")      # hyperencoder
HYPERDECODER_REGISTRY = Registry("hyperdecoder")        # hyperdecoder
CONTEXTMODEL_REGISTRY = Registry("context_model")       # context model
ENTROPYMODEL_REGISTRY = Registry("entropy_model")       # entropy model
DISCRIMINATOR_REGISTRY = Registry("discriminator")     # discriminator
CONDITION_REGISTRY = Registry("condition")             # condition


DATASET_REGISTRY = Registry("dataset")              # dataset
LOSS_REGISTRY = Registry("loss")                     # loss 
METRIC_REGISTRY = Registry("metric")                 # metric
