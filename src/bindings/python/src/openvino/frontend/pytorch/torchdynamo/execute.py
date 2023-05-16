from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from warnings import warn

import torch
import torch.overrides

from torch.fx import GraphModule
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.decoder import TorchFXPythonDecoder
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.torchdynamo.compile import openvino_compile
from openvino.runtime import Core, Type, PartialShape

from typing import Callable, Optional

from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx


DEFAULT_OPENVINO_PYTHON_CONFIG = MappingProxyType(
    {
        "use_python_fusion_cache": True,
        "allow_single_op_fusion": True,
    }
)

compiled_cache = {}
max_openvino_partitions = 0
partitioned_modules = {}

def execute(
    gm: GraphModule,
    *args,
    executor: str = "aten",
    executor_parameters: Optional[dict] = None,
):
    if executor == "openvino":
        return openvino_execute_partitioned(gm, *args, executor_parameters=executor_parameters)
    elif executor == "strictly_openvino":
        return openvino_execute(gm, *args, executor_parameters=executor_parameters)

    msg = "Received unexpected value for 'executor': {0}. Allowed values are: openvino, strictly_openvino.".format(
        executor
    )
    raise ValueError(msg)


import numpy as np

def openvino_execute(gm: GraphModule, *args, executor_parameters=None, partition_id):

    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    use_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    global compiled_cache

    if use_cache and (partition_id in compiled_cache):
        compiled = compiled_cache[partition_id]
    else:
        compiled = openvino_compile(gm, *args)
        compiled_cache[partition_id] = compiled

    flat_args, _ = tree_flatten(args)
    ov_inputs = [a.numpy() for a in flat_args]

    res = compiled(ov_inputs)

    results1 = [res[out] for out in compiled.outputs]

    if len(results1) == 1:
        results = torch.from_numpy(results1[0])
        return results
    else:
        for i in range(len(results1)):
            results1[i] = torch.from_numpy(results1[i])
        return results1


class OpenVINOGraphModule(torch.nn.Module):
    def __init__(self, gm, partition_id, use_python_fusion_cache):
        super().__init__()
        self.gm = gm
        self.partition_id = partition_id
        self.executor_parameters = {"use_python_fusion_cache": use_python_fusion_cache}

    def __call__(self, *args):
        # TODO: Revisit fallback logic to include permanent fallback for failing graphs.
        tmp_fallback = False
        for idx, input_data in enumerate(args): #subgraph.example_inputs):
            if len(input_data.shape) == 0:
                tmp_fallback = True;

        if tmp_fallback:
            return self.gm(*args)
        else:
            return openvino_execute(
                self.gm, *args, executor_parameters=self.executor_parameters, partition_id=self.partition_id
            )


def partition_graph(gm: GraphModule, use_python_fusion_cache: bool):
    partitioner = Partitioner()
    partitioned_graph = partitioner.make_partitions(gm)    
    global max_openvino_partitions
    partition_id = max_openvino_partitions
    for node in partitioned_graph.graph.nodes:
        # TODO: use a better way to identify fused submodule
        if node.op == "call_module" and "fused_" in node.name:
            openvino_submodule = getattr(partitioned_graph, node.name)
            partitioned_graph.delete_submodule(node.target)
            gm.add_submodule(
                node.target,
                OpenVINOGraphModule(openvino_submodule, partition_id, use_python_fusion_cache),
            )
            partition_id = partition_id + 1

    max_openvino_partitions = partition_id

    return partitioned_graph


def openvino_execute_partitioned(gm: GraphModule, *args, executor_parameters=None):
    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    global partitioned_modules
    
    allow_single_op_fusion = executor_parameters.get(
        "allow_single_op_fusion",
        DEFAULT_OPENVINO_PYTHON_CONFIG["allow_single_op_fusion"],
    )
    use_python_fusion_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )

    signature = str(id(gm))
    for idx, input_data in enumerate(args):
        signature = signature+"_"+str(idx)+":"+str(input_data.type())[6:]+":"+str(input_data.size())[11:-1].replace(" ","")

    if signature not in partitioned_modules:
        partitioned_modules[signature] = partition_graph(gm, use_python_fusion_cache=use_python_fusion_cache)
    return partitioned_modules[signature](*args)
