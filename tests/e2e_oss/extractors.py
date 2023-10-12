import argparse

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.pipeline.common import get_ir_version
from openvino.tools.mo.utils import class_registration


def loader_pipeline(argv: argparse.Namespace):
    graph = Graph(cmd_params=argv, name=argv.model_name, ir_version=get_ir_version(argv))
    class_registration.apply_replacements(graph, [
        class_registration.ClassType.LOADER,
    ])
    return graph
