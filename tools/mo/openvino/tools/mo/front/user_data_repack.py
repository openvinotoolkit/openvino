# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.extractor import user_data_repack
from openvino.tools.mo.graph.graph import Graph


class UserDataRepack(FrontReplacementPattern):
    enabled = True
    run_not_recursively = True

    def run_after(self):
        return []

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        argv = graph.graph['cmd_params']

        packed_user_shapes, packed_outputs, freeze_placeholder = user_data_repack(
            graph, argv.placeholder_shapes, argv.placeholder_data_types,
            argv.output, argv.freeze_placeholder_with_value)

        # save packed user shapes in arguments since nodes names and their ports
        # will be required to compose placeholder names with custom types
        # for MOCLegacyTransformations
        argv.packed_user_shapes = packed_user_shapes

        graph.graph['user_shapes'] = packed_user_shapes
        graph.graph['packed_outputs'] = packed_outputs
        graph.graph['freeze_placeholder'] = freeze_placeholder

        if argv.inputs_list is not None and isinstance(argv.inputs_list, list) and len(argv.inputs_list) > 0:
            graph.inputs_order = argv.inputs_list
        if argv.output is not None and isinstance(argv.output, list) and len(argv.output) > 0:
            graph.outputs_order = argv.output

        inputs = list(packed_user_shapes.keys()) \
            if packed_user_shapes is not None and isinstance(packed_user_shapes, dict) else None
        graph.graph['inputs'] = inputs  # save user defined inputs for other extensions
