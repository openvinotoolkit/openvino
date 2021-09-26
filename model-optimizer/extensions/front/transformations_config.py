# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.custom_replacement_registry import CustomReplacementRegistry
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.replacement import FrontReplacementFromConfigFileOp
from mo.graph.graph import Graph


class TransformationsConfig(FrontReplacementPattern):
    enabled = True
    # do not run this transformation recursively otherwise transformations which are enabled with a configuration file
    # will be registered multiple times
    run_not_recursively = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].transformations_config is not None]

    def run_before(self):
        from extensions.front.pass_separator import FrontStart
        return [FrontStart]

    def run_after(self):
        from extensions.load.loader import LoadFinish
        return [LoadFinish]

    def find_and_replace_pattern(self, graph: Graph):
        argv = graph.graph['cmd_params']
        transformations_config = argv.transformations_config
        registry = CustomReplacementRegistry()
        registry.add_custom_replacement_description_from_config(transformations_config)

        # automatically generate sub-classes for custom replacements that replace sub-graph with a single node
        for replacement_desc in registry.get_all_replacements_descriptions():
            if replacement_desc.has('op'):
                transform = type('FrontReplacementFromConfigFileOp' + replacement_desc.op,
                                 (FrontReplacementFromConfigFileOp,),
                                 {'replacement_id': replacement_desc.id})
                transform().find_and_replace_pattern(graph)
