# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ....graph.model_utils import get_nodes_by_type, models_union


def model_disjoint_union(first_model, second_model):
    first_model_relabeled = relabel_nodes(first_model)
    second_model_relabeled = relabel_nodes(second_model, first_label=first_model.number_of_nodes())
    composite_model = models_union(first_model_relabeled, second_model_relabeled)
    return composite_model


def relabel_nodes(model, first_label=0):
    N = model.number_of_nodes() + first_label
    mapping = dict(zip(model.node_labels(), [str(key) for key in range(first_label, N)]))
    model.relabel_nodes(mapping)
    return model


def get_composite_model(model, quantized_model, quantized_suffix='_quantized'):
    composite_model = model_disjoint_union(model, quantized_model)
    fq_inputs = [
        node
        for node in get_nodes_by_type(composite_model, ['Parameter'], recursively=False)
        if quantized_suffix not in node.name
    ]
    q_inputs = [
        node
        for node in get_nodes_by_type(composite_model, ['Parameter'], recursively=False)
        if quantized_suffix in node.name
    ]
    for fp_input_node, q_input_node in zip(fq_inputs, q_inputs):
        for port in q_input_node.out_ports().values():
            port.get_connection().set_source(fp_input_node.out_port(0))
    composite_model.clean_up()
    composite_model.meta_data = model.meta_data
    return composite_model
