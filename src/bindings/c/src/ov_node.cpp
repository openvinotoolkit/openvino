// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_node.h"

#include "common.h"

ov_status_e ov_node_get_shape(ov_output_const_node_t* node, ov_shape_t* tensor_shape) {
    if (!node || !tensor_shape) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto shape = node->object->get_shape();
        ov_shape_create(shape.size(), nullptr, tensor_shape);
        std::copy_n(shape.begin(), shape.size(), tensor_shape->dims);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_list_get_shape_by_index(const ov_output_node_list_t* nodes, size_t idx, ov_shape_t* tensor_shape) {
    if (!nodes || idx >= nodes->size || !tensor_shape) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto shape = nodes->output_nodes[idx].object->get_shape();
        ov_shape_create(shape.size(), nullptr, tensor_shape);
        std::copy_n(shape.begin(), shape.size(), tensor_shape->dims);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_list_get_any_name_by_index(const ov_output_node_list_t* nodes, size_t idx, char** tensor_name) {
    if (!nodes || !tensor_name || idx >= nodes->size) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        *tensor_name = str_to_char_array(nodes->output_nodes[idx].object->get_any_name());
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_list_get_partial_shape_by_index(const ov_output_node_list_t* nodes,
                                                    size_t idx,
                                                    ov_partial_shape_t* partial_shape) {
    if (!nodes || idx >= nodes->size || !partial_shape) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto pshape = nodes->output_nodes[idx].object->get_partial_shape();
        auto rank = pshape.rank();

        partial_shape->rank.min = rank.get_min_length();
        partial_shape->rank.max = rank.get_max_length();
        if (rank.is_dynamic()) {
            partial_shape->dims = nullptr;
        } else {
            auto size = rank.get_length();
            if (static_cast<size_t>(size) != pshape.size()) {
                return ov_status_e::PARAMETER_MISMATCH;
            }
            std::unique_ptr<ov_dimension_t> _dimensions(new ov_dimension_t[size]);
            partial_shape->dims = _dimensions.release();
            auto iter = pshape.begin();
            for (auto i = 0; iter != pshape.end(); iter++, i++) {
                partial_shape->dims[i].min = iter->get_min_length();
                partial_shape->dims[i].max = iter->get_max_length();
            }
        }
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_list_get_element_type_by_index(const ov_output_node_list_t* nodes,
                                                   size_t idx,
                                                   ov_element_type_e* tensor_type) {
    if (!nodes || idx >= nodes->size) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto type = (ov::element::Type_t)nodes->output_nodes[idx].object->get_element_type();
        *tensor_type = (ov_element_type_e)type;
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_output_node_list_free(ov_output_node_list_t* output_nodes) {
    if (output_nodes) {
        if (output_nodes->output_nodes)
            delete[] output_nodes->output_nodes;
        output_nodes->output_nodes = nullptr;
    }
}

void ov_output_node_free(ov_output_const_node_t* output_node) {
    if (output_node)
        delete output_node;
}
