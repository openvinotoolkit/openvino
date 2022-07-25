// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/ov_node.h"

#include "common.h"

ov_status_e ov_node_get_any_name(ov_output_const_node_t* node, char** tensor_name) {
    if (!node || !tensor_name) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        *tensor_name = str_to_char_array(node->object->get_any_name());
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_any_name_by_index(ov_output_node_list_t* nodes, size_t idx, char** tensor_name) {
    if (!nodes || !tensor_name || idx >= nodes->num) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        *tensor_name = str_to_char_array(nodes->output_nodes[idx].object->get_any_name());
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_shape(ov_output_const_node_t* node, ov_shape_t* tensor_shape) {
    if (!node || !tensor_shape) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        auto shape = node->object->get_shape();
        if (shape.size() > MAX_DIMENSION) {
            return ov_status_e::INVALID_PARAM;
        }
        tensor_shape->rank = shape.size();
        std::copy_n(shape.begin(), shape.size(), tensor_shape->dims);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_shape_by_index(ov_output_node_list_t* nodes, size_t idx, ov_shape_t* tensor_shape) {
    if (!nodes || idx >= nodes->num || !tensor_shape) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        auto shape = nodes->output_nodes[idx].object->get_shape();
        if (shape.size() > MAX_DIMENSION) {
            return ov_status_e::GENERAL_ERROR;
        }
        tensor_shape->rank = shape.size();
        std::copy_n(shape.begin(), shape.size(), tensor_shape->dims);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_partial_shape_by_index(ov_output_node_list_t* nodes,
                                               size_t idx,
                                               ov_partial_shape_t** partial_shape) {
    if (!nodes || idx >= nodes->num || !partial_shape) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        std::unique_ptr<ov_partial_shape_t> _partial_shape(new ov_partial_shape_t);
        auto shape = nodes->output_nodes[idx].object->get_partial_shape();

        _partial_shape->rank = shape.rank();
        auto iter = shape.begin();
        for (; iter != shape.end(); iter++)
            _partial_shape->dims.emplace_back(*iter);
        *partial_shape = _partial_shape.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_element_type_by_index(ov_output_node_list_t* nodes,
                                              size_t idx,
                                              ov_element_type_e* tensor_type) {
    if (!nodes || idx >= nodes->num) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        auto type = (ov::element::Type_t)nodes->output_nodes[idx].object->get_element_type();
        *tensor_type = (ov_element_type_e)type;
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_element_type(ov_output_const_node_t* node, ov_element_type_e* tensor_type) {
    if (!node || !tensor_type) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        auto type = (ov::element::Type_t)node->object->get_element_type();
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
