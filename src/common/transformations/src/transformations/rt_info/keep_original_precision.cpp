// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/keep_original_precision.hpp"

ov::KeepOriginalPrecision::KeepOriginalPrecision(const std::shared_ptr<const ov::Node>& node) {
    size_t num_inputs = node->get_input_size();
    m_input_types.reserve(node->get_input_size());
    for (size_t i = 0; i < num_inputs; i++)
        m_input_types.push_back(node->get_input_element_type(i));

    size_t num_outputs = node->get_output_size();
    m_output_types.reserve(node->get_output_size());
    for (size_t i = 0; i < num_outputs; i++)
        m_output_types.push_back(node->get_output_element_type(i));
}

void ov::set_keep_original_precision_attribute(const std::shared_ptr<ov::Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[KeepOriginalPrecision::get_type_info_static()] = KeepOriginalPrecision(node);
}

void ov::remove_keep_original_precision_attribute(const std::shared_ptr<ov::Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(KeepOriginalPrecision::get_type_info_static());
}

bool ov::has_keep_original_precision_attribute(const std::shared_ptr<const ov::Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(KeepOriginalPrecision::get_type_info_static()) > 0;
}

const ov::KeepOriginalPrecision& ov::get_keep_original_precision_attribute(
    const std::shared_ptr<const ov::Node>& node) {
    const auto& rt_info = node->get_rt_info();
    const auto& value = rt_info.at(KeepOriginalPrecision::get_type_info_static());
    return value.as<ov::KeepOriginalPrecision>();
}
