// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/preprocessing_attribute.hpp"

#include "openvino/core/node.hpp"

bool ov::is_preprocesing_node(const std::shared_ptr<ov::Node>& node) {
    return node->get_rt_info().count(PreprocessingAttribute::get_type_info_static());
}

void ov::set_is_preprocessing_node(std::shared_ptr<ov::Node> node) {
    node->get_rt_info().emplace(PreprocessingAttribute::get_type_info_static(), PreprocessingAttribute{});
}
