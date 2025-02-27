// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_formats_attribute.hpp"

#include "openvino/core/node.hpp"
#include "openvino/opsets/opset1.hpp"

namespace ov::intel_cpu {

InputMemoryFormats::~InputMemoryFormats() = default;

std::string getInputMemoryFormats(const std::shared_ptr<ov::Node>& node) {
    auto it_info = node->get_rt_info().find(InputMemoryFormats::get_type_info_static());
    if (it_info != node->get_rt_info().end()) {
        if (it_info->second.is<InputMemoryFormats>()) {
            return it_info->second.as<InputMemoryFormats>().to_string();
        }
    }
    return {};
}

OutputMemoryFormats::~OutputMemoryFormats() = default;

std::string getOutputMemoryFormats(const std::shared_ptr<ov::Node>& node) {
    auto it_info = node->get_rt_info().find(OutputMemoryFormats::get_type_info_static());
    if (it_info != node->get_rt_info().end()) {
        if (it_info->second.is<OutputMemoryFormats>()) {
            return it_info->second.as<OutputMemoryFormats>().to_string();
        }
    }
    return {};
}

}  // namespace ov::intel_cpu
