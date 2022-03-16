// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_formats_attribute.hpp"

#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>

using namespace ngraph;

namespace ov {
namespace intel_cpu {

InputMemoryFormats::~InputMemoryFormats() = default;

std::string getInputMemoryFormats(const std::shared_ptr<ngraph::Node>& node) {
    auto it_info = node->get_rt_info().find(InputMemoryFormats::get_type_info_static());
    if (it_info != node->get_rt_info().end()) {
        if (it_info->second.is<InputMemoryFormats>()) {
            return it_info->second.as<InputMemoryFormats>().to_string();
        }
    }
    return {};
}

OutputMemoryFormats::~OutputMemoryFormats() = default;

std::string getOutputMemoryFormats(const std::shared_ptr<ngraph::Node>& node) {
    auto it_info = node->get_rt_info().find(OutputMemoryFormats::get_type_info_static());
    if (it_info != node->get_rt_info().end()) {
        if (it_info->second.is<OutputMemoryFormats>()) {
            return it_info->second.as<OutputMemoryFormats>().to_string();
        }
    }
    return {};
}

void cleanMemoryFormats(const std::shared_ptr<ngraph::Node>& node) {
    auto& rt_info = node->get_rt_info();
    if (rt_info.find(InputMemoryFormats::get_type_info_static()) != rt_info.end())
        rt_info.erase(InputMemoryFormats::get_type_info_static());
    if (rt_info.find(OutputMemoryFormats::get_type_info_static()) != rt_info.end())
        rt_info.erase(OutputMemoryFormats::get_type_info_static());
}
}   // namespace intel_cpu
}   // namespace ov
