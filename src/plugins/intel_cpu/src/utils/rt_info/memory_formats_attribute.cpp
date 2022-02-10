// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_formats_attribute.hpp"

#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/variant.hpp>

using namespace ngraph;
using namespace ov;

MKLDNNInputMemoryFormats::~MKLDNNInputMemoryFormats() = default;

std::string ngraph::getMKLDNNInputMemoryFormats(const std::shared_ptr<ngraph::Node>& node) {
    auto it_info = node->get_rt_info().find(MKLDNNInputMemoryFormats::get_type_info_static());
    if (it_info != node->get_rt_info().end()) {
        if (it_info->second.is<MKLDNNInputMemoryFormats>()) {
            return it_info->second.as<MKLDNNInputMemoryFormats>().getMemoryFormats();
        }
    }
    return {};
}

MKLDNNOutputMemoryFormats::~MKLDNNOutputMemoryFormats() = default;

std::string ngraph::getMKLDNNOutputMemoryFormats(const std::shared_ptr<ngraph::Node>& node) {
    auto it_info = node->get_rt_info().find(MKLDNNOutputMemoryFormats::get_type_info_static());
    if (it_info != node->get_rt_info().end()) {
        if (it_info->second.is<MKLDNNOutputMemoryFormats>()) {
            return it_info->second.as<MKLDNNOutputMemoryFormats>().getMemoryFormats();
        }
    }
    return {};
}


