// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_formats_attribute.hpp"

#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/variant.hpp>

using namespace ngraph;
using namespace ov;

MLKDNNInputMemoryFormats::~MLKDNNInputMemoryFormats() = default;

std::string ngraph::getMLKDNNInputMemoryFormats(const std::shared_ptr<ngraph::Node>& node) {
    auto it_info = node->get_rt_info().find(MLKDNNInputMemoryFormatsAttr);
    if (it_info != node->get_rt_info().end()) {
        if (auto ptr = it_info->second.as<std::shared_ptr<MLKDNNInputMemoryFormats>>()) {
            return ptr->getMemoryFormats();
        }
    }
    return {};
}

MLKDNNOutputMemoryFormats::~MLKDNNOutputMemoryFormats() = default;

std::string ngraph::getMLKDNNOutputMemoryFormats(const std::shared_ptr<ngraph::Node>& node) {
    auto it_info = node->get_rt_info().find(MLKDNNOutputMemoryFormatsAttr);
    if (it_info != node->get_rt_info().end()) {
        if (auto ptr = it_info->second.as<std::shared_ptr<MLKDNNOutputMemoryFormats>>()) {
            return ptr->getMemoryFormats();
        }
    }
    return {};
}


