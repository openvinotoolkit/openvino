// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_formats_attribute.hpp"

#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/variant.hpp>

using namespace ngraph;
using namespace ov;

template class ov::MLKDNNMemoryFormatsHelper<MLKDNNInputMemoryFormats>;

std::string ngraph::getMLKDNNInputMemoryFormats(const std::shared_ptr<ngraph::Node>& node) {
    return MLKDNNMemoryFormatsHelper<MLKDNNInputMemoryFormats>::getMemoryFormats(node);
}

template class ov::MLKDNNMemoryFormatsHelper<MLKDNNOutputMemoryFormats>;

std::string ngraph::getMLKDNNOutputMemoryFormats(const std::shared_ptr<ngraph::Node>& node) {
    return MLKDNNMemoryFormatsHelper<MLKDNNOutputMemoryFormats>::getMemoryFormats(node);
}

