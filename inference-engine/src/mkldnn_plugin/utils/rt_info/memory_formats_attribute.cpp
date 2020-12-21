// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "memory_formats_attribute.hpp"

namespace ngraph {

template class ngraph::MLKDNNMemoryFormatsHelper<MLKDNNInputMemoryFormats>;
constexpr VariantTypeInfo VariantWrapper<MLKDNNInputMemoryFormats>::type_info;

std::string getMLKDNNInputMemoryFormats(const std::shared_ptr<ngraph::Node> & node) {
    return MLKDNNMemoryFormatsHelper<MLKDNNInputMemoryFormats>::getMemoryFormats(node);
}

template class ngraph::MLKDNNMemoryFormatsHelper<MLKDNNOutputMemoryFormats>;
constexpr VariantTypeInfo VariantWrapper<MLKDNNOutputMemoryFormats>::type_info;

std::string getMLKDNNOutputMemoryFormats(const std::shared_ptr<ngraph::Node> & node) {
    return MLKDNNMemoryFormatsHelper<MLKDNNOutputMemoryFormats>::getMemoryFormats(node);
}

}  // namespace ngraph
