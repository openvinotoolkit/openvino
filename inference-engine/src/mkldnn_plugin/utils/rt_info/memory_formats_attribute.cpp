// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "memory_formats_attribute.hpp"

namespace ngraph {

template class ngraph::MemoryFormatsHelper<InputMemoryFormats>;
constexpr VariantTypeInfo VariantWrapper<InputMemoryFormats>::type_info;

std::string getInputMemoryFormats(const std::shared_ptr<ngraph::Node> & node) {
    return MemoryFormatsHelper<InputMemoryFormats>::getMemoryFormats(node);
}

template class ngraph::MemoryFormatsHelper<OutputMemoryFormats>;
constexpr VariantTypeInfo VariantWrapper<OutputMemoryFormats>::type_info;

std::string getOutputMemoryFormats(const std::shared_ptr<ngraph::Node> & node) {
    return MemoryFormatsHelper<OutputMemoryFormats>::getMemoryFormats(node);
}

}  // namespace ngraph
