// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <vector>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include <low_precision/common/fake_quantize_dequantization.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

class GetDequantizationFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        bool isConvert, bool isSubtract, size_t subDataInput, size_t mulDataInput);

    static std::shared_ptr<ngraph::Function> getReference(
        ngraph::pass::low_precision::FakeQuantizeDequantization dequantization);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
