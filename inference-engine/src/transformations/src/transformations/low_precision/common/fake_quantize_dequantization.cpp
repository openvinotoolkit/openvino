// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/low_precision/common/fake_quantize_dequantization.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/multiply_add.hpp>

#include "transformations/low_precision/quantization_details.hpp"
#include "transformations/low_precision/common/ie_lpt_exception.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

FakeQuantizeDequantization::FakeQuantizeDequantization() {}

FakeQuantizeDequantization::FakeQuantizeDequantization(
    std::shared_ptr<Node> dataNode,
    std::shared_ptr<ngraph::opset1::Convert> convert,
    std::shared_ptr<ngraph::opset1::Subtract> subtract,
    std::shared_ptr<ngraph::opset1::Multiply> multiply) :
    dataNode(dataNode),
    convert(convert),
    subtract(subtract),
    multiply(multiply) {
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
