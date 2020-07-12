// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/low_precision/common/fake_quantize_dequantization.hpp>
#include <memory>
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

FakeQuantizeDequantization::FakeQuantizeDequantization() {}

FakeQuantizeDequantization::FakeQuantizeDequantization(
    std::shared_ptr<Node> data,
    std::shared_ptr<opset1::Convert> convert,
    std::shared_ptr<opset1::Subtract> subtract,
    std::shared_ptr<opset1::Multiply> multiply) :
    data(data),
    convert(convert),
    subtract(subtract),
    multiply(multiply) {
}

bool FakeQuantizeDequantization::empty() const {
    return (convert == nullptr) && (subtract == nullptr) && (multiply == nullptr);
}

bool FakeQuantizeDequantization::isShared() const {
    if ((convert != nullptr) && (convert->get_output_inputs(0).size() > 1ul)) {
        return true;
    }

    if ((subtract != nullptr) && (subtract->get_output_inputs(0).size() > 1ul)) {
        return true;
    }

    if ((multiply != nullptr) && (multiply->get_output_inputs(0).size() > 1ul)) {
        return true;
    }

    return false;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
