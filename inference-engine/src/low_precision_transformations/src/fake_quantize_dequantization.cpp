// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include <ngraph/opsets/opset1.hpp>

#include "low_precision/common/fake_quantize_dequantization.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

FakeQuantizeDequantization::FakeQuantizeDequantization() {}

FakeQuantizeDequantization::FakeQuantizeDequantization(
    Output<Node> data,
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
    if ((convert != nullptr) && (convert->get_output_target_inputs(0).size() > 1ul)) {
        return true;
    }

    if ((subtract != nullptr) && (subtract->get_output_target_inputs(0).size() > 1ul)) {
        return true;
    }

    if ((multiply != nullptr) && (multiply->get_output_target_inputs(0).size() > 1ul)) {
        return true;
    }

    return false;
}

bool FakeQuantizeDequantization::isLowPrecision() const {
    return (data.get_element_type() == element::i8) || (data.get_element_type() == element::u8);
}

bool FakeQuantizeDequantization::checkElementwise(const std::shared_ptr<ngraph::Node>& dequantizationElementwise) {
    const ngraph::PartialShape partialShape = dequantizationElementwise->get_input_partial_shape(0);
    if (partialShape.is_dynamic()) {
        return false;
    }

    std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(dequantizationElementwise->get_input_node_shared_ptr(1));
    if (constant == nullptr) {
        constant = as_type_ptr<opset1::Constant>(dequantizationElementwise->get_input_node_shared_ptr(0));
    }
    if (constant == nullptr) {
        THROW_IE_LPT_EXCEPTION(*dequantizationElementwise) << "unexpected operation type " <<
            dequantizationElementwise->get_type_info().name << " on the second branch";
    }

    const ngraph::Shape constShape = constant->get_output_shape(0);
    if ((constShape.size() > 5ul)) {
        return false;
    }

    if ((constShape.size() <= 1ul) || (std::all_of(constShape.begin(), constShape.end(), [](const size_t value) { return value == 1ul; }))) {
        return true;
    }

    const ngraph::Shape shape = partialShape.to_shape();
    if (constShape.size() == shape.size()) {
        if ((constShape[0] != 1ul) || (constShape[1] != shape[1])) {
            return false;
        }
        for (size_t i = 2ul; i < constShape.size(); ++i) {
            if (constShape[i] != 1ul) {
                return false;
            }
        }
    } else if (constShape.size() == (shape.size() - 1)) {
        if (constShape[0] != shape[1]) {
            return false;
        }
        for (size_t i = 1ul; i < constShape.size(); ++i) {
            if (constShape[i] != 1ul) {
                return false;
            }
        }
    } else {
        return false;
    }

    return true;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
