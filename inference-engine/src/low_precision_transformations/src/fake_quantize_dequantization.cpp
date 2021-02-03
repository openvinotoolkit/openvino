// Copyright (C) 2020-2021 Intel Corporation
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
    const Output<Node>& data,
    const std::shared_ptr<opset1::Convert>& convert,
    const std::shared_ptr<opset1::Subtract>& subtract,
    const std::shared_ptr<ngraph::opset1::Convert>& subtractConvert,
    const std::shared_ptr<ngraph::opset1::Constant>& subtractConstant,
    const std::shared_ptr<opset1::Multiply>& multiply,
    const std::shared_ptr<ngraph::opset1::Constant>& multiplyConstant) :
    data(data),
    convert(convert),
    subtract(subtract),
    subtractConvert(subtractConvert),
    subtractConstant(subtractConstant),
    multiply(multiply),
    multiplyConstant(multiplyConstant) {
}

bool FakeQuantizeDequantization::empty() const {
    return (convert == nullptr) && (subtract == nullptr) && (multiply == nullptr);
}

bool FakeQuantizeDequantization::multiplyHasZero() const {
    if (multiply == nullptr) {
        return false;
    }

    std::shared_ptr<opset1::Constant> multiplyConstant = as_type_ptr<opset1::Constant>(multiply->get_input_node_shared_ptr(1));
    if (multiplyConstant == nullptr) {
        multiplyConstant = as_type_ptr<opset1::Constant>(multiply->get_input_node_shared_ptr(0));
    }
    if (multiplyConstant == nullptr) {
        return false;
    }

    auto const values = multiplyConstant->cast_vector<float>();
    return std::any_of(values.begin(), values.end(), [](const float value) { return value == 0.f; });
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

bool FakeQuantizeDequantization::checkShape(const std::shared_ptr<ngraph::Node>& elementwise) noexcept {
    std::shared_ptr<ngraph::opset1::Convert> convert;
    std::shared_ptr<ngraph::opset1::Constant> constant;
    const int branchIndex = FakeQuantizeDequantization::fillDequantizationParams(elementwise, convert, constant);
    if (branchIndex == -1) {
        return true;
    }

    if (elementwise->output(0).get_shape() != elementwise->get_input_shape(branchIndex == 1 ? 0 : 1)) {
        return false;
    }

    return true;
}

bool FakeQuantizeDequantization::checkElementwise(const std::shared_ptr<ngraph::Node>& dequantizationElementwise) {
    const ngraph::PartialShape partialShape = dequantizationElementwise->get_input_partial_shape(0);
    if (partialShape.is_dynamic()) {
        return false;
    }

    std::shared_ptr<ngraph::opset1::Convert> convert;
    std::shared_ptr<ngraph::opset1::Constant> constant;
    FakeQuantizeDequantization::fillDequantizationParams(dequantizationElementwise, convert, constant);

    if (constant == nullptr) {
        return false;
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

int FakeQuantizeDequantization::fillDequantizationParams(
    const std::shared_ptr<ngraph::Node>& elementwise,
    std::shared_ptr<ngraph::opset1::Convert>& convert,
    std::shared_ptr<ngraph::opset1::Constant>& constant) noexcept {
    auto fill = [](
        const std::shared_ptr<ngraph::Node>& elementwise,
        const size_t branchIndex,
        std::shared_ptr<ngraph::opset1::Convert>& convert,
        std::shared_ptr<ngraph::opset1::Constant>& constant) {
        convert = as_type_ptr<opset1::Convert>(elementwise->get_input_node_shared_ptr(branchIndex));
        if (convert != nullptr) {
            constant = as_type_ptr<opset1::Constant>(convert->get_input_node_shared_ptr(0));
        } else {
            constant = as_type_ptr<opset1::Constant>(elementwise->get_input_node_shared_ptr(branchIndex));
        }
    };

    fill(elementwise, 1ul, convert, constant);
    if (constant != nullptr) {
        return 1;
    }

    fill(elementwise, 0ul, convert, constant);
    if (constant != nullptr) {
        return 0;
    }

    return -1;
}

int FakeQuantizeDequantization::fillDequantizationParams(
    const std::shared_ptr<ngraph::Node>& elementwise,
    std::shared_ptr<ngraph::opset1::Constant>& constant) noexcept {
    constant = as_type_ptr<opset1::Constant>(elementwise->get_input_node_shared_ptr(1ul));
    if (constant != nullptr) {
        return 1;
    }

    constant = as_type_ptr<opset1::Constant>(elementwise->get_input_node_shared_ptr(0ul));
    if (constant != nullptr) {
        return 0;
    }

    return -1;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
