// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <low_precision/network_helper.hpp>

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

bool FakeQuantizeDequantization::empty() const noexcept {
    return (subtract == nullptr) && (multiply == nullptr);
}

bool FakeQuantizeDequantization::multiplyHasZeroOrDenormal() const {
    if (multiply == nullptr) {
        return false;
    }

    std::shared_ptr<opset1::Constant> multiplyConstant = ov::as_type_ptr<opset1::Constant>(multiply->get_input_node_shared_ptr(1));
    if (multiplyConstant == nullptr) {
        multiplyConstant = ov::as_type_ptr<opset1::Constant>(multiply->get_input_node_shared_ptr(0));
    }
    if (multiplyConstant == nullptr) {
        return false;
    }

    auto const values = multiplyConstant->cast_vector<float>();
    return std::any_of(values.begin(), values.end(), [](const float value) { return (value == 0.f) || (std::abs(value) < 1.e-32); });
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
    return DataPrecision::isSupported(data.get_element_type());
}

bool FakeQuantizeDequantization::checkShape(const std::shared_ptr<ngraph::Node>& elementwise) {
    std::shared_ptr<ngraph::opset1::Convert> convert;
    std::shared_ptr<ngraph::opset1::Constant> constant;
    const int branchIndex = FakeQuantizeDequantization::fillDequantizationParams(elementwise, convert, constant);
    if (branchIndex == -1) {
        return true;
    }

    const auto inPShape = elementwise->get_input_partial_shape(branchIndex == 1 ? 0 : 1);
    const auto outPShape = elementwise->get_output_partial_shape(0);
    if (inPShape.rank() != outPShape.rank()) {
        return false;
    }

    if (!inPShape.rank().is_dynamic()) {
        for (int i = 0; i < inPShape.size(); ++i) {
            if (inPShape[i] != outPShape[i] && !inPShape[i].is_dynamic()) {
                return false;
            }
        }
    }

    return true;
}

bool FakeQuantizeDequantization::checkElementwise(const std::shared_ptr<ngraph::Node>& dequantizationElementwise) {
    std::shared_ptr<ngraph::opset1::Convert> convert;
    std::shared_ptr<ngraph::opset1::Constant> constant;
    FakeQuantizeDequantization::fillDequantizationParams(dequantizationElementwise, convert, constant);

    if (constant == nullptr) {
        return false;
    }

    const ngraph::Shape constShape = constant->get_shape();
    if ((constShape.size() > 5ul)) {
        return false;
    }

    if (ngraph::shape_size(constShape) == 1) {
        return true;
    }

    const auto partialShape = dequantizationElementwise->get_output_partial_shape(0);
    if (partialShape.rank().is_dynamic()) {
        return false;
    }

    const auto channelsDimension = partialShape[partialShape.size() > 1ul ? 1ul : 0ul];
    if (channelsDimension.is_dynamic()) {
        return false;
    }

    const size_t channelsShapeVal = channelsDimension.get_length();
    if (constShape.size() == 1ul) {
        return constShape[0] == channelsShapeVal;
    }

    const size_t rank = partialShape.rank().get_length();
    if (constShape.size() == rank) {
        if ((constShape[0] != 1ul) || (constShape[1] != channelsShapeVal)) {
            return false;
        }
        for (size_t i = 2ul; i < constShape.size(); ++i) {
            if (constShape[i] != 1ul) {
                return false;
            }
        }
    } else if (constShape.size() == (rank - 1)) {
        if (constShape[0] != channelsShapeVal) {
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

std::shared_ptr<Node> FakeQuantizeDequantization::copyWithNewInput(const std::shared_ptr<Node>& input) const {
    auto lastNode = input;
    if (convert) {
        lastNode = convert->copy_with_new_inputs({lastNode});
    }
    if (subtract) {
        std::shared_ptr<Node> input1 = nullptr;
        if (subtractConvert) {
            input1 = subtractConvert;
        } else {
            input1 = subtractConstant;
        }
        lastNode = subtract->copy_with_new_inputs({lastNode, input1});
    }
    if (multiply) {
        lastNode = multiply->copy_with_new_inputs({lastNode, multiplyConstant});
    }
    return lastNode;
}

int FakeQuantizeDequantization::fillDequantizationParams(
    const std::shared_ptr<ngraph::Node>& elementwise,
    std::shared_ptr<ngraph::opset1::Convert>& convert,
    std::shared_ptr<ngraph::opset1::Constant>& constant) {
    auto fill = [](
        const std::shared_ptr<ngraph::Node>& elementwise,
        const size_t branchIndex,
        std::shared_ptr<ngraph::opset1::Convert>& convert,
        std::shared_ptr<ngraph::opset1::Constant>& constant) {
        convert = ov::as_type_ptr<opset1::Convert>(elementwise->get_input_node_shared_ptr(branchIndex));
        if (convert != nullptr) {
            constant = convert->get_destination_type().is_real() ?
                ov::as_type_ptr<opset1::Constant>(convert->get_input_node_shared_ptr(0)) :
                nullptr;
        } else {
            constant = elementwise->get_input_element_type(branchIndex).is_real() ?
                ov::as_type_ptr<opset1::Constant>(elementwise->get_input_node_shared_ptr(branchIndex)) :
                nullptr;
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
    std::shared_ptr<ngraph::opset1::Constant>& constant) {
    constant = elementwise->get_input_element_type(1ul).is_real() ?
        ov::as_type_ptr<opset1::Constant>(elementwise->get_input_node_shared_ptr(1ul)) :
        nullptr;

    if (constant != nullptr) {
        return 1;
    }

    constant = elementwise->get_input_element_type(0ul).is_real() ?
        ov::as_type_ptr<opset1::Constant>(elementwise->get_input_node_shared_ptr(0ul)) :
        nullptr;

    if (constant != nullptr) {
        return 0;
    }

    return -1;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
