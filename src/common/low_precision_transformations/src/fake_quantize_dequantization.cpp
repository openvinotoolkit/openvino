// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <memory>

#include "openvino/opsets/opset1.hpp"
#include "low_precision/network_helper.hpp"

#include "low_precision/common/fake_quantize_dequantization.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"

namespace ov {
namespace pass {
namespace low_precision {

FakeQuantizeDequantization::FakeQuantizeDequantization() {}

FakeQuantizeDequantization::FakeQuantizeDequantization(
    const Output<Node>& data,
    const std::shared_ptr<opset1::Convert>& convert,
    const std::shared_ptr<opset1::Subtract>& subtract,
    const std::shared_ptr<ov::opset1::Convert>& subtractConvert,
    const std::shared_ptr<ov::opset1::Constant>& subtractConstant,
    const std::shared_ptr<opset1::Multiply>& multiply,
    const std::shared_ptr<ov::opset1::Constant>& multiplyConstant) :
    data(data),
    convert(convert),
    subtract(subtract),
    subtractConvert(subtractConvert),
    subtractConstant(subtractConstant),
    multiply(multiply),
    multiplyConstant(multiplyConstant) {
    // for most node with layout NC, NCHW, NCDWH, index of channel dimension is 1
    channelDimIndex = 1ul;

    const auto rank = data.get_partial_shape().rank();
    if (rank.is_static()) {
        std::string data_src_type = data.get_node()->get_type_name();
        if (data_src_type == "MatMul" && data.get_index() == 0) {
            // for MatMul, index of channel dimension is the last one
            channelDimIndex = static_cast<size_t>(rank.get_length()) - 1;
        } else if (rank.get_length() == 1) {
            // special 1D case: C
            channelDimIndex = 0ul;
        }
    }
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

ov::element::Type FakeQuantizeDequantization::getPrecision() const {
    if (multiply != nullptr) {
        return is_type<ov::opset1::Constant>(multiply->get_input_node_ptr(0)) ?
            multiply->get_input_element_type(1) :
            multiply->get_input_element_type(0);
    }

    if (subtract != nullptr) {
        return is_type<ov::opset1::Constant>(subtract->get_input_node_ptr(0)) ?
            subtract->get_input_element_type(1) :
            subtract->get_input_element_type(0);
    }

    THROW_IE_LPT_EXCEPTION_BASE << "dequantization is empty";
}

bool FakeQuantizeDequantization::isPerTensor() const {
    if (multiplyConstant == nullptr) {
        THROW_IE_LPT_EXCEPTION_BASE << "multiply constant can not be empty";
    }

    const std::vector<float>& scales = multiplyConstant->cast_vector<float>();
    if (scales.size() != 1ull) {
        return false;
    }

    if (subtractConstant != nullptr) {
        const std::vector<float>& scales = subtractConstant->cast_vector<float>();
        if (scales.size() != 1ull) {
            return false;
        }
    }

    return true;
}

bool FakeQuantizeDequantization::checkShape(const std::shared_ptr<ov::Node>& elementwise) {
    std::shared_ptr<ov::opset1::Convert> convert;
    std::shared_ptr<ov::opset1::Constant> constant;
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
        for (size_t i = 0; i < inPShape.size(); ++i) {
            if (inPShape[i] != outPShape[i] && !inPShape[i].is_dynamic()) {
                return false;
            }
        }
    }

    return true;
}

// check if elementwise operation inside dequantization subgraph satisfy per-tensor/per-OC broadcast requirement
bool FakeQuantizeDequantization::checkElementwise(const std::shared_ptr<ov::Node>& dequantizationElementwise) const {
    std::shared_ptr<ov::opset1::Convert> convert;
    std::shared_ptr<ov::opset1::Constant> constant;
    FakeQuantizeDequantization::fillDequantizationParams(dequantizationElementwise, convert, constant);

    if (constant == nullptr) {
        return false;
    }

    const ov::Shape constShape = constant->get_shape();
    if ((constShape.size() > 5ul)) {
        return false;
    }

    // scalar-like const tensor is broadcastable to any shape of data
    if (ov::shape_size(constShape) == 1) {
        return true;
    }

    const auto partialShape = dequantizationElementwise->get_output_partial_shape(0);
    if (partialShape.rank().is_dynamic()) {
        return false;
    }

    auto dimc = channelDimIndex;

    const auto channelsDimension = partialShape[dimc];
    if (channelsDimension.is_dynamic()) {
        return false;
    }

    const size_t channelsShapeVal = channelsDimension.get_length();

    // special case: 1D const tensor is considered to be per-channel w/o comparing actual shapes using broadcast rules
    // as long as the number of elements matches channel dimension.
    if (constShape.size() == 1ul) {
        return constShape[0] == channelsShapeVal;
    }

    auto checkConstShape = [&constShape, &channelsShapeVal] (const size_t chDimIdx) {
        for (size_t i = 0ul; i < constShape.size(); ++i) {
            auto curDim = constShape[i];
            if (curDim == 1ul)
                continue;
            if (i == chDimIdx && curDim == channelsShapeVal)
                continue;
            return false;
        }
        return true;
    };

    const size_t rank = partialShape.rank().get_length();
    if (constShape.size() == rank) {
        // special case: ND const tensor with N matches data rank
        // element-wise comparing works under any broadcast rules.
        return checkConstShape(dimc);
    } else if (constShape.size() < rank) {
        // rank mismatch, we have to apply broadcast rules to align dimensions, all dequantization nodes are constructed
        // by LPT itself, thus should has default NUMPY type
        if (dequantizationElementwise->get_autob() != ov::op::AutoBroadcastType::NUMPY)
            return false;
        // the prepended dimensions are all 1 and can be skipped;
        // derive index of channel dimension in const tensor after right aligned
        if (dimc < rank - constShape.size())
            return false;
        return checkConstShape(dimc - (rank - constShape.size()));
    }

    return false;
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
    const std::shared_ptr<ov::Node>& elementwise,
    std::shared_ptr<ov::opset1::Convert>& convert,
    std::shared_ptr<ov::opset1::Constant>& constant) {
    auto fill = [](
        const std::shared_ptr<ov::Node>& elementwise,
        const size_t branchIndex,
        std::shared_ptr<ov::opset1::Convert>& convert,
        std::shared_ptr<ov::opset1::Constant>& constant) {
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
    const std::shared_ptr<ov::Node>& elementwise,
    std::shared_ptr<ov::opset1::Constant>& constant) {
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
}  // namespace ov
