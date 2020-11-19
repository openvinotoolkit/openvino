// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/transpose.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void TransposeTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Transpose>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
}

void transposeDequantizationConstant(std::shared_ptr<Node>& transpose) {
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(transpose);

    const Shape subtractShape = dequantization.subtract == nullptr ? Shape{} : dequantization.subtract->get_input_node_ptr(1)->get_output_shape(0);
    const Shape multiplyShape = dequantization.multiply == nullptr ? Shape{} : dequantization.multiply->get_input_node_ptr(1)->get_output_shape(0);
    if ((subtractShape.empty() || (subtractShape.size() == 1ul)) && (multiplyShape.empty() || (multiplyShape.size() == 1ul))) {
        return;
    }

    if (dequantization.multiply->get_input_node_ptr(1)->get_output_shape(0).size() > 1ul) {
        auto transposeConstant = [](
            std::shared_ptr<Node> dequantizationConstant,
            const Shape& transposeOutputShape,
            const std::shared_ptr<Node>& transposeConstant) -> std::shared_ptr<Node> {
            const auto dequantizationShape = dequantizationConstant->get_output_shape(0);
            if (dequantizationShape.empty() || (dequantizationShape.size() == 1ul)) {
                return nullptr;
            }

            if (dequantizationShape.size() != transposeOutputShape.size()) {
                dequantizationConstant = fold<opset1::Unsqueeze>(
                    dequantizationConstant,
                    std::make_shared<opset1::Constant>(element::i32, Shape{ 1 }, std::vector<size_t>{0}));
            }
            return fold<opset1::Transpose>(dequantizationConstant, transposeConstant);
        };

        if (dequantization.subtract != nullptr) {
            auto constant = transposeConstant(
                dequantization.subtract->get_input_node_shared_ptr(1),
                transpose->get_output_shape(0),
                transpose->get_input_node_shared_ptr(1));
            if (constant != nullptr) {
                replace_node(
                    dequantization.subtract->get_input_node_shared_ptr(1),
                    constant);
            }
        }

        if (dequantization.multiply != nullptr) {
            auto constant = transposeConstant(
                dequantization.multiply->get_input_node_shared_ptr(1),
                transpose->get_output_shape(0),
                transpose->get_input_node_shared_ptr(1));
            if (constant != nullptr) {
                replace_node(
                    dequantization.multiply->get_input_node_shared_ptr(1),
                    constant);
            }
        }
    }
}

bool TransposeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<Node> transpose = m.get_match_root();
    if (!canBeTransformed(context, transpose)) {
        return false;
    }

    transpose = separateInStandaloneBranch(transpose);
    transposeDequantizationConstant(transpose);
    moveDequantizationAfter(context, transpose, NetworkHelper::getDequantization(transpose, 0), false);
    return true;
}

bool TransposeTransformation::isPrecisionPreserved(std::shared_ptr<Node> op) const noexcept {
    return true;
}

bool TransposeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    const std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(op->get_input_node_shared_ptr(1));
    if (constant == nullptr) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op);
    const bool isPerTensor =  [&] {
        const auto sub = dequantization.subtract;
        const auto mul = dequantization.multiply;
        if (sub) {
            auto subConst = as_type_ptr<ngraph::op::v0::Constant>(sub->get_input_node_shared_ptr(1));
            if (!NetworkHelper::isScalarLike(subConst)) {
                return false;
            }
        }
        if (mul) {
            auto mulConst = as_type_ptr<ngraph::op::v0::Constant>(mul->get_input_node_shared_ptr(1));
            if (!NetworkHelper::isScalarLike(mulConst)) {
                return false;
            }
        }
        return true;
    }();

    const auto values = constant->cast_vector<float>();
    if (!isPerTensor) {
        if ((values.size() < 2ul) || (values[0] != 0) || (values[1] != 1)) {
            return false;
        }
    }

    auto checkConstant = [](const std::shared_ptr<Node>& dequantizationConstant, const Shape& transposeOutputShape) -> bool {
        const auto dequantizationShape = dequantizationConstant->get_output_shape(0);
        if (dequantizationShape.empty() || (dequantizationShape.size() == 1ul) || (dequantizationShape.size() == transposeOutputShape.size())) {
            return true;
        }

        if (dequantizationShape.size() > transposeOutputShape.size()) {
            return false;
        }

        return (transposeOutputShape.size() - dequantizationShape.size()) == 1;
    };

    return
        !dequantization.empty() &&
        ((dequantization.subtract == nullptr) || checkConstant(dequantization.subtract->get_input_node_shared_ptr(1), op->get_output_shape(0))) &&
        ((dequantization.multiply == nullptr) || checkConstant(dequantization.multiply->get_input_node_shared_ptr(1), op->get_output_shape(0)));
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
