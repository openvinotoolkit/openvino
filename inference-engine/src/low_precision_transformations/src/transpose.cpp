// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/transpose.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::TransposeTransformation, "TransposeTransformation", 0);

TransposeTransformation::TransposeTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::Transpose>({ pattern::wrap_type<opset1::Multiply>(), pattern::wrap_type<opset1::Constant>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "TransposeTransformation");
    this->register_matcher(m, callback);
}

void transposeDequantizationConstant(std::shared_ptr<Node>& transpose) {
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(transpose);

    const Shape subtractShape = dequantization.subtract == nullptr ? Shape{} : dequantization.subtractConstant->get_shape();
    const Shape multiplyShape = dequantization.multiply == nullptr ? Shape{} : dequantization.multiplyConstant->get_shape();
    if ((subtractShape.empty() || (subtractShape.size() == 1ul)) && (multiplyShape.empty() || (multiplyShape.size() == 1ul))) {
        return;
    }

    if (dequantization.multiply->get_input_node_ptr(1)->get_output_shape(0).size() > 1ul) {
        auto transposeDeqConstant = [](
            std::shared_ptr<Node> dequantizationConstant,
            const PartialShape& transposeOutputShape,
            const std::shared_ptr<Node>& transposeConstant) -> std::shared_ptr<Node> {
            const auto dequantizationShape = dequantizationConstant->get_output_shape(0);
            if (dequantizationShape.empty() || (dequantizationShape.size() == 1ul)) {
                return nullptr;
            }

            if (dequantizationShape.size() != static_cast<size_t>(transposeOutputShape.rank().get_length())) {
                dequantizationConstant = fold<opset1::Unsqueeze>(
                    dequantizationConstant,
                    std::make_shared<opset1::Constant>(element::i32, Shape{ 1 }, std::vector<size_t>{0}));
            }
            return fold<opset1::Transpose>(dequantizationConstant, transposeConstant);
        };

        if (dequantization.subtract != nullptr) {
            auto constant = transposeDeqConstant(
                dequantization.subtractConstant,
                transpose->get_output_partial_shape(0),
                transpose->get_input_node_shared_ptr(1));
            if (constant != nullptr) {
                replace_node(
                    dequantization.subtract->get_input_node_shared_ptr(1),
                    constant);
            }
        }

        if (dequantization.multiply != nullptr) {
            auto constant = transposeDeqConstant(
                dequantization.multiplyConstant,
                transpose->get_output_partial_shape(0),
                transpose->get_input_node_shared_ptr(1));
            if (constant != nullptr) {
                replace_node(
                    dequantization.multiply->get_input_node_shared_ptr(1),
                    constant);
            }
        }
    }
}

bool TransposeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    std::shared_ptr<Node> transpose = m.get_match_root();
    if (!canBeTransformed(context, transpose)) {
        return false;
    }

    transpose = NetworkHelper::separateInStandaloneBranch(transpose);
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
        if (dequantization.subtractConstant != nullptr) {
            if (!NetworkHelper::isScalarLike(dequantization.subtractConstant)) {
                return false;
            }
        }
        if (dequantization.multiply != nullptr) {
            const auto mulConst = as_type_ptr<ngraph::op::v0::Constant>(dequantization.multiplyConstant);
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

    auto checkShape = [](const std::shared_ptr<opset1::Constant>& dequantizationConstant, const PartialShape& transposeOutputShape) -> bool {
        const auto dequantizationShape = dequantizationConstant->get_shape();
        const auto rank = transposeOutputShape.rank();
        if (rank.is_dynamic()) {
            return false;
        }

        const size_t rankValue = rank.get_length();
        if (dequantizationShape.empty() || (dequantizationShape.size() == 1ul) || (dequantizationShape.size() == rankValue)) {
            return true;
        }

        if (dequantizationShape.size() > rankValue) {
            return false;
        }

        return (rankValue - dequantizationShape.size()) == 1;
    };

    return
        !dequantization.empty() &&
        ((dequantization.subtract == nullptr) || checkShape(dequantization.subtractConstant, op->get_output_partial_shape(0))) &&
        ((dequantization.multiply == nullptr) || checkShape(dequantization.multiplyConstant, op->get_output_partial_shape(0)));
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
