// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "itt.hpp"
#include "openvino/util/log.hpp"

#include "low_precision/gather.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace pass {
namespace low_precision {

namespace {

std::shared_ptr<opset1::Constant> gatherDeqConstant(
    const std::shared_ptr<ov::Node> &gather,
    const std::shared_ptr<ov::Node> &dequantizationConstant) {
    auto constant = ov::as_type_ptr<ov::opset1::Constant>(dequantizationConstant);
    auto constantShape = constant->get_shape();
    if (shape_size(constantShape) == 1ul) {
        return NetworkHelper::toScalar(constant);
    }

    const auto rank = gather->get_input_partial_shape(0).size();
    if (rank != constantShape.size()) {
        // case when constShape without batch
        while ((constantShape.size() > 1) && (constantShape.size() < rank)) {
            constantShape.insert(constantShape.begin(), 1);
        }
        const auto newConstant = fold<ov::opset1::Broadcast>(
            constant,
            ov::opset1::Constant::create(ov::element::i32, { constantShape.size() }, constantShape));
        constant = ov::as_type_ptr<ov::opset1::Constant>(newConstant);
    }

    const int64_t axis = ov::as_type_ptr<opset1::Constant>(gather->get_input_node_shared_ptr(2))->cast_vector<int64_t>()[0];
    const size_t normalizedAxis =
        ov::util::try_normalize_axis(axis, gather->get_input_partial_shape(0).rank(), *gather);

    // Dequantization channel matches with gather axis
    if (constantShape[normalizedAxis] != 1ul) {
        const auto gather1 = ov::as_type_ptr<ov::opset1::Gather>(gather);
        if (gather1) {
            const auto output = fold<ov::opset1::Gather>(
                constant,
                gather1->input_value(1),
                gather1->input_value(2));
            constant = ov::as_type_ptr<opset1::Constant>(NetworkHelper::toScalarIfPossible(output));
        }

        const auto gather7 = ov::as_type_ptr<ov::opset7::Gather>(gather);
        if (gather7) {
            const auto output = fold<ov::opset7::Gather>(
                constant,
                gather7->input_value(1),
                gather7->input_value(2),
                gather7->get_batch_dims());
            constant = ov::as_type_ptr<opset1::Constant>(NetworkHelper::toScalarIfPossible(output));
        }

        const auto gather8 = ov::as_type_ptr<ov::opset8::Gather>(gather);
        if (gather8) {
            const auto output = fold<ov::opset8::Gather>(
                constant,
                gather8->input_value(1),
                gather8->input_value(2),
                gather8->get_batch_dims());
            constant = ov::as_type_ptr<opset1::Constant>(NetworkHelper::toScalarIfPossible(output));
        }
    }
    return constant;
}

}  // namespace

GatherTransformation::GatherTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(GatherTransformation);
    auto gather = pattern::wrap_type<opset1::Gather, opset7::Gather, opset8::Gather>({ pattern::wrap_type<opset1::Multiply>(),
                                                        pattern::any_input(),
                                                        pattern::any_input() });

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gather, matcher_name);
    this->register_matcher(m, callback);
}

bool GatherTransformation::transform(TransformationContext& context, ov::pass::pattern::Matcher &m) {
    auto node = m.get_match_root();
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> gather = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(gather, defaultPrecisions);

    if (dequantization.multiply != nullptr) {
        const auto newConstant = gatherDeqConstant(gather, dequantization.multiplyConstant);
        replace_node(dequantization.multiplyConstant, newConstant);
    }
    if (dequantization.subtract != nullptr) {
        const auto newConstant = gatherDeqConstant(gather, dequantization.subtractConstant);
        replace_node(dequantization.subtractConstant, newConstant);
    }

    const auto newOperation = moveDequantizationAfter(context, gather, NetworkHelper::getDequantization(gather, defaultPrecisions));

    OPENVINO_DEBUG("LPT: done: ", newOperation);
    return true;
}

bool GatherTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    auto dequantization = NetworkHelper::getDequantization(operation, defaultPrecisions);
    if (dequantization.empty()) {
        return false;
    }

    const auto isScalar = [&] {
        if (dequantization.multiply != nullptr) {
            if (!NetworkHelper::isScalarLike(dequantization.multiplyConstant)) {
                return false;
            }
        }
        if (dequantization.subtract != nullptr) {
            if (!NetworkHelper::isScalarLike(dequantization.subtractConstant)) {
                return false;
            }
        }
        return true;
    }();
    if (isScalar) {
        return true;
    }

    // If dequantization constant is not scalar, Gather axis must be constant.
    // If the Gather axis matches with dequantization channel, the Gather indices
    // must be constant and have 0D or 1D shape so we can do folding.
    const auto axisConstant = ov::as_type_ptr<opset1::Constant>(operation->get_input_node_shared_ptr(2));
    if (axisConstant == nullptr) {
        return false;
    }

    if (operation->get_input_partial_shape(0).rank().is_dynamic()) {
        return false;
    }
    const auto canBeFolded = [&](const std::shared_ptr<ov::Node> dequantizationConstant) {
        auto constantShape = dequantizationConstant->get_shape();
        const auto rank = operation->get_input_partial_shape(0).size();
        if (rank != constantShape.size()) {
            while ((constantShape.size() > 1) && (constantShape.size() < rank)) {
                constantShape.insert(constantShape.begin(), 1);
            }
        }
        const int64_t axis = axisConstant->cast_vector<int64_t>()[0];
        const size_t normalizedAxis =
            ov::util::try_normalize_axis(axis, operation->get_input_partial_shape(0).rank(), *operation);

        if (constantShape[normalizedAxis] != 1ul) {
            const auto indicesConstant = ov::as_type_ptr<opset1::Constant>(operation->get_input_node_shared_ptr(1));
            if (indicesConstant == nullptr)
                return false;
            const auto indicesShape = indicesConstant->get_shape();
            if (indicesShape.size() != 0 && indicesShape.size() != 1) {
                return false;
            }
        }
        return true;
    };

    if ((dequantization.multiply && !canBeFolded(dequantization.multiplyConstant)) ||
        (dequantization.subtract && !canBeFolded(dequantization.subtractConstant))) {
        return false;
    }
    return true;
}

bool GatherTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
