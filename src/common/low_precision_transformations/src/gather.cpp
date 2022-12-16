// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/gather.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

std::shared_ptr<opset1::Constant> gatherDeqConstant(
    const std::shared_ptr<ngraph::Node> gather,
    const std::shared_ptr<ngraph::Node> dequantizaitonConstant) {
    auto constant = ov::as_type_ptr<ngraph::opset1::Constant>(dequantizaitonConstant);
    auto constantShape = constant->get_shape();
    if (shape_size(constantShape) == 1ul) {
        return NetworkHelper::toScalar(constant);
    }

    const auto gatherShape = gather->get_input_partial_shape(0);
    const size_t rank = gatherShape.rank().get_length();
    if (rank != constantShape.size()) {
        ngraph::Shape newConstantShape;
        newConstantShape = constantShape;
        // case when constShape without batch
        if ((constantShape.size() > 1) &&
            (constantShape.size() < rank)) {
            newConstantShape.insert(newConstantShape.begin(), 1);
        }
        constantShape = newConstantShape;
        constant = ngraph::opset1::Constant::create(ngraph::element::i32, { newConstantShape.size() }, newConstantShape);
    }

    const int64_t axis = ov::as_type_ptr<opset1::Constant>(gather->get_input_node_shared_ptr(2))->cast_vector<int64_t>()[0];
    const size_t normalizedAxis = normalize_axis(gather->get_friendly_name(), axis, gather->get_input_partial_shape(0).rank());

    // Dequantization channel match with gather axis
    if (constantShape[normalizedAxis] != 1ul) {
        const auto gather7 = ov::as_type_ptr<ngraph::opset7::Gather>(gather);
        if (gather7) {
            const auto output = fold<ngraph::opset7::Gather>(
                constant,
                gather7->input_value(1),
                gather7->input_value(2),
                gather7->get_batch_dims());
            constant = ov::as_type_ptr<opset1::Constant>(NetworkHelper::toScalarIfPossible(output));
        }

        const auto gather8 = ov::as_type_ptr<ngraph::opset8::Gather>(gather);
        if (gather8) {
            const auto output = fold<ngraph::opset8::Gather>(
                constant,
                gather8->input_value(1),
                gather8->input_value(2),
                gather8->get_batch_dims());
            constant = ov::as_type_ptr<opset1::Constant>(NetworkHelper::toScalarIfPossible(output));
        }
    }
    return constant;
}

GatherTransformation::GatherTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(GatherTransformation);
    auto gather8 = pattern::wrap_type<opset8::Gather>({ pattern::wrap_type<opset1::Multiply>(),
                                                        pattern::any_input(),
                                                        pattern::any_input() });
    auto gather7 = pattern::wrap_type<opset7::Gather>({ pattern::wrap_type<opset1::Multiply>(),
                                                        pattern::any_input(),
                                                        pattern::any_input() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(
        std::make_shared<pattern::op::Or>(OutputVector{ gather8, gather7 }), matcher_name);

    this->register_matcher(m, callback);
}

bool GatherTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    auto node = m.get_match_root();
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> gather = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(gather, defaultPrecisions);

    if (dequantization.multiply != nullptr) {
        auto newConstant = gatherDeqConstant(gather, dequantization.multiplyConstant);
        replace_node(dequantization.multiplyConstant, newConstant);
    }
    if (dequantization.subtract != nullptr) {
        auto newConstant = gatherDeqConstant(gather, dequantization.subtractConstant);
        replace_node(dequantization.subtractConstant, newConstant);
    }

    moveDequantizationAfter(context, gather, NetworkHelper::getDequantization(gather, defaultPrecisions), false);
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

    return true;
}

bool GatherTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
