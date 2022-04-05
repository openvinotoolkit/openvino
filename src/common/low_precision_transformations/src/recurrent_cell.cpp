// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/recurrent_cell.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <memory>
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/or.hpp>

#include "low_precision/network_helper.hpp"
#include "../include/low_precision/rt_info/skip_cleanup_attribute.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

RecurrentCellTransformation::RecurrentCellTransformation(const Params& params) : LayerTransformation(params) {
    const auto X = ngraph::pattern::wrap_type<ngraph::opset1::Parameter>();
    const auto H = ngraph::pattern::wrap_type<ngraph::opset1::Parameter>();
    const auto C = ngraph::pattern::wrap_type<ngraph::opset1::Parameter>();
    const auto W = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto R = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto B = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();

    const auto fq_X = wrap_fake_quantize(X);
    const auto fq_H = wrap_fake_quantize(H);
    const auto fq_W = wrap_fake_quantize(W);
    const auto fq_R = wrap_fake_quantize(R);

    const auto dequantization_X = wrap_dequantization(ngraph::pattern::any_input(), true);
    const auto dequantization_H = wrap_dequantization(ngraph::pattern::any_input(), true);

    const auto dequantization_without_subtract_X = wrap_dequantization(ngraph::pattern::any_input(), false);
    const auto dequantization_without_subtract_H = wrap_dequantization(ngraph::pattern::any_input(), false);

    const auto lstm_cell = ngraph::pattern::wrap_type<ngraph::opset5::LSTMCell>(
        {fq_X, fq_H, C, fq_W, fq_R, B});
    const auto lstm_cell_with_dequantizations = ngraph::pattern::wrap_type<ngraph::opset5::LSTMCell>(
        {dequantization_X, dequantization_H, C, fq_W, fq_R, B});
    const auto lstm_cell_with_dequantizations_without_subtract = ngraph::pattern::wrap_type<ngraph::opset5::LSTMCell>(
        {dequantization_without_subtract_X, dequantization_without_subtract_H, C, fq_W, fq_R, B});

    const auto gru_cell = ngraph::pattern::wrap_type<ngraph::opset4::GRUCell>({fq_X, fq_H, fq_W, fq_R, B});
    const auto gru_cell_with_dequantizations = ngraph::pattern::wrap_type<ngraph::opset4::GRUCell>(
        {dequantization_X, dequantization_H, fq_W, fq_R, B});
    const auto gru_cell_with_dequantizations_without_subtract = ngraph::pattern::wrap_type<ngraph::opset4::GRUCell>(
        {dequantization_without_subtract_X, dequantization_without_subtract_H, fq_W, fq_R, B});

    const auto rnn_cell = ngraph::pattern::wrap_type<ngraph::opset4::RNNCell>({fq_X, fq_H, fq_W, fq_R, B});
    const auto rnn_cell_with_dequantizations = ngraph::pattern::wrap_type<ngraph::opset4::RNNCell>(
        {dequantization_X, dequantization_H, fq_W, fq_R, B});
    const auto rnn_cell_with_dequantizations_without_subtract = ngraph::pattern::wrap_type<ngraph::opset4::RNNCell>(
        {dequantization_without_subtract_X, dequantization_without_subtract_H, fq_W, fq_R, B});

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }

        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(
        std::make_shared<pattern::op::Or>(OutputVector{lstm_cell,
                                                       lstm_cell_with_dequantizations,
                                                       lstm_cell_with_dequantizations_without_subtract,
                                                       gru_cell,
                                                       gru_cell_with_dequantizations,
                                                       gru_cell_with_dequantizations_without_subtract,
                                                       rnn_cell,
                                                       rnn_cell_with_dequantizations,
                                                       rnn_cell_with_dequantizations_without_subtract}),
        "LSTM");
    this->register_matcher(m, callback);
}

bool RecurrentCellTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    const auto lstm = m.get_match_root();
    if (!canBeTransformed(context, lstm)) {
        return false;
    }
    for (size_t parentIndex = 0ul; parentIndex < lstm->get_input_size(); parentIndex++) {
        auto lstm_parent = lstm->get_input_node_shared_ptr(parentIndex);
        if (is_type<ngraph::opset1::FakeQuantize>(lstm_parent)) {
            auto fq_parent = lstm_parent->get_input_node_shared_ptr(0);
            if (is_type<ngraph::opset5::Constant>(fq_parent)) {
                auto fq_node = as_type_ptr<ngraph::opset1::FakeQuantize>(lstm_parent);
                const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fq_node);
                const auto precisionsAttribute = getAttributeFromOutput<PrecisionsAttribute>(lstm_parent);
                const auto precisions = precisionsAttribute.empty()
                                            ? defaultPrecisions
                                            : precisionsAttribute.as<PrecisionsAttribute>().value();
                const DataPrecision dataPrecision = getDataPrecision(lstm_parent, quantizationDetails, precisions);
                auto QDQ = NetworkHelper::decomposeFakeQuantize(fq_node,
                                                                  dataPrecision.precision,
                                                                  dataPrecision.min,
                                                                  dataPrecision.max,
                                                                  dataPrecision.hasZeroPoint,
                                                                  updatePrecisions);
                std::shared_ptr<ngraph::Node> new_fq = std::get<0>(QDQ);
                std::shared_ptr<ngraph::Node> deq_multiply = std::get<1>(QDQ);
                auto multiply_parent = deq_multiply->get_input_node_shared_ptr(0);
                ov::disable_constant_folding(multiply_parent);
                propagateSkipCleanupAttribute(deq_multiply);
                this->register_new_node(new_fq);
                updateOutput(context, deq_multiply, new_fq);
            } else {
                continue;
            }
        } else {
            if (is_type<ngraph::opset1::Multiply>(lstm_parent)) {
                propagateSkipCleanupAttribute(lstm_parent);
            }
            continue;
        }
    }
    return true;
}

bool RecurrentCellTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> lstm) const {
    std::shared_ptr<ov::Node> W, R;
    if (is_type<opset5::LSTMCell>(lstm)) {
        W = lstm->get_input_node_shared_ptr(3);
        R = lstm->get_input_node_shared_ptr(4);
    } else {
        W = lstm->get_input_node_shared_ptr(2);
        R = lstm->get_input_node_shared_ptr(3);
    }
    for (auto fq_on_weight : {W, R}) {
        auto fq_node = as_type_ptr<ngraph::opset1::FakeQuantize>(fq_on_weight);
        const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fq_node);
        const auto precisionsAttribute = getAttributeFromOutput<PrecisionsAttribute>(fq_on_weight);
        const auto precisions = precisionsAttribute.empty()
                                    ? defaultPrecisions
                                    : precisionsAttribute.as<PrecisionsAttribute>().value();
        const DataPrecision dataPrecision = getDataPrecision(fq_on_weight, quantizationDetails, precisions);
        auto QDQ = NetworkHelper::decomposeFakeQuantize(fq_node,
                                                        dataPrecision.precision,
                                                        dataPrecision.min,
                                                        dataPrecision.max,
                                                        dataPrecision.hasZeroPoint,
                                                        updatePrecisions);
        std::shared_ptr<ngraph::Node> new_fq = std::get<0>(QDQ);
        std::shared_ptr<ngraph::Node> deq_multiply = std::get<1>(QDQ);
        if (deq_multiply == nullptr || new_fq == nullptr) {
            return false;
        }
        auto multiply_parent = deq_multiply->get_input_node_shared_ptr(0);
        if (is_type<ngraph::opset1::Subtract>(multiply_parent)) {
            return false;
        }
    }
    return true;
}

bool RecurrentCellTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return true;
}

void RecurrentCellTransformation::propagateSkipCleanupAttribute(std::shared_ptr<Node> multiply) {
    SkipCleanupAttribute::create(multiply);
    auto multiply_parent = multiply->get_input_node_shared_ptr(0);
    SkipCleanupAttribute::create(multiply_parent);
    if (is_type<ngraph::opset1::Subtract>(multiply_parent)) {
        auto subtract_parent = multiply_parent->get_input_node_shared_ptr(0);
        SkipCleanupAttribute::create(subtract_parent);
    }
}

std::shared_ptr<ov::Node> RecurrentCellTransformation::wrap_fake_quantize(
    const std::shared_ptr<ov::Node> parameter) {
    const auto input_low = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto input_high = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto output_low = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto output_high = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    return ngraph::pattern::wrap_type<opset1::FakeQuantize>({
        parameter,
        input_low,
        input_high,
        output_low,
        output_high});
}

std::shared_ptr<ov::Node> RecurrentCellTransformation::wrap_quantization(
    const std::shared_ptr<ov::Node> parameter) {
    const auto quantization_fake_quantize = wrap_fake_quantize(parameter);
    const auto quantization_convert = ngraph::pattern::wrap_type<ngraph::opset1::Convert>(
        {quantization_fake_quantize});
    return quantization_convert;
}

std::shared_ptr<ov::Node> RecurrentCellTransformation::wrap_dequantization(
    const std::shared_ptr<ov::Node> parameter,
    const bool with_subtract) {
    const auto dequantization_convert = ngraph::pattern::wrap_type<ngraph::opset1::Convert>({parameter});
    const auto subtract_constant = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto dequantization_subtract = ngraph::pattern::wrap_type<ngraph::opset1::Subtract>(
        {dequantization_convert, subtract_constant});
    const auto multiply_constant = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto multiply_parent = with_subtract ? dequantization_subtract : dequantization_convert;
    const auto dequantization_multiply = ngraph::pattern::wrap_type<ngraph::opset1::Multiply>(
        {multiply_parent, multiply_constant});
    return dequantization_multiply;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
