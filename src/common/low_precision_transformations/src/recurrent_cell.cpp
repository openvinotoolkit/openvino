// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/recurrent_cell.hpp"

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/disable_cleanup_attribute.hpp"

namespace ov {
namespace pass {
namespace low_precision {

RecurrentCellTransformation::RecurrentCellTransformation(const Params& params) : LayerTransformation(params) {
    const auto C = ov::pass::pattern::any_input();
    const auto S = ov::pass::pattern::any_input();
    const auto B = ov::pass::pattern::wrap_type<ov::opset1::Constant>();

    auto X_in = ov::pass::pattern::any_input();
    auto H_in = ov::pass::pattern::any_input();
    auto W_in = ov::pass::pattern::any_input();
    auto R_in = ov::pass::pattern::any_input();

    const auto lstm_seq = ov::pass::pattern::wrap_type<ov::opset5::LSTMSequence>(
        {X_in, H_in, C, S, W_in, R_in, B});
    const auto gru_seq  = ov::pass::pattern::wrap_type<ov::opset5::GRUSequence>(
        {X_in, H_in,    S, W_in, R_in, B});

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }

        return transform(*context, m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        std::make_shared<pass::pattern::op::Or>(
            OutputVector {
                lstm_seq,
                gru_seq
            }),
        "RecurrentCellTransformation");

    this->register_matcher(m, callback);
}

namespace {

std::shared_ptr<ov::opset1::FakeQuantize> find_fake_quantize_upper(const std::shared_ptr<Node>& parent) {
    if (auto fq = as_type_ptr<ov::opset1::FakeQuantize>(parent)) {
        return fq;
    }

    if (!NetworkHelper::isPrecisionPreserved(parent)) {
        return nullptr;
    }

    return find_fake_quantize_upper(parent->get_input_node_shared_ptr(0));
}

template <class Operation>
std::string name() {
    return Operation::get_type_info_static().name;
}

bool isSupportedForPerChannelQuantization(const std::shared_ptr<Node>& node) {
    static const std::unordered_set<std::string> supportedForPerChannelQuantization = {
        { name<opset1::DepthToSpace>() },
        { name<opset1::Interpolate>() },
        { name<opset1::MaxPool>() },
        { name<opset1::ReduceMax>() },
        { name<opset1::ReduceMin>() },
        { name<opset1::Relu>() },
        { name<opset2::BatchToSpace>() },
        { name<opset1::Broadcast>() },
        { name<opset3::Broadcast>() },
        { name<opset1::Pad>() },
        { name<opset12::Pad>() },
        { name<opset1::Reshape>() },
        { name<opset1::Squeeze>() },
        { name<opset2::SpaceToBatch>() },
        { name<opset1::StridedSlice>() },
        { name<opset1::ShuffleChannels>() },
        { name<opset1::Transpose>() },
        { name<opset1::Unsqueeze>() }
    };

    return supportedForPerChannelQuantization.find(node->get_type_name()) != supportedForPerChannelQuantization.end();
}

std::vector<std::pair<size_t, element::Type>> get_supported_precisions(std::shared_ptr<ov::Node> lstm) {
    // pair fields:
    // 0 - input number,
    // 1 - input type, `element::undefined` - any precision
    if (is_type<ov::opset5::LSTMSequence>(lstm)) {
        return std::vector<std::pair<size_t, element::Type>>{ {0, element::u8}, { 1, element::u8 }, { 4, element::undefined }, { 5, element::undefined } };
    } else if (is_type<ov::opset5::GRUSequence>(lstm)) {
        return std::vector<std::pair<size_t, element::Type>>{ {0, element::u8}, { 1, element::u8 }, { 3, element::undefined }, { 4, element::undefined } };
    }

    OPENVINO_THROW("unsupported operation type: ", lstm->get_type_name());
}

} // namespace

void RecurrentCellTransformation::propagate(TransformationContext& context, const std::shared_ptr<ov::Node> node) {
    if (!isSupportedForPerChannelQuantization(node)) {
        return;
    }

    const auto& normalized_node = NetworkHelper::separateInStandaloneBranch(node, defaultPrecisions);
    auto dequantization = NetworkHelper::getDequantization(node, defaultPrecisions);
    if (dequantization.empty()) {
        return;
    }
    const auto& new_node = moveDequantizationAfter(context, normalized_node, dequantization);

    const auto& new_dequantization = NetworkHelper::getDequantizationBelow(new_node);
    if (new_dequantization.empty()) {
        return;
    }

    for (auto output : new_dequantization.multiply->outputs()) {
        for (auto input : output.get_target_inputs()) {
            auto child = input.get_node()->shared_from_this();
            propagate(context, child);
        }
    }
}

bool RecurrentCellTransformation::transform(TransformationContext& context, ov::pass::pattern::Matcher& m) {
    const auto lstm = m.get_match_root();
    const auto inputs = get_supported_precisions(lstm);
    for (const auto& input : inputs) {
        const auto& parent = lstm->get_input_node_shared_ptr(input.first);
        if (!isSupportedForPerChannelQuantization(parent)) {
            continue;
        }

        const auto& fq = find_fake_quantize_upper(parent);
        if (fq != nullptr) {
            const auto& quantizationDetails = QuantizationDetails::getDetails(fq);
            if ((quantizationDetails.inputLowValues.size() != 1) || (quantizationDetails.inputHighValues.size() != 1) ||
                (quantizationDetails.outputLowValues.size() != 1) || (quantizationDetails.outputHighValues.size() != 1)) {
                continue;
            }

            const auto& precisionsAttribute = getAttributeFromOutput<PrecisionsAttribute>(fq);
            const auto& precisions = precisionsAttribute.empty() ?
                defaultPrecisions :
                precisionsAttribute.as<PrecisionsAttribute>().value();
            const auto& dataPrecision = getDataPrecision(fq, quantizationDetails, precisions);
            if (dataPrecision.empty() || ((input.second != element::undefined) && (dataPrecision.precision != input.second))) {
                return false;
            }

            auto result = NetworkHelper::decomposeFakeQuantize(
                fq,
                dataPrecision.precision,
                dataPrecision.min,
                dataPrecision.max,
                dataPrecision.hasZeroPoint,
                updatePrecisions);
            auto multiply = std::get<1>(result);

            for (const auto& output : multiply->outputs()) {
                for (const auto& input : output.get_target_inputs()) {
                    const auto input_node = input.get_node();
                    propagate(context, input_node->shared_from_this());
                }
            }
        }
    }

    if (!canBeTransformed(context, lstm)) {
        return false;
    }

    for (size_t parentIndex = 0ul; parentIndex < lstm->get_input_size(); parentIndex++) {
        auto lstm_parent = lstm->get_input_node_shared_ptr(parentIndex);
        if (is_type<ov::opset1::FakeQuantize>(lstm_parent)) {
            auto fq_parent = lstm_parent->get_input_node_shared_ptr(0);
            if (is_type<ov::opset5::Constant>(fq_parent)) {
                auto fq_node = as_type_ptr<ov::opset1::FakeQuantize>(lstm_parent);
                const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fq_node);
                const auto precisionsAttribute = getAttributeFromOutput<PrecisionsAttribute>(lstm_parent);
                const auto precisions = precisionsAttribute.empty()
                                            ? defaultPrecisions
                                            : precisionsAttribute.as<PrecisionsAttribute>().value();
                const DataPrecision dataPrecision = getDataPrecision(lstm_parent, quantizationDetails, precisions);
                if (dataPrecision.empty() || dataPrecision.hasZeroPoint) {
                    return false;
                }

                auto QDQ = NetworkHelper::decomposeFakeQuantize(fq_node,
                                                                  dataPrecision.precision,
                                                                  dataPrecision.min,
                                                                  dataPrecision.max,
                                                                  dataPrecision.hasZeroPoint,
                                                                  updatePrecisions);
                std::shared_ptr<ov::Node> new_fq = std::get<0>(QDQ);
                std::shared_ptr<ov::Node> deq_multiply = std::get<1>(QDQ);
                if (deq_multiply == nullptr || new_fq == nullptr) {
                    return false;
                }

                std::shared_ptr<ov::Node> convert;
                auto multiply_parent = deq_multiply->get_input_node_shared_ptr(0);
                if (is_type<ov::opset1::Subtract>(multiply_parent)) {
                    convert = multiply_parent->get_input_node_shared_ptr(0);
                } else {
                    convert = multiply_parent;
                }
                ov::disable_constant_folding(convert);
                propagateSkipCleanupAttribute(deq_multiply);

                this->register_new_node(new_fq);
                updateOutput(context, deq_multiply, new_fq);
            } else {
                continue;
            }
        } else {
            if (is_type<ov::opset1::Multiply>(lstm_parent)) {
                auto multiply = lstm_parent->get_input_node_shared_ptr(0);
                ov::disable_constant_folding(multiply);
                propagateSkipCleanupAttribute(lstm_parent);
            }
            continue;
        }
    }

    return true;
}

bool RecurrentCellTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> lstm) const {
    const auto inputs = get_supported_precisions(lstm);
    for (const auto& index : inputs) {
        const auto& input = lstm->get_input_node_ptr(index.first);
        if (as_type<ov::opset1::Constant>(input) || as_type<ov::opset1::FakeQuantize>(input)) {
            continue;
        }

        const auto dequantization = NetworkHelper::getDequantization(lstm, defaultPrecisions, index.first);
        if (dequantization.empty()) {
            continue;
        }
        if ((index.second != element::undefined) && (dequantization.data.get_element_type() != index.second)) {
            return false;
        }
    }
    return true;
}

bool RecurrentCellTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return true;
}

void RecurrentCellTransformation::propagateSkipCleanupAttribute(std::shared_ptr<Node> multiply) {
    DisableCleanupAttribute::create(multiply);
    auto multiply_parent = multiply->get_input_node_shared_ptr(0);
    DisableCleanupAttribute::create(multiply_parent);
    if (is_type<ov::opset1::Subtract>(multiply_parent)) {
        auto subtract_parent = multiply_parent->get_input_node_shared_ptr(0);
        DisableCleanupAttribute::create(subtract_parent);
    }
}

std::shared_ptr<ov::Node> RecurrentCellTransformation::wrap_fake_quantize(
    const std::shared_ptr<ov::Node> parameter) {
    const auto input_low = ov::pass::pattern::wrap_type<ov::opset1::Constant>();
    const auto input_high = ov::pass::pattern::wrap_type<ov::opset1::Constant>();
    const auto output_low = ov::pass::pattern::wrap_type<ov::opset1::Constant>();
    const auto output_high = ov::pass::pattern::wrap_type<ov::opset1::Constant>();
    return ov::pass::pattern::wrap_type<opset1::FakeQuantize>({
        parameter,
        input_low,
        input_high,
        output_low,
        output_high});
}

std::shared_ptr<ov::Node> RecurrentCellTransformation::wrap_quantization(
    const std::shared_ptr<ov::Node> parameter) {
    const auto quantization_fake_quantize = wrap_fake_quantize(parameter);
    const auto quantization_convert = ov::pass::pattern::wrap_type<ov::opset1::Convert>(
        {quantization_fake_quantize});
    return quantization_convert;
}

std::shared_ptr<ov::Node> RecurrentCellTransformation::wrap_dequantization(
    const std::shared_ptr<ov::Node> parameter,
    const bool with_subtract) {
    const auto dequantization_convert = ov::pass::pattern::wrap_type<ov::opset1::Convert>({parameter});
    const auto subtract_constant = ov::pass::pattern::wrap_type<ov::opset1::Constant>();
    const auto dequantization_subtract = ov::pass::pattern::wrap_type<ov::opset1::Subtract>(
        {dequantization_convert, subtract_constant});
    const auto multiply_constant = ov::pass::pattern::wrap_type<ov::opset1::Constant>();
    const auto multiply_parent = with_subtract ? dequantization_subtract : dequantization_convert;
    const auto dequantization_multiply = ov::pass::pattern::wrap_type<ov::opset1::Multiply>(
        {multiply_parent, multiply_constant});
    return dequantization_multiply;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
