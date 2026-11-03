// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/rt_info/is_shape_subgraph.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

namespace v0 = ov::op::v0;

namespace ov::pass {

EnableDecompressionConvertConstantFolding::EnableDecompressionConvertConstantFolding() {
    MATCHER_SCOPE(EnableDecompressionConvertConstantFolding);
    auto convert = pattern::wrap_type<v0::Convert>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        if (!is_decompression(node))
            return false;
        enable_constant_folding(node);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(convert, matcher_name);
    this->register_matcher(m, callback);
}

DisableDecompressionConvertConstantFolding::DisableDecompressionConvertConstantFolding() {
    MATCHER_SCOPE(DisableDecompressionConvertConstantFolding);
    auto convert = pattern::wrap_type<v0::Convert>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        if (!is_decompression(node))
            return false;
        disable_constant_folding(node);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(convert, matcher_name);
    this->register_matcher(m, callback);
}

KeepConstAndDecompression::KeepConstAndDecompression() {
    MATCHER_SCOPE(KeepDecompressionsInFP32Matcher);

    auto node_pattern = pattern::wrap_type<v0::Convert>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto node = m.get_match_root();
        if (!is_decompression(node) || !is_type<v0::Convert>(node) || ov::is_shape_subgraph(node->shared_from_this()))
            return false;

        if (transformation_callback(node)) {
            return false;
        }

        disable_constant_folding(node);

        if (!is_type<v0::Constant>(node->input_value(0).get_node_shared_ptr()))
            return false;
        enable_keep_const_precision(node->input_value(0).get_node_shared_ptr());

        return false;
    };
    auto m = std::make_shared<pattern::Matcher>(node_pattern, matcher_name);
    register_matcher(m, callback);
}

KeepConstantsPrecisionAndAddConverts::KeepConstantsPrecisionAndAddConverts() {
    MATCHER_SCOPE(KeepConstantsPrecisionAndAddConverts);
    auto const_pattern = pattern::wrap_type<v0::Constant>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto const_node = m.get_match_root();

        if (transformation_callback(const_node)) {
            return false;
        }

        enable_keep_const_precision(const_node);

        const auto& constant_target_inputs = const_node->get_output_target_inputs(0);
        const auto& next_node = constant_target_inputs.begin()->get_node()->shared_from_this();
        if (is_type<v0::Convert>(next_node)) {
            disable_constant_folding(next_node);
            if (is_decompression(next_node)) {
                unmark_as_decompression(next_node);
            }
            return true;
        }

        auto convert = std::make_shared<v0::Convert>(const_node, const_node->get_element_type());
        convert->set_friendly_name(const_node->get_friendly_name());

        std::string postfix = const_node->get_element_type() == ov::element::f32 ? "compression" : "decompression";
        const_node->set_friendly_name(const_node->get_friendly_name() + "_postponed_" + postfix);

        ov::copy_runtime_info(const_node, convert);
        disable_constant_folding(convert);

        for (const auto& target_input : constant_target_inputs) {
            target_input.replace_source_output(convert);
        }

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(const_pattern, matcher_name);
    this->register_matcher(m, callback);
}

MarkCompressedFloatConstants::MarkCompressedFloatConstants() {
    MATCHER_SCOPE(MarkCompressedFloatConstants);

    auto constant = pattern::wrap_type<v0::Constant>();
    auto convert = pattern::wrap_type<v0::Convert>({constant});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& convert_node = as_type_ptr<v0::Convert>(m.get_match_root());
        if (convert_node == nullptr)
            return false;

        const auto& const_node = convert_node->input_value(0).get_node_shared_ptr();
        if (const_node == nullptr)
            return false;

        if (convert_node->get_destination_type() != element::f32)
            return false;
        if (const_node->get_output_element_type(0) != element::f16 &&
            const_node->get_output_element_type(0) != element::bf16 &&
            const_node->get_output_element_type(0) != element::f4e2m1 &&
            const_node->get_output_element_type(0) != element::f8e4m3 &&
            const_node->get_output_element_type(0) != element::f8e5m2 &&
            const_node->get_output_element_type(0) != element::f8e8m0)
            return false;

        mark_as_decompression(convert_node);
        disable_constant_folding(const_node);
        disable_constant_folding(convert_node);
        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(convert, matcher_name);
    this->register_matcher(m, callback);
}

MoveDecompressionAfterSplit::MoveDecompressionAfterSplit() {
    MATCHER_SCOPE(MoveDecompressionAfterSplit);
    auto constant = pattern::wrap_type<v0::Constant>([](const Output<Node>& output) {
        const auto bitwidth = output.get_element_type().bitwidth();
        return bitwidth >= 8 && bitwidth % 8 == 0;
    });
    auto convert = pattern::wrap_type<v0::Convert>({constant}, [](const Output<Node>& output) {
        return is_decompression(output.get_node_shared_ptr());
    });
    auto axis = pattern::any_input();
    auto split = pattern::wrap_type<ov::op::v1::Split>({convert, axis});
    auto variadic_split = pattern::wrap_type<ov::op::v1::VariadicSplit>({convert, axis, pattern::any_input()});
    auto split_pattern = std::make_shared<pattern::op::Or>(OutputVector{split, variadic_split});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto split_node = m.get_match_root();
        if (transformation_callback(split_node) || constant_folding_is_disabled(split_node))
            return false;
        const auto convert_node = m.get_pattern_value_map().at(convert).get_node_shared_ptr();

        OutputVector inputs{convert_node->input_value(0)};
        for (size_t i = 1; i < split_node->get_input_size(); ++i) {
            auto input = ov::util::get_constant_from_source(split_node->input_value(i));
            if (!input)
                return false;
            inputs.push_back(input);
        }
        auto low_precision_split = split_node->clone_with_new_inputs(inputs);
        OutputVector folded(split_node->get_output_size());
        if (!low_precision_split->constant_fold(folded, inputs))
            return false;

        OutputVector replacements;
        for (size_t i = 0; i < folded.size(); ++i) {
            auto replacement = convert_node->clone_with_new_inputs({folded[i]});
            replacement->set_friendly_name(split_node->get_friendly_name() + "." + std::to_string(i));
            copy_runtime_info({inputs[0].get_node_shared_ptr(), split_node}, folded[i].get_node_shared_ptr());
            copy_runtime_info({convert_node, split_node}, replacement);
            mark_as_decompression(replacement);
            disable_constant_folding(replacement);
            disable_constant_folding(folded[i].get_node_shared_ptr());
            replacements.push_back(replacement);
        }
        replace_node(split_node, replacements);
        return true;
    };

    register_matcher(std::make_shared<pattern::Matcher>(split_pattern, matcher_name), callback);
}

}  // namespace ov::pass
