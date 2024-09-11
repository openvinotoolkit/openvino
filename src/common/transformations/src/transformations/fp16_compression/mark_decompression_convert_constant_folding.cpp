// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/is_shape_subgraph.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

pass::EnableDecompressionConvertConstantFolding::EnableDecompressionConvertConstantFolding() {
    MATCHER_SCOPE(EnableDecompressionConvertConstantFolding);
    auto convert = pattern::wrap_type<ov::op::v0::Convert>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        if (!is_decompression(node))
            return false;
        enable_constant_folding(node);
        return true;
    };

    auto m = std::make_shared<pass::pattern::Matcher>(convert, matcher_name);
    this->register_matcher(m, callback);
}

pass::DisableDecompressionConvertConstantFolding::DisableDecompressionConvertConstantFolding() {
    MATCHER_SCOPE(DisableDecompressionConvertConstantFolding);
    auto convert = pattern::wrap_type<ov::op::v0::Convert>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        if (!is_decompression(node))
            return false;
        disable_constant_folding(node);
        return true;
    };

    auto m = std::make_shared<pass::pattern::Matcher>(convert, matcher_name);
    this->register_matcher(m, callback);
}

pass::KeepConstAndDecompression::KeepConstAndDecompression() {
    MATCHER_SCOPE(KeepDecompressionsInFP32Matcher);

    auto node_pattern = pattern::wrap_type<ov::op::v0::Convert>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto node = m.get_match_root();
        if (!is_decompression(node) || !is_type<ov::op::v0::Convert>(node) ||
            ov::is_shape_subgraph(node->shared_from_this()))
            return false;

        if (transformation_callback(node)) {
            return false;
        }

        disable_constant_folding(node);

        if (!is_type<ov::op::v0::Constant>(node->input_value(0).get_node_shared_ptr()))
            return false;
        enable_keep_const_precision(node->input_value(0).get_node_shared_ptr());

        return false;
    };
    auto m = std::make_shared<pattern::Matcher>(node_pattern, matcher_name);
    register_matcher(m, callback);
}

pass::KeepConstantsPrecisionAndAddConverts::KeepConstantsPrecisionAndAddConverts() {
    MATCHER_SCOPE(KeepConstantsPrecisionAndAddConverts);
    auto const_pattern = pattern::wrap_type<ov::op::v0::Constant>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto const_node = m.get_match_root();

        if (transformation_callback(const_node)) {
            return false;
        }

        enable_keep_const_precision(const_node);

        const auto& constant_target_inputs = const_node->get_output_target_inputs(0);
        const auto& next_node = constant_target_inputs.begin()->get_node()->shared_from_this();
        if (is_type<ov::op::v0::Convert>(next_node)) {
            disable_constant_folding(next_node);
            if (is_decompression(next_node)) {
                unmark_as_decompression(next_node);
            }
            return true;
        }

        auto convert = std::make_shared<ov::op::v0::Convert>(const_node, const_node->get_element_type());
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

    auto m = std::make_shared<pass::pattern::Matcher>(const_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::MarkCompressedFloatConstants::MarkCompressedFloatConstants() {
    MATCHER_SCOPE(MarkCompressedFloatConstants);

    auto constant = pattern::wrap_type<ov::op::v0::Constant>();
    auto convert = pattern::wrap_type<ov::op::v0::Convert>({constant});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& convert_node = as_type_ptr<ov::op::v0::Convert>(m.get_match_root());
        const auto& const_node = convert_node->input_value(0).get_node_shared_ptr();
        if (convert_node == nullptr || const_node == nullptr)
            return false;
        if (convert_node->get_destination_type() != element::f32)
            return false;
        if (const_node->get_output_element_type(0) != element::f16 &&
            const_node->get_output_element_type(0) != element::bf16)
            return false;

        mark_as_decompression(convert_node);
        disable_constant_folding(const_node);
        disable_constant_folding(convert_node);
        return false;
    };

    auto m = std::make_shared<pass::pattern::Matcher>(convert, matcher_name);
    this->register_matcher(m, callback);
}
