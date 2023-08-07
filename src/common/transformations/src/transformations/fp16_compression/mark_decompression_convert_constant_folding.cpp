// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/is_shape_subgraph.hpp"
#include "transformations/rt_info/keep_fp16_const.hpp"

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

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        if (!is_decompression(node) || !is_type<ov::op::v0::Convert>(node) ||
            ov::is_shape_subgraph(node->shared_from_this()))
            return false;

        disable_constant_folding(node);

        if (!is_type<ov::op::v0::Constant>(node->input_value(0).get_node_shared_ptr()))
            return false;
        enable_keep_fp16_const(node->input_value(0).get_node_shared_ptr());

        return false;
    };
    auto m = std::make_shared<pattern::Matcher>(node_pattern, matcher_name);
    register_matcher(m, callback);
}

pass::KeepConstAndDecompressionForMatMul::KeepConstAndDecompressionForMatMul() {
    MATCHER_SCOPE(KeepConstAndDecompressionForMatMul);
    auto matmul = pass::pattern::wrap_type<ov::op::v0::MatMul>();

    matcher_pass_callback callback = [=](pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        // input to matmul is decompression Convert
        const auto& inp_convert = node->input_value(1).get_node_shared_ptr();
        if (!is_type<ov::op::v0::Convert>(inp_convert) || inp_convert->get_output_target_inputs(0).size() != 1 ||
            !is_decompression(inp_convert))
            return false;

        disable_constant_folding(inp_convert);

        if (!is_type<ov::op::v0::Constant>(inp_convert->input_value(0).get_node_shared_ptr()))
            return false;
        enable_keep_fp16_const(inp_convert->input_value(0).get_node_shared_ptr());

        return false;
    };

    auto m = std::make_shared<pass::pattern::Matcher>(matmul, matcher_name);
    this->register_matcher(m, callback);
}
