// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"

#include "itt.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"

using namespace ov;

pass::EnableDecompressionConvertConstantFolding::EnableDecompressionConvertConstantFolding() {
    MATCHER_SCOPE(EnableDecompressionConvertConstantFolding);
    auto convert = pattern::wrap_type<opset8::Convert>();

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
    auto convert = pattern::wrap_type<opset8::Convert>();

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

    auto node_pattern = pattern::wrap_type<opset8::Convert>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        if (!is_decompression(node))
            return false;

        if (!is_type<opset8::Convert>(node))
            return false;
        disable_constant_folding(node);

        if (!is_type<opset8::Constant>(node->input_value(0).get_node_shared_ptr()))
            return true;
        disable_constant_folding(node->input_value(0).get_node_shared_ptr());

        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(node_pattern, matcher_name);
    register_matcher(m, callback);
}
