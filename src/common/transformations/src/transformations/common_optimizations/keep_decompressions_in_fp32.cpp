// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/keep_decompressions_in_fp32.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace pass;

ov::pass::KeepDecompressionsInFP32::KeepDecompressionsInFP32() {
    MATCHER_SCOPE(KeepDecompressionsInFP32Matcher);

    auto node_pattern = pattern::wrap_type<opset10::Constant, opset10::Convert>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();

        auto const_node = dynamic_pointer_cast<opset10::Constant>(node);
        if ((const_node && node->get_output_element_type(0) == element::f16) || is_decompression(node)) {
            ov::disable_constant_folding(node);
            return true;
        }

        return false;
    };
    auto m = make_shared<pattern::Matcher>(node_pattern, matcher_name);
    register_matcher(m, callback);
}
