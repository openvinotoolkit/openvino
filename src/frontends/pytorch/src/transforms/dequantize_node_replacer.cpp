// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dequantize_node_replacer.hpp"

#include <list>
#include <memory>
#include <utility>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

#define MAX_SEARCH_NODES 100000

using namespace ov::op;

/**
 * Dequantize Node Replacer
 * Replacer finds the unconverted dequantize ops and removes them.
 * This matches the behavior of OV's LPT.
 */
DequantizeNodeReplacer::DequantizeNodeReplacer() {
    auto dequantize_node = ov::pass::pattern::wrap_type<ov::frontend::pytorch::PtFrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto dequantize_node = cast_fw_node(m.get_match_root(), "aten::dequantize");
        if (!dequantize_node)
            return false;

        auto dequantized_input = dequantize_node->input_value(0);
        dequantize_node->output(0).replace(dequantized_input);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(dequantize_node,
                                                          "ov::frontend::pytorch::pass::DequantizeNodeReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
