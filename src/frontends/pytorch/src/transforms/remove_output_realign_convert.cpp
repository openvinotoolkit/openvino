// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remove_output_realign_convert.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "rt_info/type_realign_convert.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

RemoveOutputRealignConvert::RemoveOutputRealignConvert() {
    const auto convert_pattern = ov::pass::pattern::wrap_type<v0::Convert, v1::ConvertLike>();

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(convert_pattern,
                                                     "ov::frontend::pytorch::pass::RemoveOutputRealignConvert"),
        [=](ov::pass::pattern::Matcher& m) {
            const auto& convert_node = m.get_match_root();
            if (!ov::frontend::pytorch::is_type_realign_convert(convert_node))
                return false;

            // Always erase the marker: it must never survive normalization.
            ov::frontend::pytorch::unmark_type_realign_convert(convert_node);

            const auto& target_inputs = convert_node->output(0).get_target_inputs();
            if (target_inputs.empty())
                return false;
            bool all_results = true;
            for (const auto& input : target_inputs) {
                if (!ov::as_type<v0::Result>(input.get_node())) {
                    all_results = false;
                    break;
                }
            }
            if (!all_results)
                return false;

            // All consumers are Result(s): bypass the convert so the higher precision value
            // reaches the output directly.
            replace_output_update_name(convert_node->output(0), convert_node->input_value(0));
            return true;
        });
}

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
