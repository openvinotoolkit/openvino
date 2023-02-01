// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "listconstruct_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/roll.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::pass;
using namespace ov::op;

ListConstructReplacer::ListConstructReplacer() {
    // Transformation for torch operators for cases where prim::ListConstruct can be replaced with Concat.
    auto list_construct = pattern::wrap_type<ov::op::util::FrameworkNode>();

    // Both aten::view and aten::reshape are using same translation returning Reshape operator.
    auto reshape_op = pattern::wrap_type<v1::Reshape>({pattern::any_input(), list_construct});
    auto roll_op = pattern::wrap_type<v7::Roll>({pattern::any_input(), list_construct, pattern::any_input()});
    auto broadcast_op = pattern::wrap_type<v1::Broadcast>({pattern::any_input(), list_construct});

    auto lc_pattern = std::make_shared<pattern::op::Or>(OutputVector{reshape_op, roll_op, broadcast_op});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();

        auto list_construct_node = pattern_map.at(list_construct).get_node_shared_ptr();
        if (auto list_unpack_node = cast_fw_node(list_construct_node, "prim::ListConstruct")) {
            // Concatenation is possible because all elements in list should be scalar intigers.
            OutputVector inputs;
            auto axis_0 = v0::Constant::create(element::i64, Shape{}, {0});
            for (auto& input : list_construct_node->inputs()) {
                auto rank = input.get_partial_shape().rank();
                FRONT_END_OP_CONVERSION_CHECK(rank.is_dynamic() || rank.get_length() == 0, "Rank must be 0");
                auto unsqueeze = std::make_shared<v0::Unsqueeze>(input.get_source_output(), axis_0);
                inputs.push_back(unsqueeze);
            }
            auto concat = std::make_shared<v0::Concat>(inputs, 0);
            copy_runtime_info({list_construct_node}, concat);
            replace_node(list_construct_node, concat);
            return true;
        };
        return false;
    };
    auto m = std::make_shared<pattern::Matcher>(lc_pattern, "ov::frontend::pytorch::pass::ListConstructReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
