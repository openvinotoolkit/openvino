// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "listconstruct_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/variadic_split.hpp"
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
    const auto& list = pattern::wrap_type<ov::op::util::FrameworkNode>();

    const auto& broadcast_op = pattern::wrap_type<v3::Broadcast>({pattern::any_input(), list});
    const auto& shape_of_op = pattern::wrap_type<v3::ShapeOf>({list});
    const auto& equal_op = pattern::wrap_type<v1::Equal>({list, pattern::any_input()});
    const auto& select_op = pattern::wrap_type<v1::Select>({pattern::any_input(), pattern::any_input(), list});
    // replace list construct for aten::repeat(tensor,  prim::ListConstruct(shapes)))
    const auto& tile_op = pattern::wrap_type<v0::Tile>({pattern::any_input(), list});
    // aten::split_with_sizes case
    const auto& vsplit_op = pattern::wrap_type<v1::VariadicSplit>({pattern::any_input(), pattern::any_input(), list});
    // aten::upsample... case inside the body when body was removed
    const auto& interpolate_convert_op = pattern::wrap_type<v0::Convert>({list});
    const auto& interpolate_mul_op = pattern::wrap_type<v1::Multiply>({interpolate_convert_op, pattern::any_input()});
    const auto& interpolate_op =
        pattern::wrap_type<v11::Interpolate>({pattern::any_input(), interpolate_mul_op, pattern::any_input()});
    // aten::randint case
    const auto& rand_op = pattern::wrap_type<v8::RandomUniform>({list, pattern::any_input(), pattern::any_input()});
    const auto& lc_pattern = std::make_shared<pattern::op::Or>(
        OutputVector{broadcast_op, shape_of_op, equal_op, select_op, tile_op, vsplit_op, interpolate_op, rand_op});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();

        auto list_out = pattern_map.at(list);
        // Concatenation is possible because all elements in list should be scalar or 1D tensors,
        // result should be 1D tensor.
        OutputVector inputs;
        ov::pass::NodeRegistry rg;
        auto neg_1 = v0::Constant::create(element::i32, Shape{1}, {-1});
        const auto& start_output = list_out;
        for (const auto& input : get_list_as_outputs(start_output)) {
            if (input == start_output) {
                // Start output exist in list elements, it might mean we have only 1 element in list inputs and it is
                // already a list, we do not need to concat it
                return false;
            }
            auto rank = input.get_partial_shape().rank();
            if (rank.is_static() && rank.get_length() > 1) {
                // if list elements of rank higher then 1D we cannot resolve it
                add_exception_to_fw_node(list, "unsupported list: all inputs must be 1D.");
                return false;
            }
            // reshape all elements to 1D
            auto reshape = rg.make<v1::Reshape>(input, neg_1, false);
            if (const auto list_const = ov::util::get_constant_from_source(reshape)) {
                inputs.push_back(list_const);
            } else {
                inputs.push_back(reshape);
            }
        }
        auto concat = rg.make<v0::Concat>(inputs, 0);
        copy_runtime_info_and_name(list_out.get_node_shared_ptr(), rg.get());
        replace_node(list_out.get_node_shared_ptr(), concat);
        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(lc_pattern, "ov::frontend::pytorch::pass::ListConstructReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
