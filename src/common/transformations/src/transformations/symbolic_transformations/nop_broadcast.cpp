// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/nop_broadcast.hpp"

#include "compare.hpp"
#include "itt.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/symbolic_transformations/utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::symbol::util;

namespace {
shared_ptr<Node> broadcast_label(const OutputVector& inputs) {
    return ov::pass::pattern::wrap_type<op::v1::Broadcast, op::v3::Broadcast>(inputs, [](Output<Node> output) {
        const auto& op = output.get_node_shared_ptr();
        auto data_rank = op->get_input_partial_shape(0).rank();
        auto new_shape_shape = op->get_input_partial_shape(1);
        return data_rank.is_static() && new_shape_shape.is_static() && data_rank == new_shape_shape[0];
    });
}
}  // namespace

ov::pass::NopBroadcast::NopBroadcast() {
    MATCHER_SCOPE(NopBroadcast);
    auto data_label = pattern::any_input(pattern::has_static_rank());

    auto shape_label = pattern::wrap_type<op::v0::ShapeOf, op::v3::ShapeOf>();
    auto ones = INT_CONSTANT_WITH_PREDICATE(std::all_of(value.begin(), value.end(), cmp::Equal<int64_t>(1)));
    auto maximum = pattern::wrap_type<op::v1::Maximum>({shape_label, ones});

    auto broadcast_3_ins = broadcast_label({data_label, maximum, pattern::any_input()});
    auto broadcast_2_ins = broadcast_label({data_label, maximum});
    auto broadcast = make_shared<pattern::op::Or>(OutputVector{broadcast_2_ins, broadcast_3_ins});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& vm = m.get_pattern_value_map();
        auto data = vm.at(data_label);
        auto shape = vm.at(shape_label);

        ov::TensorSymbol data_symbols, shape_symbols;
        if (!get_symbols(data.get_partial_shape(), data_symbols) || !get_symbols(shape, shape_symbols) ||
            !are_unique_and_equal_symbols(data_symbols, shape_symbols))
            return false;
        return ov::replace_output_update_name(m.get_match_root(), data);
    };

    auto m = std::make_shared<pattern::Matcher>(broadcast, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
