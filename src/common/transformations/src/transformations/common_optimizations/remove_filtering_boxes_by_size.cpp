// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/remove_filtering_boxes_by_size.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/subtract_fusion.hpp"

ov::pass::FuseFilteringBoxesBySize::FuseFilteringBoxesBySize() {
    ADD_MATCHER_FOR_THIS(SubtractFusion);
    ADD_MATCHER_FOR_THIS(RemoveFilteringBoxesBySize);
}

ov::pass::RemoveFilteringBoxesBySize::RemoveFilteringBoxesBySize() {
    MATCHER_SCOPE(RemoveFilteringBoxesBySize);
    // variadic split
    auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1000, 4});
    auto sizes = ov::op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>({1, 1, 1, 1}));
    auto axis = ov::op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>({1}));
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(data, axis, sizes);

    // sub -> add
    auto sub_2_0 = std::make_shared<ov::op::v1::Subtract>(split->output(2), split->output(0));
    auto term_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto add_1 = std::make_shared<ov::op::v1::Add>(sub_2_0, term_1);

    auto sub_3_1 = std::make_shared<ov::op::v1::Subtract>(split->output(3), split->output(1));
    auto term_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto add_2 = std::make_shared<ov::op::v1::Add>(sub_3_1, term_2);

    // concat
    auto concat = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector({split->output(0), split->output(1), add_1->output(0), add_2->output(0)}),
        1);

    // second variadic split
    auto sizes_1 = ov::op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>({1, 1, 1, 1}));
    auto axis_1 = ov::op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>({1}));
    auto split_1 = std::make_shared<ov::op::v1::VariadicSplit>(concat, axis_1, sizes_1);

    // squeeze
    auto squeeze_1_axis = ov::op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>({1}));
    auto squeeze_1 = std::make_shared<ov::op::v0::Squeeze>(split_1->output(2), squeeze_1_axis);

    auto squeeze_2_axis = ov::op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>({1}));
    auto squeeze_2 = std::make_shared<ov::op::v0::Squeeze>(split_1->output(3), squeeze_2_axis);

    // less
    auto less_1_constant = ov::op::v0::Constant::create(element::f32, Shape{1}, std::vector<float>({0}));
    auto less_1 = std::make_shared<ov::op::v1::Less>(squeeze_1, less_1_constant);

    auto less_2_constant = ov::op::v0::Constant::create(element::f32, Shape{1}, std::vector<float>({0}));
    auto less_2 = std::make_shared<ov::op::v1::Less>(squeeze_2, less_2_constant);

    // Logical Not
    auto not_1 = std::make_shared<ov::op::v1::LogicalNot>(less_1);
    auto not_2 = std::make_shared<ov::op::v1::LogicalNot>(less_2);

    // cast
    auto cast_11 = std::make_shared<ov::op::v0::Convert>(not_1, ov::element::u8);
    auto cast_12 = std::make_shared<ov::op::v0::Convert>(cast_11, ov::element::boolean);

    auto cast_21 = std::make_shared<ov::op::v0::Convert>(not_2, ov::element::u8);
    auto cast_22 = std::make_shared<ov::op::v0::Convert>(cast_21, ov::element::boolean);

    // logical and
    auto and_1 = std::make_shared<ov::op::v1::LogicalAnd>(cast_12, cast_22);

    // cast
    auto cast_31 = std::make_shared<ov::op::v0::Convert>(and_1, ov::element::u8);
    auto cast_32 = std::make_shared<ov::op::v0::Convert>(cast_31, ov::element::f32);

    // nonzero
    auto non_zero = std::make_shared<ov::op::v3::NonZero>(cast_32);

    auto order = ov::op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>({1, 0}));
    auto transpose = std::make_shared<ov::op::v1::Transpose>(non_zero, order);

    auto squeeze_3_axis = ov::op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>({1}));
    auto squeeze_3 = std::make_shared<ov::op::v0::Squeeze>(transpose, squeeze_3_axis);

    auto cast = std::make_shared<ov::op::v0::Convert>(squeeze_3, ov::element::i64);

    ov::matcher_pass_callback callback = [data](pattern::Matcher& m) {
        auto start = ov::op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>({0}));
        auto step = ov::op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>({1}));

        const auto& pattern_map = m.get_pattern_map();

        auto input = pattern_map.at(data);
        auto output = m.get_match_root();

        auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input);

        auto axis = ov::op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>({0}));
        auto index = ov::op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>({0}));
        auto stop = std::make_shared<ov::op::v1::Gather>(input_shape, index, axis);

        auto range = std::make_shared<ov::op::v0::Range>(start, stop, step);

        range->set_friendly_name(output->get_friendly_name());
        // TODO: add copy_runtime_info
        ov::replace_node(output, range);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(cast, matcher_name);
    register_matcher(m, callback);
}
