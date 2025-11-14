// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/convert_one_hot_v16_to_v1.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

namespace {
template <typename TOpFunc>
std::shared_ptr<ov::Model> create_model(const TOpFunc& op_func) {
    const std::vector<int64_t> indices{0, 1, 2};
    const float on_value = 1.123f;
    const float off_value = 0.321f;

    const auto indices_const = ov::op::v0::Constant::create(element::i64, Shape{3}, indices);
    const auto depth_const = ov::op::v0::Constant::create(element::i64, Shape{}, {3});
    const auto on_const = ov::op::v0::Constant::create(element::f32, Shape{}, {on_value});
    const auto off_const = ov::op::v0::Constant::create(element::f32, Shape{}, {off_value});

    auto one_hot = op_func(indices_const, depth_const, on_const, off_const, -1);

    return std::make_shared<ov::Model>(one_hot, ParameterVector{});
}

}  // namespace

TEST_F(TransformationTestsF, ConvertOneHot16To1) {
    auto pass = manager.register_pass<ov::pass::ConvertOneHot16To1>();
    model = create_model([](const Output<Node>& indices,
                            const Output<Node>& depth,
                            const Output<Node>& on_value,
                            const Output<Node>& off_value,
                            int64_t axis) {
        return std::make_shared<op::v16::OneHot>(indices,
                                                 depth,
                                                 on_value,
                                                 off_value,
                                                 axis,
                                                 op::v16::OneHot::NegativeIndicesMode::IGNORE_NEGATIVE);
    });
    model_ref = create_model([](const Output<Node>& indices,
                                const Output<Node>& depth,
                                const Output<Node>& on_value,
                                const Output<Node>& off_value,
                                int64_t axis) {
        return std::make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    });

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertOneHot16To1NotApplied) {
    auto pass = manager.register_pass<ov::pass::ConvertOneHot16To1>();
    model = create_model([](const Output<Node>& indices,
                            const Output<Node>& depth,
                            const Output<Node>& on_value,
                            const Output<Node>& off_value,
                            int64_t axis) {
        return std::make_shared<op::v16::OneHot>(indices,
                                                 depth,
                                                 on_value,
                                                 off_value,
                                                 axis,
                                                 op::v16::OneHot::NegativeIndicesMode::NORMALIZE);
    });
    model_ref = model->clone();

    // First: explicit check if transformation returns false on not supported version of OneHot
    auto one_hot_node = model->get_ordered_ops()[4];
    ASSERT_EQ(one_hot_node->get_type_info(), ov::op::v16::OneHot::get_type_info_static());
    ASSERT_FALSE(pass->apply(one_hot_node));

    // Second: explicit check that model is not changed after transformation.
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
