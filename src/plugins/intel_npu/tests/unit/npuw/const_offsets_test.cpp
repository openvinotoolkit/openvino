// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/runtime/tensor.hpp"
#include "serialization.hpp"

namespace {

using namespace ov;

std::shared_ptr<op::v0::Constant> make_constant(const Tensor& tensor, std::size_t bin_offset) {
    auto constant = std::make_shared<op::v0::Constant>(tensor);
    constant->get_rt_info()[WeightlessCacheAttribute::get_type_info_static()] =
        WeightlessCacheAttribute(tensor.get_byte_size(), bin_offset, tensor.get_element_type());
    return constant;
}

std::shared_ptr<Model> make_model(const std::vector<std::shared_ptr<op::v0::Constant>>& constants) {
    std::shared_ptr<Node> sum = constants.front();
    for (size_t i = 1; i < constants.size(); ++i) {
        sum = std::make_shared<op::v1::Add>(sum, constants[i]);
    }
    return std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(sum)}, ParameterVector{});
}

}  // namespace

TEST(CollectConstOffsetsTest, KeepsConstantsWithDistinctBuffers) {
    Tensor first(element::f32, Shape{4});
    Tensor second(element::f32, Shape{4});
    const auto c1 = make_constant(first, 0);
    const auto c2 = make_constant(second, 16);

    const auto offsets = npuw::s11n::collect_const_offsets(make_model({c1, c2}));

    ASSERT_EQ(offsets.size(), 2u);
    EXPECT_EQ(offsets.at(c1->get_data_ptr()), 0u);
    EXPECT_EQ(offsets.at(c2->get_data_ptr()), 16u);
}

TEST(CollectConstOffsetsTest, KeepsSharedBufferWithEqualOffsets) {
    Tensor shared(element::f32, Shape{4});
    const auto c1 = make_constant(shared, 32);
    const auto c2 = make_constant(shared, 32);
    ASSERT_EQ(c1->get_data_ptr(), c2->get_data_ptr());

    const auto offsets = npuw::s11n::collect_const_offsets(make_model({c1, c2}));

    ASSERT_EQ(offsets.size(), 1u);
    EXPECT_EQ(offsets.at(c1->get_data_ptr()), 32u);
}

TEST(CollectConstOffsetsTest, DropsOnlyTheSharedBufferWithDifferentOffsets) {
    Tensor shared(element::f32, Shape{4});
    Tensor other(element::f32, Shape{4});
    const auto c1 = make_constant(shared, 0);
    const auto c2 = make_constant(shared, 64);
    const auto c3 = make_constant(other, 128);
    ASSERT_EQ(c1->get_data_ptr(), c2->get_data_ptr());

    const auto offsets = npuw::s11n::collect_const_offsets(make_model({c1, c2, c3}));

    EXPECT_EQ(offsets.count(c1->get_data_ptr()), 0u);
    ASSERT_EQ(offsets.size(), 1u);
    EXPECT_EQ(offsets.at(c3->get_data_ptr()), 128u);
}

TEST(CollectConstOffsetsTest, SkipsConstantsWithoutWeightlessAttribute) {
    Tensor with_attr(element::f32, Shape{4});
    Tensor without_attr(element::f32, Shape{4});
    const auto c1 = make_constant(with_attr, 8);
    const auto c2 = std::make_shared<op::v0::Constant>(without_attr);

    const auto offsets = npuw::s11n::collect_const_offsets(make_model({c1, c2}));

    ASSERT_EQ(offsets.size(), 1u);
    EXPECT_EQ(offsets.at(c1->get_data_ptr()), 8u);
}
