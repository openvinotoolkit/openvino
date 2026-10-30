// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/recurrent_cache_update.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"

namespace {
ov::OutputVector conv_inputs() {
    using namespace ov;
    OutputVector args;
    for (const auto& s : std::vector<PartialShape>{{1, 1, 12, 21},
                                                   {1, 1, 1, -1},
                                                   {1, 1, 1, -1},
                                                   {1, 1, 1, 1},
                                                   {1, 1, 1, 1},
                                                   {1, 1, 1, 2}})
        args.push_back(std::make_shared<op::v0::Parameter>(args.empty() ? element::f32 : element::i64, s));
    args.push_back(std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, -1, 7, -1}));
    args.push_back(std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1, 1, -1, 2}));
    args.push_back(std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 1, 7, 4}));
    return args;
}
ov::OutputVector gdn_inputs() {
    using namespace ov;
    auto p = [](element::Type t, PartialShape s) {
        return std::make_shared<op::v0::Parameter>(t, s);
    };
    return {p(element::f16, {-1, -1, 2, 128}),
            p(element::f16, {-1, -1, 2, 128}),
            p(element::f16, {-1, -1, 4, 8}),
            p(element::f32, {1, 1, 12, 4096}),
            p(element::f16, {-1, -1, 4}),
            p(element::f16, {-1, -1, 4}),
            p(element::i32, {1, 1, 1, -1}),
            p(element::i64, {1, 1, 1, -1}),
            p(element::i32, {1, 1, 1, 1}),
            p(element::i64, {1, 1, 1, 1}),
            p(element::i32, {1, 1, 1, 2}),
            p(element::i64, {1, 1, 1, 4})};
}
}  // namespace
TEST(type_prop, recurrent_conv_cache_update) {
    using namespace ov;
    auto args = conv_inputs();
    auto op = std::make_shared<op::internal::RecurrentConvCacheUpdate>(args, 3);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, -1, -1, 7}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 1, 12, 21}));
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_element_type(1), element::f32);
    auto clone = as_type_ptr<op::internal::RecurrentConvCacheUpdate>(op->clone_with_new_inputs(args));
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->get_history(), 3);
    EXPECT_THROW(std::make_shared<op::internal::RecurrentConvCacheUpdate>(args, 0), NodeValidationFailure);
    args.pop_back();
    EXPECT_THROW(std::make_shared<op::internal::RecurrentConvCacheUpdate>(args, 3), NodeValidationFailure);
}
TEST(type_prop, recurrent_gdn_update) {
    using namespace ov;
    auto args = gdn_inputs();
    auto op = std::make_shared<op::internal::RecurrentGatedDeltaNetUpdate>(args, true, 1e-5f, 2e-5f, 3);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, -1, 4, 8}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 1, 12, 4096}));
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_element_type(1), element::f32);
    auto clone = as_type_ptr<op::internal::RecurrentGatedDeltaNetUpdate>(op->clone_with_new_inputs(args));
    ASSERT_NE(clone, nullptr);
    EXPECT_TRUE(clone->get_fuse_qk_l2norm());
    EXPECT_FLOAT_EQ(clone->get_q_l2_norm_eps(), 1e-5f);
    EXPECT_FLOAT_EQ(clone->get_k_l2_norm_eps(), 2e-5f);
    EXPECT_EQ(clone->get_snapshot_count(), 3);
    EXPECT_THROW(std::make_shared<op::internal::RecurrentGatedDeltaNetUpdate>(args, false, 1e-6f, 1e-6f, 0),
                 NodeValidationFailure);
    args[11] = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 1, 1, 3});
    EXPECT_THROW(std::make_shared<op::internal::RecurrentGatedDeltaNetUpdate>(args), NodeValidationFailure);
}
