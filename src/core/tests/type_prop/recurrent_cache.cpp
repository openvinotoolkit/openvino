// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/recurrent_cache.hpp"
#include "openvino/op/parameter.hpp"
#include <gtest/gtest.h>

namespace {
ov::OutputVector cache_inputs() {
    using namespace ov;
    OutputVector args;
    for (const auto &shape : std::vector<PartialShape>{
             {1, 1, 12, 21}, {1, 1, 1, -1}, {1, 1, 1, -1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 2}}) {
        args.push_back(std::make_shared<op::v0::Parameter>(args.empty() ? element::f32 : element::i32, shape));
    }
    return args;
}
TEST(type_prop, recurrent_cache_dynamic) {
    auto args = cache_inputs();
    auto gather = std::make_shared<ov::op::internal::RecurrentStateGather>(args);
    EXPECT_EQ(gather->get_output_partial_shape(0), (ov::PartialShape{1, 1, -1, 21}));
    EXPECT_EQ(gather->get_output_partial_shape(1), (ov::PartialShape{1, 1, 12, 21}));
    args.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 7, -1}));
    args.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 1, -1, 2}));
    auto conv = std::make_shared<ov::op::internal::RecurrentConvCache>(args, 3);
    EXPECT_EQ(conv->get_output_partial_shape(0), (ov::PartialShape{1, -1, 7, ov::Dimension(3, -1)}));
    EXPECT_EQ(conv->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(conv->get_output_element_type(1), ov::element::f32);
    EXPECT_EQ(ov::as_type_ptr<ov::op::internal::RecurrentConvCache>(conv->clone_with_new_inputs(args))->get_history(),
              3);
}
TEST(type_prop, recurrent_state_update_validation) {
    using namespace ov;
    OutputVector args{std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 1, 12, 35}),
                      std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 1, -1, 35}),
                      std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1, 1, 1, 4})};
    auto update = std::make_shared<op::internal::RecurrentStateUpdate>(args, 5, 7);
    EXPECT_EQ(update->get_output_partial_shape(0), (PartialShape{1, 1, 12, 35}));
    EXPECT_EQ(update->get_output_element_type(0), element::f32);
    auto clone = ov::as_type_ptr<op::internal::RecurrentStateUpdate>(update->clone_with_new_inputs(args));
    EXPECT_EQ(clone->get_key_size(), 5);
    EXPECT_EQ(clone->get_value_size(), 7);
    EXPECT_THROW(std::make_shared<op::internal::RecurrentStateUpdate>(args, 0, 7), NodeValidationFailure);
    EXPECT_THROW(std::make_shared<op::internal::RecurrentStateUpdate>(args, 6, 7), NodeValidationFailure);
    args[2] = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1, 1, 1, 3});
    EXPECT_THROW(std::make_shared<op::internal::RecurrentStateUpdate>(args), NodeValidationFailure);
}
TEST(type_prop, recurrent_cache_validation) {
    using namespace ov;
    auto args = cache_inputs();
    args[3] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 1, 1, 1});
    EXPECT_THROW(std::make_shared<op::internal::RecurrentStateGather>(args), NodeValidationFailure);
    args = cache_inputs();
    args[0] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    EXPECT_THROW(std::make_shared<op::internal::RecurrentStateGather>(args), NodeValidationFailure);
    args = cache_inputs();
    args.push_back(std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, -1, 8, -1}));
    args.push_back(std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 1, -1, 2}));
    EXPECT_THROW(std::make_shared<op::internal::RecurrentConvCache>(args, 3), NodeValidationFailure);
    EXPECT_THROW(std::make_shared<op::internal::RecurrentConvCache>(args, 0), NodeValidationFailure);
}
} // namespace
