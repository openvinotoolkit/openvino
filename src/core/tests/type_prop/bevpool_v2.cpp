// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bevpool_v2.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace test {

using op::v0::Constant;
using op::v0::Parameter;
using testing::HasSubstr;

class TypePropBevPoolV2Test : public TypePropOpTest<op::v15::BevPoolV2> {
protected:
    static op::v15::Bound default_bound() {
        return op::v15::Bound{0.f, 1.f, 0.5f};
    }
};

TEST_F(TypePropBevPoolV2Test, default_ctor) {
    const auto op = make_op();

    const auto cf = std::make_shared<Parameter>(element::f32, PartialShape{2, 4, 3, 5});
    const auto dw = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 3, 5});
    const auto idx = std::make_shared<Parameter>(element::i32, PartialShape{6});
    const auto itv = std::make_shared<Parameter>(element::i32, PartialShape{6});

    op->set_arguments(OutputVector{cf, dw, idx, itv});
    op->set_input_channels(4);
    op->set_output_channels(2);
    op->set_image_width(5);
    op->set_image_height(3);
    op->set_feature_width(7);
    op->set_feature_height(6);
    op->set_x_bound(default_bound());
    op->set_y_bound(default_bound());
    op->set_z_bound(default_bound());
    op->set_d_bound(default_bound());
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 2, 6, 7}));
}

TEST_F(TypePropBevPoolV2Test, static_shapes) {
    const auto cf = std::make_shared<Parameter>(element::f16, PartialShape{3, 8, 10, 12});
    const auto dw = std::make_shared<Parameter>(element::f16, PartialShape{3, 4, 10, 12});
    const auto idx = std::make_shared<Parameter>(element::i64, PartialShape{9});
    const auto itv = std::make_shared<Parameter>(element::i64, PartialShape{3, 3});

    const auto op = std::make_shared<op::v15::BevPoolV2>(OutputVector{cf, dw, idx, itv},
                                                          8,
                                                          16,
                                                          12,
                                                          10,
                                                          20,
                                                          30,
                                                          default_bound(),
                                                          default_bound(),
                                                          default_bound(),
                                                          default_bound());

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 16, 30, 20}));
}

TEST_F(TypePropBevPoolV2Test, invalid_input_count) {
    const auto cf = std::make_shared<Parameter>(element::f32, PartialShape{2, 4, 3, 5});
    const auto dw = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 3, 5});
    const auto idx = std::make_shared<Parameter>(element::i64, PartialShape{6});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v15::BevPoolV2>(OutputVector{cf, dw, idx},
                                                                        4,
                                                                        2,
                                                                        5,
                                                                        3,
                                                                        7,
                                                                        6,
                                                                        default_bound(),
                                                                        default_bound(),
                                                                        default_bound(),
                                                                        default_bound()),
                    NodeValidationFailure,
                    HasSubstr("BevPoolV2 expects exactly 4 inputs"));
}

TEST_F(TypePropBevPoolV2Test, cf_rank_must_be_4d) {
    const auto cf = std::make_shared<Parameter>(element::f32, PartialShape{2, 4, 3});
    const auto dw = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 3, 5});
    const auto idx = std::make_shared<Parameter>(element::i32, PartialShape{6});
    const auto itv = std::make_shared<Parameter>(element::i32, PartialShape{6});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v15::BevPoolV2>(OutputVector{cf, dw, idx, itv},
                                                                        4,
                                                                        2,
                                                                        5,
                                                                        3,
                                                                        7,
                                                                        6,
                                                                        default_bound(),
                                                                        default_bound(),
                                                                        default_bound(),
                                                                        default_bound()),
                    NodeValidationFailure,
                    HasSubstr("Input 0 (cf) rank must be compatible with 4"));
}

TEST_F(TypePropBevPoolV2Test, cf_channel_dim_must_match_attribute) {
    const auto cf = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 3, 5});
    const auto dw = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 3, 5});
    const auto idx = std::make_shared<Parameter>(element::i64, PartialShape{6});
    const auto itv = std::make_shared<Parameter>(element::i64, PartialShape{6});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v15::BevPoolV2>(OutputVector{cf, dw, idx, itv},
                                                                        4,
                                                                        2,
                                                                        5,
                                                                        3,
                                                                        7,
                                                                        6,
                                                                        default_bound(),
                                                                        default_bound(),
                                                                        default_bound(),
                                                                        default_bound()),
                    NodeValidationFailure,
                    HasSubstr("Input 0 (cf) channel dimension does not match input_channels attribute"));
}

TEST_F(TypePropBevPoolV2Test, itv_1d_length_must_be_divisible_by_3) {
    const auto cf = std::make_shared<Parameter>(element::f32, PartialShape{2, 4, 3, 5});
    const auto dw = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 3, 5});
    const auto idx = std::make_shared<Parameter>(element::i32, PartialShape{7});
    const auto itv = std::make_shared<Parameter>(element::i32, PartialShape{7});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v15::BevPoolV2>(OutputVector{cf, dw, idx, itv},
                                                                        4,
                                                                        2,
                                                                        5,
                                                                        3,
                                                                        7,
                                                                        6,
                                                                        default_bound(),
                                                                        default_bound(),
                                                                        default_bound(),
                                                                        default_bound()),
                    NodeValidationFailure,
                    HasSubstr("Input 3 (itv) 1D length must be divisible by 3"));
}

TEST_F(TypePropBevPoolV2Test, idx_must_be_integral) {
    const auto cf = std::make_shared<Parameter>(element::f32, PartialShape{2, 4, 3, 5});
    const auto dw = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 3, 5});
    const auto idx = std::make_shared<Parameter>(element::f32, PartialShape{6});
    const auto itv = std::make_shared<Parameter>(element::i32, PartialShape{6});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v15::BevPoolV2>(OutputVector{cf, dw, idx, itv},
                                                                        4,
                                                                        2,
                                                                        5,
                                                                        3,
                                                                        7,
                                                                        6,
                                                                        default_bound(),
                                                                        default_bound(),
                                                                        default_bound(),
                                                                        default_bound()),
                    NodeValidationFailure,
                    HasSubstr("Input 2 (idx) must be an integer tensor"));
}

}  // namespace test
}  // namespace ov
