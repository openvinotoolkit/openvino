// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_selective_ssm.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/openvino.hpp"

namespace ov::test {
namespace {

std::shared_ptr<op::internal::PagedSelectiveSSM> make_paged_selective_ssm(const element::Type& et,
                                                                          const element::Type& ind_et,
                                                                          const PartialShape& A,
                                                                          const PartialShape& dt,
                                                                          const PartialShape& B,
                                                                          const PartialShape& x,
                                                                          const PartialShape& C,
                                                                          const PartialShape& state) {
    auto A_p = std::make_shared<op::v0::Parameter>(et, A);
    auto dt_p = std::make_shared<op::v0::Parameter>(et, dt);
    auto B_p = std::make_shared<op::v0::Parameter>(et, B);
    auto x_p = std::make_shared<op::v0::Parameter>(et, x);
    auto C_p = std::make_shared<op::v0::Parameter>(et, C);
    auto state_p = std::make_shared<op::v0::Parameter>(et, state);
    auto subseq = std::make_shared<op::v0::Parameter>(ind_et, PartialShape{-1});
    auto block_idx = std::make_shared<op::v0::Parameter>(ind_et, PartialShape{-1});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(ind_et, PartialShape{-1});
    auto processed = std::make_shared<op::v0::Parameter>(ind_et, PartialShape{-1});
    auto cache_interval = std::make_shared<op::v0::Parameter>(ind_et, PartialShape{-1});

    return std::make_shared<op::internal::PagedSelectiveSSM>(OutputVector{A_p,
                                                                          dt_p,
                                                                          B_p,
                                                                          x_p,
                                                                          C_p,
                                                                          state_p,
                                                                          subseq,
                                                                          block_idx,
                                                                          block_idx_begins,
                                                                          processed,
                                                                          cache_interval});
}

}  // namespace

TEST(type_prop, paged_selective_ssm_static_f32) {
    const auto op = make_paged_selective_ssm(element::f32,
                                             element::i32,
                                             Shape{4},
                                             Shape{6, 4},
                                             Shape{6, 2, 16},
                                             Shape{6, 4, 8},
                                             Shape{6, 2, 16},
                                             Shape{3, 4, 8, 16});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{6, 4, 8}));
}

TEST(type_prop, paged_selective_ssm_state_type_must_match) {
    auto A_p = std::make_shared<op::v0::Parameter>(element::f16, Shape{4});
    auto dt_p = std::make_shared<op::v0::Parameter>(element::f16, Shape{6, 4});
    auto B_p = std::make_shared<op::v0::Parameter>(element::f16, Shape{6, 2, 16});
    auto x_p = std::make_shared<op::v0::Parameter>(element::f16, Shape{6, 4, 8});
    auto C_p = std::make_shared<op::v0::Parameter>(element::f16, Shape{6, 2, 16});
    auto state_p = std::make_shared<op::v0::Parameter>(element::bf16, Shape{3, 4, 8, 16});
    auto subseq = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1});
    auto block_idx = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1});
    auto processed = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1});
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{-1});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::PagedSelectiveSSM>(OutputVector{A_p,
                                                                                                 dt_p,
                                                                                                 B_p,
                                                                                                 x_p,
                                                                                                 C_p,
                                                                                                 state_p,
                                                                                                 subseq,
                                                                                                 block_idx,
                                                                                                 block_idx_begins,
                                                                                                 processed,
                                                                                                 cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("all real-valued inputs"));
}

TEST(type_prop, paged_selective_ssm_bad_index_type) {
    OV_EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                           element::f32,
                                                           Shape{4},
                                                           Shape{6, 4},
                                                           Shape{6, 2, 16},
                                                           Shape{6, 4, 8},
                                                           Shape{6, 2, 16},
                                                           Shape{3, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("Integer inputs must have i32 or i64 element type."));
}

TEST(type_prop, paged_selective_ssm_heads_not_divisible_by_groups) {
    OV_EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                           element::i32,
                                                           Shape{4},
                                                           Shape{6, 4},
                                                           Shape{6, 3, 16},
                                                           Shape{6, 4, 8},
                                                           Shape{6, 3, 16},
                                                           Shape{3, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads should be divisible by the number of groups."));
}

}  // namespace ov::test
