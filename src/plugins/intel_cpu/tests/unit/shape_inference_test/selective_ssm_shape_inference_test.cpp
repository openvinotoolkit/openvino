// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, SelectiveSSM) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto x = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto C = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto state = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto op = std::make_shared<op::internal::SelectiveSSM>(A, dt, B, x, C, state);

    const std::vector<StaticShape> input_shapes = {StaticShape{4},
                                                   StaticShape{2, 5, 4},
                                                   StaticShape{2, 5, 2, 16},
                                                   StaticShape{2, 5, 4, 8},
                                                   StaticShape{2, 5, 2, 16},
                                                   StaticShape{2, 4, 8, 16}};
    const auto output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({2, 5, 4, 8}));
    EXPECT_EQ(output_shapes[1], StaticShape({2, 4, 8, 16}));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMLogicalAndPhysicalBlocksDiffer) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    auto B = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    auto x = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    auto C = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3));
    auto state = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));
    auto block_indices = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));
    auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));
    auto processed = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));
    auto interval = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));
    auto op = std::make_shared<op::internal::PagedSelectiveSSM>(A,
                                                                dt,
                                                                B,
                                                                x,
                                                                C,
                                                                state,
                                                                subsequence_begins,
                                                                block_indices,
                                                                block_indices_begins,
                                                                processed,
                                                                interval);

    const std::vector<StaticShape> input_shapes = {StaticShape{4},
                                                   StaticShape{6, 4},
                                                   StaticShape{6, 2, 16},
                                                   StaticShape{6, 4, 8},
                                                   StaticShape{6, 2, 16},
                                                   StaticShape{3, 4, 8, 16},
                                                   StaticShape{3},
                                                   StaticShape{5},
                                                   StaticShape{3},
                                                   StaticShape{2},
                                                   StaticShape{2}};
    const auto output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({6, 4, 8}));
}
