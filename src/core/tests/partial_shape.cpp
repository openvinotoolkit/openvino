// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/partial_shape.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_tools.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;

TEST(partial_shape, interators) {
    const ov::PartialShape ps({1, 2, 3});
    ASSERT_TRUE(ps.is_static());
    {
        auto p = ps;
        for (auto& d : p) {
            d = Dimension::dynamic();
        }
        ASSERT_TRUE(p.is_dynamic());
    }
    {
        auto p = ps;
        auto it = p.begin();
        *it = Dimension::dynamic();
        ASSERT_TRUE(p.is_dynamic());
    }
    {
        auto p = ps;
        auto it = p.rbegin();
        *it = Dimension::dynamic();
        ASSERT_TRUE(p.is_dynamic());
    }
    {
        auto p = ps;
        auto it = p.end();
        --it;
        *it = Dimension::dynamic();
        ASSERT_TRUE(p.is_dynamic());
    }
    {
        auto p = ps;
        auto it = p.rend();
        --it;
        *it = Dimension::dynamic();
        ASSERT_TRUE(p.is_dynamic());
    }
}

TEST(partial_shape, ps_construction_empty) {
    auto ps = PartialShape{};
    ASSERT_TRUE(ps.rank().is_static());
    ASSERT_TRUE(ps.is_static());
    ASSERT_EQ(ps.rank().get_length(), 0);
}

TEST(partial_shape, ps_construction_rank_dynamic) {
    auto ps = PartialShape::dynamic();
    ASSERT_TRUE(ps.rank().is_dynamic());
    ASSERT_TRUE(ps.is_dynamic());
}

TEST(partial_shape, ps_construction_rank_static_shape_dynamic) {
    auto ps = PartialShape{2, Dimension::dynamic(), 3};
    ASSERT_TRUE(ps.rank().is_static());
    ASSERT_TRUE(ps.is_dynamic());
    ASSERT_EQ(ps.rank().get_length(), 3);
}

TEST(partial_shape, ps_construction_static) {
    auto ps = PartialShape{2, 5, 3, 6};
    ASSERT_TRUE(ps.rank().is_static());
    ASSERT_TRUE(ps.is_static());
    ASSERT_EQ(ps.rank().get_length(), 4);
}

TEST(partial_shape, dim_construction_static) {
    Dimension dim{3};
    ASSERT_EQ(dim.get_length(), 3);
    ASSERT_TRUE(dim.is_static());
}

TEST(partial_shape, dim_construction_dynamic) {
    Dimension dim = Dimension::dynamic();
    ASSERT_TRUE(dim.is_dynamic());
}

TEST(partial_shape, dim_conversion_dynamic) {
    EXPECT_ANY_THROW({ Dimension::dynamic().get_length(); });
}

TEST(partial_shape, rank_construction_static) {
    Rank r{4};
    ASSERT_EQ(r.get_length(), 4);
    ASSERT_TRUE(r.is_static());
}

TEST(partial_shape, rank_construction_dynamic) {
    Rank r = Rank::dynamic();
    ASSERT_TRUE(r.is_dynamic());
}

TEST(partial_shape, dim_compatible_left_dynamic) {
    Dimension d1{Dimension::dynamic()};
    Dimension d2{3};

    ASSERT_TRUE(d1.compatible(d2));
}

TEST(partial_shape, dim_compatible_right_dynamic) {
    Dimension d1{3};
    Dimension d2{Dimension::dynamic()};

    ASSERT_TRUE(d1.compatible(d2));
}

TEST(partial_shape, dim_compatible_both_dynamic) {
    Dimension d1{Dimension::dynamic()};
    Dimension d2{Dimension::dynamic()};

    ASSERT_TRUE(d1.compatible(d2));
}

TEST(partial_shape, dim_compatible_both_static) {
    Dimension d1{3};
    Dimension d2{8};
    Dimension d3{3};

    ASSERT_FALSE(d1.compatible(d2));
    ASSERT_TRUE(d1.compatible(d3));
}

TEST(partial_shape, shapes_compatible_both_rank_dynamic) {
    PartialShape ps1{PartialShape::dynamic()};
    PartialShape ps2{PartialShape::dynamic()};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_left_rank_dynamic) {
    PartialShape ps1{3};
    PartialShape ps2{PartialShape::dynamic()};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_right_rank_dynamic) {
    PartialShape ps1{PartialShape::dynamic()};
    PartialShape ps2{4};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_both_partial_all_known_equal) {
    PartialShape ps1{2, Dimension::dynamic(), 3, Dimension::dynamic(), 5};
    PartialShape ps2{2, Dimension::dynamic(), Dimension::dynamic(), 4, 5};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_both_partial_some_known_unequal) {
    PartialShape ps1{2, Dimension::dynamic(), 3, Dimension::dynamic(), 5};
    PartialShape ps2{1, Dimension::dynamic(), Dimension::dynamic(), 4, 5};

    ASSERT_FALSE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_compatible_both_static_different_rank) {
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 6, 8, 10};

    ASSERT_FALSE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_equal_both_static_same_rank_same_dims) {
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 6, 8};

    ASSERT_TRUE(ps1.compatible(ps2));
}

TEST(partial_shape, shapes_equal_both_static_same_rank_different_dims) {
    PartialShape ps1{2, 4, 6, 8};
    PartialShape ps2{2, 4, 3, 8};

    ASSERT_FALSE(ps1.compatible(ps2));
}

TEST(partial_shape, from_shape) {
    Shape s{2, 4, 6, 8};
    PartialShape ps1{s};

    ASSERT_TRUE(ps1.rank().is_static());
    ASSERT_EQ(ps1.rank().get_length(), s.size());
    ASSERT_TRUE(ps1.is_static());
    ASSERT_EQ(ps1[0].get_length(), 2);
    ASSERT_EQ(ps1[1].get_length(), 4);
    ASSERT_EQ(ps1[2].get_length(), 6);
    ASSERT_EQ(ps1[3].get_length(), 8);
}

TEST(partial_shape, to_shape_static) {
    PartialShape ps{2, 4, 6, 8};
    Shape s{ps.to_shape()};

    ASSERT_EQ(s, (Shape{2, 4, 6, 8}));
}

TEST(partial_shape, to_shape_dims_dynamic) {
    PartialShape ps{2, 4, Dimension::dynamic(), 8};
    ASSERT_THROW({ ps.to_shape(); }, ov::Exception);
}

TEST(partial_shape, to_shape_rank_dynamic) {
    PartialShape ps{PartialShape::dynamic()};
    ASSERT_THROW({ ps.to_shape(); }, ov::Exception);
}

TEST(partial_shape, tensor_descriptor_from_shape) {
    descriptor::Tensor t{element::i32, Shape{1, 2, 3}};

    ASSERT_EQ(t.get_partial_shape().to_shape(), (Shape{1, 2, 3}));
    ASSERT_EQ(t.get_partial_shape().rank().get_length(), 3);
    ASSERT_TRUE(t.get_partial_shape().same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, tensor_descriptor_from_static_partial_shape) {
    descriptor::Tensor t{element::i32, PartialShape{1, 2, 3}};

    ASSERT_EQ(t.get_partial_shape().to_shape(), (Shape{1, 2, 3}));
    ASSERT_EQ(t.get_partial_shape().rank().get_length(), 3);
    ASSERT_TRUE(t.get_partial_shape().same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, tensor_descriptor_from_rank_static_dynamic_partial_shape) {
    descriptor::Tensor t{element::i32, PartialShape{1, Dimension::dynamic(), 3}};

    ASSERT_EQ(t.get_partial_shape().rank().get_length(), 3);
    ASSERT_THROW({ t.get_partial_shape().to_shape(); }, ov::Exception);
    ASSERT_TRUE(t.get_partial_shape().same_scheme(PartialShape{1, Dimension::dynamic(), 3}));
}

TEST(partial_shape, tensor_descriptor_from_rank_dynamic_partial_shape) {
    descriptor::Tensor t{element::i32, PartialShape::dynamic()};

    ASSERT_TRUE(t.get_partial_shape().rank().is_dynamic());
    ASSERT_THROW({ t.get_partial_shape().to_shape(); }, ov::Exception);
    ASSERT_TRUE(t.get_partial_shape().same_scheme(PartialShape::dynamic()));
}

TEST(partial_shape, dim_same_scheme_both_dynamic) {
    ASSERT_TRUE(Dimension::dynamic().same_scheme(Dimension::dynamic()));
}

TEST(partial_shape, dim_same_scheme_left_dynamic) {
    ASSERT_FALSE(Dimension::dynamic().same_scheme(6));
}

TEST(partial_shape, dim_same_scheme_right_dynamic) {
    ASSERT_FALSE(Dimension(6).same_scheme(Dimension::dynamic()));
}

TEST(partial_shape, dim_same_scheme_both_static_same) {
    ASSERT_TRUE(Dimension(6).same_scheme(Dimension(6)));
}

TEST(partial_shape, dim_same_scheme_both_static_different) {
    ASSERT_FALSE(Dimension(6).same_scheme(Dimension(7)));
}

TEST(partial_shape, partial_shape_same_scheme_both_dynamic) {
    ASSERT_TRUE(PartialShape::dynamic().same_scheme(PartialShape::dynamic()));
}

TEST(partial_shape, partial_shape_same_scheme_left_dynamic_right_rank_static_dynamic) {
    ASSERT_FALSE(PartialShape::dynamic().same_scheme(PartialShape{1, Dimension::dynamic(), 3}));
}

TEST(partial_shape, partial_shape_same_scheme_left_dynamic_right_static) {
    ASSERT_FALSE(PartialShape::dynamic().same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, partial_shape_same_scheme_right_dynamic_left_rank_static_dynamic) {
    ASSERT_FALSE((PartialShape{1, Dimension::dynamic(), 3}.same_scheme(PartialShape::dynamic())));
}

TEST(partial_shape, partial_shape_same_scheme_right_dynamic_left_static) {
    ASSERT_FALSE((PartialShape{1, 2, 3}.same_scheme(PartialShape::dynamic())));
}

TEST(partial_shape, partial_shape_same_scheme_both_static_different_rank) {
    ASSERT_FALSE((PartialShape{1, 2, 3}.same_scheme(PartialShape{1, 2, 3, 4})));
}

TEST(partial_shape, partial_shape_same_scheme_both_rank_static_dynamic_different_rank) {
    ASSERT_FALSE((PartialShape{1, Dimension::dynamic(), 3}.same_scheme(PartialShape{1, Dimension::dynamic(), 3, 4})));
}

TEST(partial_shape, partial_shape_same_scheme_both_static_same_rank_different_dims) {
    ASSERT_FALSE((PartialShape{1, 2, 3}.same_scheme(PartialShape{1, 3, 3})));
}

TEST(partial_shape, partial_shape_same_scheme_both_rank_static_dynamic_same_rank_different_dims) {
    ASSERT_FALSE((PartialShape{1, 2, Dimension::dynamic()}.same_scheme(PartialShape{1, 3, Dimension::dynamic()})));
}

TEST(partial_shape, partial_shape_same_scheme_both_rank_static_dynamic_same_rank_compatible_not_same) {
    ASSERT_FALSE((PartialShape{1, 2, Dimension::dynamic()}.same_scheme(PartialShape{1, Dimension::dynamic(), 3})));
}

TEST(partial_shape, partial_shape_same_scheme_both_rank_static_dynamic_same_rank_compatible_same) {
    ASSERT_TRUE((PartialShape{1, 2, Dimension::dynamic()}.same_scheme(PartialShape{1, 2, Dimension::dynamic()})));
}

TEST(partial_shape, partial_shape_same_scheme_both_static_same_rank_same_dims) {
    ASSERT_TRUE((PartialShape{1, 2, 3}.same_scheme(PartialShape{1, 2, 3})));
}

TEST(partial_shape, partial_shape_same_scheme_scalar) {
    ASSERT_TRUE((PartialShape{}.same_scheme(PartialShape{})));
}

TEST(partial_shape, dim_merge_both_dynamic) {
    Dimension d;
    ASSERT_TRUE(Dimension::merge(d, Dimension::dynamic(), Dimension::dynamic()));
    ASSERT_TRUE(d.is_dynamic());
}

TEST(partial_shape, dim_merge_left_dynamic) {
    Dimension d;
    ASSERT_TRUE(Dimension::merge(d, Dimension::dynamic(), 3));
    ASSERT_TRUE(d.is_static());
    ASSERT_EQ(d.get_length(), 3);
}

TEST(partial_shape, dim_merge_right_dynamic) {
    Dimension d;
    ASSERT_TRUE(Dimension::merge(d, 3, Dimension::dynamic()));
    ASSERT_TRUE(d.is_static());
    ASSERT_EQ(d.get_length(), 3);
}

TEST(partial_shape, dim_merge_both_static_equal) {
    Dimension d;
    ASSERT_TRUE(Dimension::merge(d, 3, 3));
    ASSERT_TRUE(d.is_static());
    ASSERT_EQ(d.get_length(), 3);
}

TEST(partial_shape, dim_merge_both_static_unequal) {
    Dimension d = 163;
    ASSERT_FALSE(Dimension::merge(d, 3, 4));
    ASSERT_TRUE(d.is_static());
    ASSERT_EQ(d.get_length(), 163);
}

TEST(partial_shape, partial_shape_merge_both_rank_dynamic) {
    PartialShape s1{PartialShape::dynamic()};
    const PartialShape s2{PartialShape::dynamic()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.rank().is_dynamic());
}

TEST(partial_shape, partial_shape_merge_left_rank_dynamic_right_rank_static_dynamic) {
    PartialShape s1{PartialShape::dynamic()};
    const PartialShape s2{1, 2, Dimension::dynamic()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, Dimension::dynamic()}));
}

TEST(partial_shape, partial_shape_merge_left_rank_dynamic_right_static) {
    PartialShape s1{PartialShape::dynamic()};
    const PartialShape s2{1, 2, 3};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, partial_shape_merge_left_rank_static_dynamic_right_rank_dynamic) {
    PartialShape s1{1, 2, Dimension::dynamic()};
    const PartialShape s2{PartialShape::dynamic()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, Dimension::dynamic()}));
}

TEST(partial_shape, partial_shape_merge_left_static_right_rank_dynamic) {
    PartialShape s1{1, 2, 3};
    const PartialShape s2{PartialShape::dynamic()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, partial_shape_merge_both_rank_static_dynamic_consistent) {
    PartialShape s1{1, Dimension::dynamic(), 3, Dimension::dynamic()};
    const PartialShape s2{1, 2, Dimension::dynamic(), Dimension::dynamic()};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, 3, Dimension::dynamic()}));
}

TEST(partial_shape, partial_shape_merge_both_rank_static_dynamic_same_rank_inconsistent) {
    PartialShape s1{1, Dimension::dynamic(), 3, Dimension::dynamic()};
    const PartialShape s2{2, 2, Dimension::dynamic(), Dimension::dynamic()};
    ASSERT_FALSE(PartialShape::merge_into(s1, s2));
}

TEST(partial_shape, partial_shape_merge_both_rank_static_dynamic_different_rank) {
    PartialShape s1{1, Dimension::dynamic(), 3, Dimension::dynamic()};
    const PartialShape s2{1, 2, Dimension::dynamic()};
    ASSERT_FALSE(PartialShape::merge_into(s1, s2));
}

TEST(partial_shape, partial_shape_merge_both_static_consistent) {
    PartialShape s1{1, 2, 3};
    const PartialShape s2{1, 2, 3};
    ASSERT_TRUE(PartialShape::merge_into(s1, s2));
    ASSERT_TRUE(s1.same_scheme(PartialShape{1, 2, 3}));
}

TEST(partial_shape, partial_shape_merge_both_static_inconsistent) {
    PartialShape s1{1, 2, 3};
    const PartialShape s2{1, 2, 4};
    ASSERT_FALSE(PartialShape::merge_into(s1, s2));
}

TEST(partial_shape, partial_shape_merge_both_static_different_rank) {
    PartialShape s1{1, 2, 3};
    const PartialShape s2{1, 2, 3, 4};
    ASSERT_FALSE(PartialShape::merge_into(s1, s2));
}

TEST(partial_shape, partial_shape_broadcast_merge_into_fails) {
    PartialShape s1{2, Dimension::dynamic(), 3, 4};
    ASSERT_FALSE(PartialShape::broadcast_merge_into(s1, PartialShape{3}, op::AutoBroadcastType::NUMPY));
    ASSERT_FALSE(PartialShape::broadcast_merge_into(s1, PartialShape{4, 4}, op::AutoBroadcastType::NUMPY));
    ASSERT_FALSE(PartialShape::broadcast_merge_into(s1, PartialShape{2, 5, 3, 3, 4}, op::AutoBroadcastType::NUMPY));
}

TEST(partial_shape, partial_shape_broadcast_merge_into_dynamic_rank) {
    PartialShape s1{PartialShape::dynamic()};
    ASSERT_TRUE(PartialShape::broadcast_merge_into(s1, PartialShape{3, 2, 4}, op::AutoBroadcastType::NUMPY));
    ASSERT_TRUE(s1.same_scheme(PartialShape::dynamic()));

    PartialShape s2{2, Dimension::dynamic()};
    ASSERT_TRUE(PartialShape::broadcast_merge_into(s2, PartialShape::dynamic(), op::AutoBroadcastType::NUMPY));
    ASSERT_TRUE(s2.same_scheme(PartialShape::dynamic()));
}

TEST(partial_shape, partial_shape_broadcast_merge_into) {
    PartialShape s1{5, Dimension::dynamic(), 3, 4};
    const PartialShape s2{3, 4};
    ASSERT_TRUE(PartialShape::broadcast_merge_into(s1, s2, op::AutoBroadcastType::NUMPY));
    ASSERT_TRUE(s1.same_scheme(PartialShape{5, Dimension::dynamic(), 3, 4}));

    PartialShape s3{Dimension::dynamic()};
    ASSERT_TRUE(PartialShape::broadcast_merge_into(s3, s2, op::AutoBroadcastType::NUMPY));
    ASSERT_TRUE(s3.same_scheme(PartialShape{3, 4}));

    PartialShape s4{2, 4, 1, 5};
    ASSERT_TRUE(PartialShape::broadcast_merge_into(s4, PartialShape{2, 1, 3, 5}, op::AutoBroadcastType::NUMPY));
    ASSERT_TRUE(s4.same_scheme(PartialShape{2, 4, 3, 5}));
}

TEST(partial_shape, dim_pluseq_left_dynamic) {
    Dimension d1{Dimension::dynamic()};
    Dimension d2{2};

    d1 += d2;

    ASSERT_TRUE(d1.is_dynamic());
}

TEST(partial_shape, dim_pluseq_right_dynamic) {
    Dimension d1{2};
    Dimension d2{Dimension::dynamic()};

    d1 += d2;

    ASSERT_TRUE(d1.is_dynamic());
}

TEST(partial_shape, dim_pluseq_both_static) {
    Dimension d1{3};
    Dimension d2{2};

    d1 += d2;

    ASSERT_TRUE(d1.is_static());
    ASSERT_EQ(d1.get_length(), 5);
}

TEST(partial_shape, dim_timeseq_left_dynamic_right_nonzero) {
    Dimension d1{Dimension::dynamic()};
    Dimension d2{2};

    d1 *= d2;

    ASSERT_TRUE(d1.is_dynamic());
}

TEST(partial_shape, dim_timeseq_left_dynamic_right_zero) {
    Dimension d1{Dimension::dynamic()};
    Dimension d2{0};

    d1 *= d2;

    ASSERT_TRUE(d1.is_static());
    ASSERT_EQ(d1.get_length(), 0);
}

TEST(partial_shape, dim_timeseq_right_dynamic_left_nonzero) {
    Dimension d1{2};
    Dimension d2{Dimension::dynamic()};

    d1 *= d2;

    ASSERT_TRUE(d1.is_dynamic());
}

TEST(partial_shape, dim_timeseq_right_dynamic_left_zero) {
    Dimension d1{0};
    Dimension d2{Dimension::dynamic()};

    d1 *= d2;

    ASSERT_TRUE(d1.is_static());
    ASSERT_EQ(d1.get_length(), 0);
}

TEST(partial_shape, dim_timeseq_both_static) {
    Dimension d1{3};
    Dimension d2{2};

    d1 *= d2;

    ASSERT_TRUE(d1.is_static());
    ASSERT_EQ(d1.get_length(), 6);
}

TEST(partial_shape, dim_relaxes_refines_dyn_dyn) {
    Dimension d1{Dimension::dynamic()};
    Dimension d2{Dimension::dynamic()};

    ASSERT_TRUE(d1.refines(d2));
    ASSERT_TRUE(d1.relaxes(d2));
    ASSERT_TRUE(d2.refines(d1));
    ASSERT_TRUE(d2.relaxes(d1));
}

TEST(partial_shape, dim_relaxes_refines_dyn_static) {
    Dimension d1{Dimension::dynamic()};
    Dimension d2{3};

    ASSERT_FALSE(d1.refines(d2));
    ASSERT_TRUE(d1.relaxes(d2));
    ASSERT_TRUE(d2.refines(d1));
    ASSERT_FALSE(d2.relaxes(d1));
}

TEST(partial_shape, dim_relaxes_refines_static_static_eq) {
    Dimension d1{3};
    Dimension d2{3};

    ASSERT_TRUE(d1.refines(d2));
    ASSERT_TRUE(d1.relaxes(d2));
    ASSERT_TRUE(d2.refines(d1));
    ASSERT_TRUE(d2.relaxes(d1));
}

TEST(partial_shape, dim_relaxes_refines_static_static_not_eq) {
    Dimension d1{3};
    Dimension d2{4};

    ASSERT_FALSE(d1.refines(d2));
    ASSERT_FALSE(d1.relaxes(d2));
    ASSERT_FALSE(d2.refines(d1));
    ASSERT_FALSE(d2.relaxes(d1));
}

TEST(partial_shape, partial_shape_relaxes_refines_rank_dynamic_rank_dynamic) {
    PartialShape s1{PartialShape::dynamic()};
    PartialShape s2{PartialShape::dynamic()};

    ASSERT_TRUE(s1.refines(s2));
    ASSERT_TRUE(s1.relaxes(s2));
    ASSERT_TRUE(s2.refines(s1));
    ASSERT_TRUE(s2.relaxes(s1));
}

TEST(partial_shape, partial_shape_relaxes_refines_rank_dynamic_rank_static_dynamic) {
    PartialShape s1{PartialShape::dynamic()};
    PartialShape s2{3, Dimension::dynamic(), 7, 9};

    ASSERT_FALSE(s1.refines(s2));
    ASSERT_TRUE(s1.relaxes(s2));
    ASSERT_TRUE(s2.refines(s1));
    ASSERT_FALSE(s2.relaxes(s1));
}

TEST(partial_shape, partial_shape_relaxes_refines_rank_dynamic_static) {
    PartialShape s1{PartialShape::dynamic()};
    PartialShape s2{3, 5, 7, 9};

    ASSERT_FALSE(s1.refines(s2));
    ASSERT_TRUE(s1.relaxes(s2));
    ASSERT_TRUE(s2.refines(s1));
    ASSERT_FALSE(s2.relaxes(s1));
}

TEST(partial_shape, partial_shape_relaxes_refines_rank_dynamic_static_rank_dynamic_static_incompatible) {
    PartialShape s1{3, 5, Dimension::dynamic(), 9};
    PartialShape s2{4, Dimension::dynamic(), 7, 9};

    ASSERT_FALSE(s1.refines(s2));
    ASSERT_FALSE(s1.relaxes(s2));
    ASSERT_FALSE(s2.refines(s1));
    ASSERT_FALSE(s2.relaxes(s1));
}

TEST(partial_shape, partial_shape_relaxes_refines_rank_dynamic_static_rank_dynamic_static_compatible_neither) {
    PartialShape s1{3, 5, Dimension::dynamic(), 9};
    PartialShape s2{3, Dimension::dynamic(), 7, 9};

    ASSERT_FALSE(s1.refines(s2));
    ASSERT_FALSE(s1.relaxes(s2));
    ASSERT_FALSE(s2.refines(s1));
    ASSERT_FALSE(s2.relaxes(s1));
}

TEST(partial_shape, partial_shape_relaxes_refines_rank_dynamic_static_rank_dynamic_static_compatible_one_way) {
    PartialShape s1{3, Dimension::dynamic(), Dimension::dynamic(), 9};
    PartialShape s2{3, Dimension::dynamic(), 7, 9};

    ASSERT_FALSE(s1.refines(s2));
    ASSERT_TRUE(s1.relaxes(s2));
    ASSERT_TRUE(s2.refines(s1));
    ASSERT_FALSE(s2.relaxes(s1));
}

TEST(partial_shape, partial_shape_relaxes_refines_rank_dynamic_static_rank_dynamic_static_compatible_both_ways) {
    PartialShape s1{3, Dimension::dynamic(), 7, 9};
    PartialShape s2{3, Dimension::dynamic(), 7, 9};

    ASSERT_TRUE(s1.refines(s2));
    ASSERT_TRUE(s1.relaxes(s2));
    ASSERT_TRUE(s2.refines(s1));
    ASSERT_TRUE(s2.relaxes(s1));
}

TEST(partial_shape, partial_shape_relaxes_refines_rank_dynamic_static_static_incompatible) {
    PartialShape s1{3, Dimension::dynamic(), 7, 9};
    PartialShape s2{4, 5, 7, 9};

    ASSERT_FALSE(s1.refines(s2));
    ASSERT_FALSE(s1.relaxes(s2));
    ASSERT_FALSE(s2.refines(s1));
    ASSERT_FALSE(s2.relaxes(s1));
}

TEST(partial_shape, partial_shape_relaxes_refines_rank_dynamic_static_static_compatible) {
    PartialShape s1{3, Dimension::dynamic(), 7, 9};
    PartialShape s2{3, 5, 7, 9};

    ASSERT_FALSE(s1.refines(s2));
    ASSERT_TRUE(s1.relaxes(s2));
    ASSERT_TRUE(s2.refines(s1));
    ASSERT_FALSE(s2.relaxes(s1));
}

TEST(partial_shape, partial_shape_relaxes_refines_static_static_eq) {
    PartialShape s1{3, 5, 7, 9};
    PartialShape s2{3, 5, 7, 9};

    ASSERT_TRUE(s1.refines(s2));
    ASSERT_TRUE(s1.relaxes(s2));
    ASSERT_TRUE(s2.refines(s1));
    ASSERT_TRUE(s2.relaxes(s1));
}

TEST(partial_shape, partial_shape_relaxes_refines_static_static_not_eq) {
    PartialShape s1{3, 5, 7, 9};
    PartialShape s2{4, 5, 7, 9};

    ASSERT_FALSE(s1.refines(s2));
    ASSERT_FALSE(s1.relaxes(s2));
    ASSERT_FALSE(s2.refines(s1));
    ASSERT_FALSE(s2.relaxes(s1));
}

OPENVINO_SUPPRESS_DEPRECATED_START
TEST(partial_shape, partial_shape_project_rank_dynamic) {
    PartialShape s1{PartialShape::dynamic()};
    PartialShape s2 = ngraph::project(s1, AxisSet{284, 0, 103});

    ASSERT_TRUE(s2.rank().is_dynamic());
}

TEST(partial_shape, partial_shape_project_rank_static_dynamic) {
    PartialShape s1{Dimension::dynamic(), 2, Dimension::dynamic(), 3};
    PartialShape s2 = ngraph::project(s1, AxisSet{0, 3});

    ASSERT_TRUE(s2.same_scheme(PartialShape{Dimension::dynamic(), 3}));
}

TEST(partial_shape, partial_shape_reduce_rank_dynamic) {
    PartialShape s1{PartialShape::dynamic()};
    PartialShape s2 = ngraph::reduce(s1, AxisSet{284, 0, 103}, false);

    ASSERT_TRUE(s2.rank().is_dynamic());
}

TEST(partial_shape, partial_shape_reduce_rank_static_dynamic) {
    PartialShape s1{Dimension::dynamic(), 2, Dimension::dynamic(), 3};
    PartialShape s2 = ngraph::reduce(s1, AxisSet{0, 3}, false);

    ASSERT_TRUE(s2.same_scheme(PartialShape{2, Dimension::dynamic()}));
}

TEST(partial_shape, partial_shape_inject_pairs_rank_dynamic) {
    PartialShape s1{PartialShape::dynamic()};
    PartialShape s2 =
        ngraph::inject_pairs(s1, std::vector<std::pair<size_t, Dimension>>{{0, Dimension::dynamic()}, {207, 909}});

    ASSERT_TRUE(s2.rank().is_dynamic());
}

TEST(partial_shape, partial_shape_inject_pairs_rank_static) {
    PartialShape s1{1, Dimension::dynamic()};
    PartialShape s2 = ngraph::inject_pairs(
        s1,
        std::vector<std::pair<size_t, Dimension>>{{0, Dimension::dynamic()}, {2, 909}, {4, Dimension::dynamic()}});

    ASSERT_TRUE(s2.same_scheme(PartialShape{Dimension::dynamic(), 1, 909, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(partial_shape, merge_rank_dyn_dyn) {
    PartialShape s{PartialShape::dynamic()};

    ASSERT_TRUE(s.merge_rank(Rank::dynamic()));
    ASSERT_TRUE(s.rank().is_dynamic());
}

TEST(partial_shape, merge_rank_dyn_static) {
    PartialShape s{PartialShape::dynamic()};

    ASSERT_TRUE(s.merge_rank(4));
    ASSERT_TRUE(s.same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(partial_shape, merge_rank_static_dyn) {
    PartialShape s{2, 3, Dimension::dynamic(), 5};

    ASSERT_TRUE(s.merge_rank(Rank::dynamic()));
    ASSERT_TRUE(s.same_scheme(PartialShape{2, 3, Dimension::dynamic(), 5}));
}

TEST(partial_shape, merge_rank_static_static_ok) {
    PartialShape s{2, 3, Dimension::dynamic(), 5};

    ASSERT_TRUE(s.merge_rank(4));
    ASSERT_TRUE(s.same_scheme(PartialShape{2, 3, Dimension::dynamic(), 5}));
}

TEST(partial_shape, merge_rank_static_static_fail) {
    PartialShape s{2, 3, Dimension::dynamic(), 5};

    ASSERT_FALSE(s.merge_rank(5));
    ASSERT_TRUE(s.same_scheme(PartialShape{2, 3, Dimension::dynamic(), 5}));
}

TEST(partial_shape, changed_dimension_by_reference) {
    PartialShape s{1, 2, 3};

    Dimension& d = s[1];

    ASSERT_TRUE(s.is_static());

    d = Dimension::dynamic();

    ASSERT_TRUE(s.is_dynamic());

    d = 2;

    ASSERT_TRUE(s.is_static());
}

TEST(partial_shape, emplace_back_new_dimension) {
    PartialShape s{2, 3, Dimension::dynamic(), 5};

    s.emplace_back(3, 5);

    ASSERT_EQ(s, PartialShape({2, 3, -1, 5, {3, 5}}));
}

TEST(partial_shape, copy_with_back_inserter_iterator) {
    PartialShape s{2, 3, Dimension::dynamic(), 5}, s_copy;

    std::copy(s.begin(), s.end(), std::back_inserter(s_copy));

    ASSERT_EQ(s_copy, s);
}

TEST(partial_shape, infer_windowed_reduction_rank_dynamic_rank_dynamic_ok) {
    auto node = std::make_shared<op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{PartialShape::dynamic()};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 0, 0};
    CoordinateDiff data_padding_above{0, 0, 0, 0};
    PartialShape window_shape{PartialShape::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;
    OPENVINO_SUPPRESS_DEPRECATED_START
    PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                              data_shape,
                                                                              data_dilation,
                                                                              data_padding_below,
                                                                              data_padding_above,
                                                                              window_shape,
                                                                              window_strides,
                                                                              window_dilation,
                                                                              is_window_all_in_padding_allowed);
    OPENVINO_SUPPRESS_DEPRECATED_END

    ASSERT_TRUE(result_shape.same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(partial_shape, infer_windowed_reduction_rank_dynamic_rank_dynamic_zero_data_dilation) {
    auto node = std::make_shared<op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{PartialShape::dynamic()};
    Strides data_dilation{1, 1, 0, 1};
    CoordinateDiff data_padding_below{0, 0, 0, 0};
    CoordinateDiff data_padding_above{0, 0, 0, 0};
    PartialShape window_shape{PartialShape::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;
    OPENVINO_SUPPRESS_DEPRECATED_START
    ASSERT_THROW(
        {
            PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                                      data_shape,
                                                                                      data_dilation,
                                                                                      data_padding_below,
                                                                                      data_padding_above,
                                                                                      window_shape,
                                                                                      window_strides,
                                                                                      window_dilation,
                                                                                      is_window_all_in_padding_allowed);
        },
        NodeValidationFailure);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST(partial_shape, infer_windowed_reduction_rank_dynamic_rank_dynamic_zero_window_dilation) {
    auto node = std::make_shared<op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{PartialShape::dynamic()};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 0, 0};
    CoordinateDiff data_padding_above{0, 0, 0, 0};
    PartialShape window_shape{PartialShape::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 0, 1, 1};
    bool is_window_all_in_padding_allowed = true;
    OPENVINO_SUPPRESS_DEPRECATED_START
    ASSERT_THROW(
        {
            PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                                      data_shape,
                                                                                      data_dilation,
                                                                                      data_padding_below,
                                                                                      data_padding_above,
                                                                                      window_shape,
                                                                                      window_strides,
                                                                                      window_dilation,
                                                                                      is_window_all_in_padding_allowed);
        },
        NodeValidationFailure);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST(partial_shape, infer_windowed_reduction_rank_dynamic_rank_dynamic_zero_window_strides) {
    auto node = std::make_shared<op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{PartialShape::dynamic()};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 0, 0};
    CoordinateDiff data_padding_above{0, 0, 0, 0};
    PartialShape window_shape{PartialShape::dynamic()};
    Strides window_strides{1, 1, 1, 0};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;
    OPENVINO_SUPPRESS_DEPRECATED_START
    ASSERT_THROW(
        {
            PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                                      data_shape,
                                                                                      data_dilation,
                                                                                      data_padding_below,
                                                                                      data_padding_above,
                                                                                      window_shape,
                                                                                      window_strides,
                                                                                      window_dilation,
                                                                                      is_window_all_in_padding_allowed);
        },
        NodeValidationFailure);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST(partial_shape, infer_windowed_reduction_rank_static_dynamic_rank_dynamic_ok) {
    auto node = std::make_shared<op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 0, 0};
    CoordinateDiff data_padding_above{0, 0, 0, 0};
    PartialShape window_shape{PartialShape::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;

    OPENVINO_SUPPRESS_DEPRECATED_START
    PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                              data_shape,
                                                                              data_dilation,
                                                                              data_padding_below,
                                                                              data_padding_above,
                                                                              window_shape,
                                                                              window_strides,
                                                                              window_dilation,
                                                                              is_window_all_in_padding_allowed);
    OPENVINO_SUPPRESS_DEPRECATED_END

    ASSERT_TRUE(result_shape.same_scheme(PartialShape::dynamic(4)));
}

TEST(partial_shape, infer_windowed_reduction_rank_static_dynamic_rank_dynamic_zero_data_post_padding) {
    auto node = std::make_shared<op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, -1, 0, 0};
    CoordinateDiff data_padding_above{0, -1, 0, 0};
    PartialShape window_shape{PartialShape::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;
    OPENVINO_SUPPRESS_DEPRECATED_START
    ASSERT_THROW(
        {
            PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                                      data_shape,
                                                                                      data_dilation,
                                                                                      data_padding_below,
                                                                                      data_padding_above,
                                                                                      window_shape,
                                                                                      window_strides,
                                                                                      window_dilation,
                                                                                      is_window_all_in_padding_allowed);
        },
        NodeValidationFailure);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST(partial_shape, infer_windowed_reduction_rank_static_dynamic_rank_dynamic_neg_padding_ok) {
    auto node = std::make_shared<op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{Dimension::dynamic(), 4, 3, Dimension::dynamic()};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, -1, 0, 0};
    CoordinateDiff data_padding_above{0, -2, 0, 0};
    PartialShape window_shape{PartialShape::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;
    OPENVINO_SUPPRESS_DEPRECATED_START
    PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                              data_shape,
                                                                              data_dilation,
                                                                              data_padding_below,
                                                                              data_padding_above,
                                                                              window_shape,
                                                                              window_strides,
                                                                              window_dilation,
                                                                              is_window_all_in_padding_allowed);
    OPENVINO_SUPPRESS_DEPRECATED_END

    ASSERT_TRUE(result_shape.same_scheme(PartialShape::dynamic(4)));
}

TEST(partial_shape, infer_windowed_reduction_rank_dynamic_rank_static_dynamic_ok) {
    auto node = std::make_shared<op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{PartialShape::dynamic()};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 0, 0};
    CoordinateDiff data_padding_above{0, 0, 0, 0};
    PartialShape window_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;

    OPENVINO_SUPPRESS_DEPRECATED_START
    PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                              data_shape,
                                                                              data_dilation,
                                                                              data_padding_below,
                                                                              data_padding_above,
                                                                              window_shape,
                                                                              window_strides,
                                                                              window_dilation,
                                                                              is_window_all_in_padding_allowed);
    OPENVINO_SUPPRESS_DEPRECATED_END

    ASSERT_TRUE(result_shape.same_scheme(PartialShape::dynamic(4)));
}

TEST(partial_shape, infer_windowed_reduction_rank_dynamic_rank_static_dynamic_window_dim_zero) {
    auto node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{PartialShape::dynamic()};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 0, 0};
    CoordinateDiff data_padding_above{0, 0, 0, 0};
    PartialShape window_shape{Dimension::dynamic(), 2, 0, Dimension::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;

    OPENVINO_SUPPRESS_DEPRECATED_START
    ASSERT_THROW(
        {
            PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                                      data_shape,
                                                                                      data_dilation,
                                                                                      data_padding_below,
                                                                                      data_padding_above,
                                                                                      window_shape,
                                                                                      window_strides,
                                                                                      window_dilation,
                                                                                      is_window_all_in_padding_allowed);
        },
        NodeValidationFailure);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST(partial_shape, infer_windowed_reduction_rank_dynamic_rank_static_dynamic_window_dilated_dim_zero) {
    auto node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{PartialShape::dynamic()};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 0, 0};
    CoordinateDiff data_padding_above{0, 0, 0, 0};
    PartialShape window_shape{Dimension::dynamic(), 2, 0, Dimension::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 3, 1};
    bool is_window_all_in_padding_allowed = true;

    OPENVINO_SUPPRESS_DEPRECATED_START
    ASSERT_THROW(
        {
            PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                                      data_shape,
                                                                                      data_dilation,
                                                                                      data_padding_below,
                                                                                      data_padding_above,
                                                                                      window_shape,
                                                                                      window_strides,
                                                                                      window_dilation,
                                                                                      is_window_all_in_padding_allowed);
        },
        NodeValidationFailure);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST(partial_shape, infer_windowed_reduction_rank_dynamic_rank_static_dynamic_window_all_in_padding_ok) {
    auto node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{PartialShape::dynamic()};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 3, 0};
    CoordinateDiff data_padding_above{0, 0, 0, 0};
    PartialShape window_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;

    OPENVINO_SUPPRESS_DEPRECATED_START
    PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                              data_shape,
                                                                              data_dilation,
                                                                              data_padding_below,
                                                                              data_padding_above,
                                                                              window_shape,
                                                                              window_strides,
                                                                              window_dilation,
                                                                              is_window_all_in_padding_allowed);
    OPENVINO_SUPPRESS_DEPRECATED_END

    ASSERT_TRUE(result_shape.same_scheme(PartialShape::dynamic(4)));
}

TEST(partial_shape, infer_windowed_reduction_rank_dynamic_rank_static_dynamic_window_all_in_padding_not_ok) {
    auto node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{PartialShape::dynamic()};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 3, 0};
    CoordinateDiff data_padding_above{0, 0, 0, 0};
    PartialShape window_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = false;

    OPENVINO_SUPPRESS_DEPRECATED_START
    ASSERT_THROW(
        {
            PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                                      data_shape,
                                                                                      data_dilation,
                                                                                      data_padding_below,
                                                                                      data_padding_above,
                                                                                      window_shape,
                                                                                      window_strides,
                                                                                      window_dilation,
                                                                                      is_window_all_in_padding_allowed);
        },
        NodeValidationFailure);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST(partial_shape, infer_windowed_reduction_rank_dynamic_rank_static_dynamic_dilated_window_not_all_in_padding) {
    auto node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{PartialShape::dynamic()};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 3, 0};
    CoordinateDiff data_padding_above{0, 0, 0, 0};
    PartialShape window_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 2, 1};
    bool is_window_all_in_padding_allowed = false;

    OPENVINO_SUPPRESS_DEPRECATED_START
    PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                              data_shape,
                                                                              data_dilation,
                                                                              data_padding_below,
                                                                              data_padding_above,
                                                                              window_shape,
                                                                              window_strides,
                                                                              window_dilation,
                                                                              is_window_all_in_padding_allowed);
    OPENVINO_SUPPRESS_DEPRECATED_END

    ASSERT_TRUE(result_shape.same_scheme(PartialShape::dynamic(4)));
}

TEST(partial_shape, infer_windowed_reduction_rank_static_dynamic_rank_static_dynamic_ok) {
    auto node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{Dimension::dynamic(), Dimension::dynamic(), 6, 4};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 0, 0};
    CoordinateDiff data_padding_above{0, 0, 0, 0};
    PartialShape window_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;

    OPENVINO_SUPPRESS_DEPRECATED_START
    PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                              data_shape,
                                                                              data_dilation,
                                                                              data_padding_below,
                                                                              data_padding_above,
                                                                              window_shape,
                                                                              window_strides,
                                                                              window_dilation,
                                                                              is_window_all_in_padding_allowed);
    OPENVINO_SUPPRESS_DEPRECATED_END

    ASSERT_TRUE(
        result_shape.same_scheme(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 4, Dimension::dynamic()}));
}

TEST(partial_shape, infer_windowed_reduction_rank_static_dynamic_rank_static_dynamic_with_padding_ok) {
    auto node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{Dimension::dynamic(), Dimension::dynamic(), 6, 4};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 2, 0};
    CoordinateDiff data_padding_above{0, 0, -1, 0};
    PartialShape window_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;

    OPENVINO_SUPPRESS_DEPRECATED_START
    PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                              data_shape,
                                                                              data_dilation,
                                                                              data_padding_below,
                                                                              data_padding_above,
                                                                              window_shape,
                                                                              window_strides,
                                                                              window_dilation,
                                                                              is_window_all_in_padding_allowed);
    OPENVINO_SUPPRESS_DEPRECATED_END

    ASSERT_TRUE(
        result_shape.same_scheme(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 5, Dimension::dynamic()}));
}

TEST(partial_shape, infer_windowed_reduction_rank_static_dynamic_rank_static_dynamic_with_padding_and_stride_ok) {
    auto node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{Dimension::dynamic(), Dimension::dynamic(), 6, 4};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 2, 0};
    CoordinateDiff data_padding_above{0, 0, -1, 0};
    PartialShape window_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    Strides window_strides{1, 1, 2, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;

    OPENVINO_SUPPRESS_DEPRECATED_START
    PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                              data_shape,
                                                                              data_dilation,
                                                                              data_padding_below,
                                                                              data_padding_above,
                                                                              window_shape,
                                                                              window_strides,
                                                                              window_dilation,
                                                                              is_window_all_in_padding_allowed);
    OPENVINO_SUPPRESS_DEPRECATED_END

    ASSERT_TRUE(
        result_shape.same_scheme(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()}));
}

TEST(partial_shape, infer_windowed_reduction_rank_static_dynamic_rank_static_dynamic_window_too_big) {
    auto node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{Dimension::dynamic(), Dimension::dynamic(), 6, 4};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 0, 0};
    CoordinateDiff data_padding_above{0, 0, 0, 0};
    PartialShape window_shape{Dimension::dynamic(), 2, 7, Dimension::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;

    OPENVINO_SUPPRESS_DEPRECATED_START
    ASSERT_THROW(
        {
            PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                                      data_shape,
                                                                                      data_dilation,
                                                                                      data_padding_below,
                                                                                      data_padding_above,
                                                                                      window_shape,
                                                                                      window_strides,
                                                                                      window_dilation,
                                                                                      is_window_all_in_padding_allowed);
        },
        NodeValidationFailure);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST(partial_shape, infer_windowed_reduction_rank_static_dynamic_rank_static_dynamic_window_not_too_big_padding) {
    auto node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{Dimension::dynamic(), Dimension::dynamic(), 6, 4};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 5, 0};
    CoordinateDiff data_padding_above{0, 0, -3, 0};
    PartialShape window_shape{Dimension::dynamic(), 2, 7, Dimension::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 1, 1};
    bool is_window_all_in_padding_allowed = true;

    OPENVINO_SUPPRESS_DEPRECATED_START
    PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                              data_shape,
                                                                              data_dilation,
                                                                              data_padding_below,
                                                                              data_padding_above,
                                                                              window_shape,
                                                                              window_strides,
                                                                              window_dilation,
                                                                              is_window_all_in_padding_allowed);
    OPENVINO_SUPPRESS_DEPRECATED_END

    ASSERT_TRUE(
        result_shape.same_scheme(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 2, Dimension::dynamic()}));
}

TEST(partial_shape, infer_windowed_reduction_rank_static_dynamic_rank_static_dynamic_window_dilated_too_big) {
    auto node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    PartialShape data_shape{Dimension::dynamic(), Dimension::dynamic(), 6, 4};
    Strides data_dilation{1, 1, 1, 1};
    CoordinateDiff data_padding_below{0, 0, 5, 0};
    CoordinateDiff data_padding_above{0, 0, -3, 0};
    PartialShape window_shape{Dimension::dynamic(), 2, 7, Dimension::dynamic()};
    Strides window_strides{1, 1, 1, 1};
    Strides window_dilation{1, 1, 2, 1};
    bool is_window_all_in_padding_allowed = true;

    OPENVINO_SUPPRESS_DEPRECATED_START
    ASSERT_THROW(
        {
            PartialShape result_shape = ngraph::infer_windowed_reduction_output_shape(node.get(),
                                                                                      data_shape,
                                                                                      data_dilation,
                                                                                      data_padding_below,
                                                                                      data_padding_above,
                                                                                      window_shape,
                                                                                      window_strides,
                                                                                      window_dilation,
                                                                                      is_window_all_in_padding_allowed);
        },
        NodeValidationFailure);
    OPENVINO_SUPPRESS_DEPRECATED_END
}
