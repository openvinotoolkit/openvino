// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/symbol_optimization.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/common_optimizations/shared_ops_optimization.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/symbolic_transformations/utils.hpp"

using namespace ov;
using namespace std;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v8 = ov::op::v8;
TEST(TransformationTests, ApplySymbolEquivalence_Concat) {
    auto input_1 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto input_2 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto concat = make_shared<v0::Concat>(OutputVector{input_1, input_2}, -1);
    // shape inference notes that all the non-axis dimensions are equal to each other
    auto model = make_shared<Model>(OutputVector{concat}, ParameterVector{input_2, input_1});

    pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<pass::SymbolicPropagation>();
    manager.register_pass<pass::ApplySymbolEquivalence>();
    manager.run_passes(model);

    const auto& pshape_1 = input_1->get_output_partial_shape(0);
    const auto& pshape_2 = input_2->get_output_partial_shape(0);
    const auto& pshape_3 = concat->get_output_partial_shape(0);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(symbol::are_equal(pshape_1[i].get_symbol(), pshape_2[i].get_symbol()));
        EXPECT_TRUE(symbol::are_equal(pshape_2[i].get_symbol(), pshape_3[i].get_symbol()));
        EXPECT_TRUE(symbol::are_equal(pshape_1[i].get_symbol(), pshape_3[i].get_symbol()));
    }
    EXPECT_FALSE(symbol::are_equal(pshape_1[3].get_symbol(), pshape_2[3].get_symbol()));
    EXPECT_FALSE(symbol::are_equal(pshape_2[3].get_symbol(), pshape_3[3].get_symbol()));
    EXPECT_FALSE(symbol::are_equal(pshape_1[3].get_symbol(), pshape_3[3].get_symbol()));
}

TEST_F(TransformationTestsF, ApplySymbolEquivalence_Concat_Values) {
    {
        auto input_1 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
        auto input_2 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
        auto concat = make_shared<v0::Concat>(OutputVector{input_1, input_2}, -1);

        auto shape = make_shared<v0::ShapeOf>(concat);
        auto gather = make_shared<v1::Gather>(shape,
                                              v0::Constant::create(element::i64, {1}, {-1}),
                                              v0::Constant::create(element::i64, {}, {0}));

        auto reshape = make_shared<v1::Reshape>(
            concat,
            make_shared<v0::Concat>(OutputVector{gather, v0::Constant::create(element::i64, {1}, {-1})}, 0),
            false);

        model = make_shared<Model>(OutputVector{reshape}, ParameterVector{input_2, input_1});

        manager.set_per_pass_validation(false);
        manager.register_pass<pass::SymbolicPropagation>();
        manager.register_pass<pass::ApplySymbolEquivalence>();
        manager.register_pass<pass::OptimizeSymbolsUsedAsValues>();
    }
    {
        auto input_1 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
        auto input_2 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
        auto concat = make_shared<v0::Concat>(OutputVector{input_1, input_2}, -1);

        auto shape_1 = make_shared<v3::ShapeOf>(input_1);
        auto gather_1 = make_shared<v8::Gather>(shape_1,
                                                v0::Constant::create(element::i64, {1}, {3}),
                                                v0::Constant::create(element::i64, {}, {0}));

        auto shape_2 = make_shared<v3::ShapeOf>(input_2);
        auto gather_2 = make_shared<v8::Gather>(shape_2,
                                                v0::Constant::create(element::i64, {1}, {3}),
                                                v0::Constant::create(element::i64, {}, {0}));

        auto sum = make_shared<v1::Add>(gather_1, gather_2);

        auto reshape = make_shared<v1::Reshape>(
            concat,
            make_shared<v0::Concat>(OutputVector{sum, v0::Constant::create(element::i64, {1}, {-1})}, 0),
            false);
        model_ref = make_shared<Model>(OutputVector{reshape}, ParameterVector{input_2, input_1});
    }
}

Output<Node> get_dim_by_idx(const Output<Node>& source, const int64_t& idx, element::Type type = element::i64) {
    auto shape = make_shared<v3::ShapeOf>(source, type);
    auto gather = make_shared<v1::Gather>(shape,
                                          v0::Constant::create(element::i64, {}, {idx}),
                                          v0::Constant::create(element::i64, {}, {0}));
    return gather->output(0);
}

Output<Node> get_dim_by_idx(const Output<Node>& source,
                            initializer_list<int64_t> idx,
                            element::Type type = element::i64) {
    auto shape = make_shared<v3::ShapeOf>(source, type);
    auto gather = make_shared<v8::Gather>(shape,
                                          v0::Constant::create(element::i64, {idx.size()}, idx),
                                          v0::Constant::create(element::i64, {}, {0}));
    return gather->output(0);
}

TEST_F(TransformationTestsF, ValueOptimizationSingleValue) {
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));

        auto dim_0 = get_dim_by_idx(input, {-1}, element::i64);
        auto dim_1 = get_dim_by_idx(input, {3}, element::i32);
        auto dim_2 = get_dim_by_idx(input, -1, element::i32);

        auto reshape_0 = make_shared<v1::Reshape>(
            input,
            make_shared<v0::Concat>(OutputVector{v0::Constant::create(element::i64, {1}, {-1}), dim_0}, 0),
            false);
        auto reshape_1 = make_shared<v1::Reshape>(
            input,
            make_shared<v0::Concat>(OutputVector{v0::Constant::create(element::i32, {1}, {0}), dim_1}, 0),
            false);
        auto range = make_shared<v4::Range>(v0::Constant::create(element::i32, {}, {0}),
                                            dim_2,
                                            v0::Constant::create(element::i32, {}, {1}),
                                            element::i32);

        model = make_shared<Model>(OutputVector{reshape_0, reshape_1, range}, ParameterVector{input});

        manager.set_per_pass_validation(false);
        manager.register_pass<pass::SymbolicPropagation>();
        manager.register_pass<pass::OptimizeSymbolsUsedAsValues>();
        manager.register_pass<pass::SharedOpOptimization>();
    }
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
        auto dim_1 = get_dim_by_idx(input, {3}, element::i32);
        auto dim_0 = std::make_shared<v0::Convert>(dim_1, element::i64);
        auto dim_2 = std::make_shared<v0::Squeeze>(dim_1);
        auto reshape_0 = make_shared<v1::Reshape>(
            input,
            make_shared<v0::Concat>(OutputVector{v0::Constant::create(element::i64, {1}, {-1}), dim_0}, 0),
            false);
        auto reshape_1 = make_shared<v1::Reshape>(
            input,
            make_shared<v0::Concat>(OutputVector{v0::Constant::create(element::i32, {1}, {0}), dim_1}, 0),
            false);
        auto range = make_shared<v4::Range>(v0::Constant::create(element::i32, {}, {0}),
                                            dim_2,
                                            v0::Constant::create(element::i32, {}, {1}),
                                            element::i32);

        model_ref = make_shared<Model>(OutputVector{reshape_0, reshape_1, range}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ValueOptimizationDoubleValue) {
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));

        auto dim_0 = get_dim_by_idx(input, {-1, -2}, element::i64);
        auto dim_1 = get_dim_by_idx(input, {3, 2}, element::i32);

        auto reshape_0 = make_shared<v1::Reshape>(
            input,
            make_shared<v0::Concat>(OutputVector{v0::Constant::create(element::i64, {1}, {-1}), dim_0}, 0),
            false);
        auto reshape_1 = make_shared<v1::Reshape>(
            input,
            make_shared<v0::Concat>(OutputVector{v0::Constant::create(element::i32, {1}, {0}), dim_1}, 0),
            false);

        model = make_shared<Model>(OutputVector{reshape_0, reshape_1}, ParameterVector{input});

        manager.set_per_pass_validation(false);
        manager.register_pass<pass::SymbolicPropagation>();
        manager.register_pass<pass::OptimizeSymbolsUsedAsValues>();
        manager.register_pass<pass::SharedOpOptimization>();
    }
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
        auto dim_0 = get_dim_by_idx(input, {3, 2}, element::i32);
        auto dim_1 = std::make_shared<v0::Convert>(dim_0, element::i64);

        auto reshape_0 = make_shared<v1::Reshape>(
            input,
            make_shared<v0::Concat>(OutputVector{v0::Constant::create(element::i64, {1}, {-1}), dim_1}, 0),
            false);
        auto reshape_1 = make_shared<v1::Reshape>(
            input,
            make_shared<v0::Concat>(OutputVector{v0::Constant::create(element::i32, {1}, {0}), dim_0}, 0),
            false);

        model_ref = make_shared<Model>(OutputVector{reshape_0, reshape_1}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ValueOptimizationSymbolAndValue) {
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape({-1, -1, 4, -1}));

        auto dim_0 = get_dim_by_idx(input, {-1, -2}, element::i64);
        auto dim_1 = get_dim_by_idx(input, {3, 2}, element::i32);

        auto reshape_0 = make_shared<v1::Reshape>(
            input,
            make_shared<v0::Concat>(OutputVector{v0::Constant::create(element::i64, {1}, {-1}), dim_0}, 0),
            false);
        auto reshape_1 = make_shared<v1::Reshape>(
            input,
            make_shared<v0::Concat>(OutputVector{v0::Constant::create(element::i32, {1}, {-1}), dim_1}, 0),
            false);

        model = make_shared<Model>(OutputVector{reshape_0, reshape_1}, ParameterVector{input});

        manager.set_per_pass_validation(false);
        manager.register_pass<pass::SymbolicPropagation>();
        manager.register_pass<pass::OptimizeSymbolsUsedAsValues>();
        manager.register_pass<pass::SharedOpOptimization>();
    }
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape({-1, -1, 4, -1}));
        auto dim_0 = make_shared<v0::Concat>(
            OutputVector{v0::Constant::create(element::i32, {1}, {-1}), get_dim_by_idx(input, {3, 2}, element::i32)},
            0);
        auto dim_1 = std::make_shared<v0::Convert>(dim_0, element::i64);

        auto reshape_0 = make_shared<v1::Reshape>(input, dim_1, false);
        auto reshape_1 = make_shared<v1::Reshape>(input, dim_0, false);

        model_ref = make_shared<Model>(OutputVector{reshape_0, reshape_1}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ValueOptimizationKeepsSourceAcrossSliceStep2) {
    // The Reshape pattern reads H and W from the output of a step-2 Slice. Those values are NOT equal to the
    // dimensions of the Slice input, so OptimizeSymbolsUsedAsValues must not re-source them to ShapeOf(input); the
    // pattern [B, H/2, W/2, 8] is the shape of the Slice output itself, hence the expected ShapeOf(slice).
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape{-1, Dimension(1, -1), Dimension(1, -1), 8});
        auto batch = get_dim_by_idx(input, {0});  // ShapeOf(input) is the earliest shape source of every input dim

        auto start = v0::Constant::create(element::i64, {2}, {0, 0});
        auto stop = v0::Constant::create(element::i64, {2}, {INT64_MAX, INT64_MAX});
        auto step = v0::Constant::create(element::i64, {2}, {2, 2});
        auto axes = v0::Constant::create(element::i64, {2}, {1, 2});
        auto slice = make_shared<v8::Slice>(input, start, stop, step, axes);

        auto h = get_dim_by_idx(slice, {1});
        auto w = get_dim_by_idx(slice, {2});
        auto pattern =
            make_shared<v0::Concat>(OutputVector{batch, h, w, v0::Constant::create(element::i64, {1}, {8})->output(0)}, 0);
        auto reshape = make_shared<v1::Reshape>(slice, pattern, false);

        model = make_shared<Model>(OutputVector{reshape}, ParameterVector{input});

        manager.set_per_pass_validation(false);
        manager.register_pass<pass::SymbolicPropagation>();
        manager.register_pass<pass::OptimizeSymbolsUsedAsValues>();
        manager.register_pass<pass::SharedOpOptimization>();
    }
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape{-1, Dimension(1, -1), Dimension(1, -1), 8});
        auto start = v0::Constant::create(element::i64, {2}, {0, 0});
        auto stop = v0::Constant::create(element::i64, {2}, {INT64_MAX, INT64_MAX});
        auto step = v0::Constant::create(element::i64, {2}, {2, 2});
        auto axes = v0::Constant::create(element::i64, {2}, {1, 2});
        auto slice = make_shared<v8::Slice>(input, start, stop, step, axes);
        auto pattern = make_shared<v3::ShapeOf>(slice, element::i64);
        auto reshape = make_shared<v1::Reshape>(slice, pattern, false);

        model_ref = make_shared<Model>(OutputVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ValueOptimizationKeepsSourceAcrossStridedSliceStride2) {
    // Same as above with the StridedSlice form (masked full range, stride 2 on H and W).
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape{-1, Dimension(1, -1), Dimension(1, -1), 8});
        auto batch = get_dim_by_idx(input, {0});

        auto begin = v0::Constant::create(element::i64, {4}, {0, 0, 0, 0});
        auto end = v0::Constant::create(element::i64, {4}, {0, 0, 0, 0});
        auto stride = v0::Constant::create(element::i64, {4}, {1, 2, 2, 1});
        auto mask = std::vector<int64_t>(4, 1);
        auto slice = make_shared<v1::StridedSlice>(input, begin, end, stride, mask, mask);

        auto h = get_dim_by_idx(slice, {1});
        auto w = get_dim_by_idx(slice, {2});
        auto pattern =
            make_shared<v0::Concat>(OutputVector{batch, h, w, v0::Constant::create(element::i64, {1}, {8})->output(0)}, 0);
        auto reshape = make_shared<v1::Reshape>(slice, pattern, false);

        model = make_shared<Model>(OutputVector{reshape}, ParameterVector{input});

        manager.set_per_pass_validation(false);
        manager.register_pass<pass::SymbolicPropagation>();
        manager.register_pass<pass::OptimizeSymbolsUsedAsValues>();
        manager.register_pass<pass::SharedOpOptimization>();
    }
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape{-1, Dimension(1, -1), Dimension(1, -1), 8});
        auto begin = v0::Constant::create(element::i64, {4}, {0, 0, 0, 0});
        auto end = v0::Constant::create(element::i64, {4}, {0, 0, 0, 0});
        auto stride = v0::Constant::create(element::i64, {4}, {1, 2, 2, 1});
        auto mask = std::vector<int64_t>(4, 1);
        auto slice = make_shared<v1::StridedSlice>(input, begin, end, stride, mask, mask);
        auto pattern = make_shared<v3::ShapeOf>(slice, element::i64);
        auto reshape = make_shared<v1::Reshape>(slice, pattern, false);

        model_ref = make_shared<Model>(OutputVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST(TransformationTests, ValueOptimizationReusesSourceAcrossIdentitySlice) {
    // A step-1 full-range slice preserves H and W, so the values read from ShapeOf(slice) are legitimately re-sourced
    // to the earliest shape source, ShapeOf(input).
    auto input = make_shared<v0::Parameter>(element::f32, PartialShape{-1, Dimension(1, -1), Dimension(1, -1), 8});
    auto batch = get_dim_by_idx(input, {0});

    auto start = v0::Constant::create(element::i64, {2}, {0, 0});
    auto stop = v0::Constant::create(element::i64, {2}, {INT64_MAX, INT64_MAX});
    auto step = v0::Constant::create(element::i64, {2}, {1, 1});
    auto axes = v0::Constant::create(element::i64, {2}, {1, 2});
    auto slice = make_shared<v8::Slice>(input, start, stop, step, axes);

    auto h = get_dim_by_idx(slice, {1});
    auto w = get_dim_by_idx(slice, {2});
    auto pattern =
        make_shared<v0::Concat>(OutputVector{batch, h, w, v0::Constant::create(element::i64, {1}, {8})->output(0)}, 0);
    auto reshape = make_shared<v1::Reshape>(slice, pattern, false);
    auto model = make_shared<Model>(OutputVector{reshape}, ParameterVector{input});

    pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<pass::SymbolicPropagation>();
    manager.register_pass<pass::OptimizeSymbolsUsedAsValues>();
    manager.run_passes(model);

    for (size_t i = 1; i <= 2; ++i) {
        const auto gather = pattern->get_input_node_shared_ptr(i);
        ASSERT_TRUE(ov::is_type<ov::op::util::GatherBase>(gather)) << "pattern input " << i;
        const auto shape_of = gather->get_input_node_shared_ptr(0);
        ASSERT_TRUE(ov::is_type<ov::op::util::ShapeOfBase>(shape_of)) << "pattern input " << i;
        EXPECT_EQ(shape_of->get_input_node_shared_ptr(0), input) << "pattern input " << i;
    }
}

TEST(TransformationTests, SliceStep2ConcatDoesNotEquateInputDimSymbols) {
    // Concat merges the symbols of its non-axis dimensions, so a step-2 slice output concatenated with another tensor
    // must not make the slice input dimension equal to that tensor's dimension.
    auto x = make_shared<v0::Parameter>(element::f32, PartialShape{-1, Dimension(1, -1), 4});
    auto z = make_shared<v0::Parameter>(element::f32, PartialShape{-1, Dimension(1, -1), 6});

    auto start = v0::Constant::create(element::i64, {1}, {0});
    auto stop = v0::Constant::create(element::i64, {1}, {INT64_MAX});
    auto step = v0::Constant::create(element::i64, {1}, {2});
    auto axes = v0::Constant::create(element::i64, {1}, {1});
    auto slice = make_shared<v8::Slice>(x, start, stop, step, axes);
    auto concat = make_shared<v0::Concat>(OutputVector{slice, z}, 2);
    auto model = make_shared<Model>(OutputVector{concat}, ParameterVector{x, z});

    pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<pass::SymbolicPropagation>();
    manager.run_passes(model);

    const auto x_dim = x->get_output_partial_shape(0)[1].get_symbol();
    const auto z_dim = z->get_output_partial_shape(0)[1].get_symbol();
    const auto concat_dim = concat->get_output_partial_shape(0)[1].get_symbol();
    ASSERT_NE(x_dim, nullptr);
    ASSERT_NE(z_dim, nullptr);
    EXPECT_FALSE(symbol::are_equal(x_dim, z_dim));
    EXPECT_FALSE(symbol::are_equal(x_dim, concat_dim));
    EXPECT_TRUE(symbol::are_equal(z_dim, concat_dim));
}
