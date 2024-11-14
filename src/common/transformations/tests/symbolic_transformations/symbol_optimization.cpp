// Copyright (C) 2018-2024 Intel Corporation
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
#include "openvino/op/squeeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/common_optimizations/shared_ops_optimization.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/symbolic_transformations/utils.hpp"

using namespace ov;
using namespace ov::op;
using namespace std;

TEST(TransformationTests, ApplySymbolEquivalence_Concat) {
    auto input_1 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto input_2 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto concat = make_shared<v0::Concat>(OutputVector{input_1, input_2}, -1);
    // shape inference notes that all the non-axis dimensions are equal to each other
    auto model = make_shared<Model>(NodeVector{concat}, ParameterVector{input_2, input_1});

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

        model = make_shared<Model>(NodeVector{reshape}, ParameterVector{input_2, input_1});

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
        model_ref = make_shared<Model>(NodeVector{reshape}, ParameterVector{input_2, input_1});
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

        model = make_shared<Model>(NodeVector{reshape_0, reshape_1, range}, ParameterVector{input});

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

        model_ref = make_shared<Model>(NodeVector{reshape_0, reshape_1, range}, ParameterVector{input});
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

        model = make_shared<Model>(NodeVector{reshape_0, reshape_1}, ParameterVector{input});

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

        model_ref = make_shared<Model>(NodeVector{reshape_0, reshape_1}, ParameterVector{input});
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

        model = make_shared<Model>(NodeVector{reshape_0, reshape_1}, ParameterVector{input});

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

        model_ref = make_shared<Model>(NodeVector{reshape_0, reshape_1}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
