// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/common_optimizations/algebraic_simplification.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/common_optimizations/transpose_sinking.hpp>
#include <transformations/common_optimizations/transpose_to_reshape.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

using InputShape = ngraph::PartialShape;
using TransposeOrder = std::vector<int64_t>;

struct ReferenceParams {
    bool no_changes = false;
    bool is_empty = false;
    std::vector<int64_t> reshape_value;

    ReferenceParams() = default;

    explicit ReferenceParams(bool no_changes, bool is_empty) : no_changes(no_changes), is_empty(is_empty) {}

    explicit ReferenceParams(const std::vector<int64_t> & reshape_value): reshape_value(reshape_value) {}
};

class TransposeToReshapeTests: public CommonTestUtils::TestsCommon,
                               public testing::WithParamInterface<std::tuple<InputShape, TransposeOrder, ReferenceParams> > {
public:
    std::shared_ptr<ngraph::Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& transpose_order = std::get<1>(GetParam());
        const auto& reference_params = std::get<2>(GetParam());

        f = get_initial_function(input_shape, transpose_order);
        f_ref = get_reference_function(input_shape, transpose_order, reference_params);
    }

private:
    std::shared_ptr<ngraph::Function> get_initial_function(const ngraph::PartialShape & input_shape,
                                                           const std::vector<int64_t> & transpose_order) {
        auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, input_shape);
        auto order_const = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{transpose_order.size()}, transpose_order);
        auto transpose = std::make_shared<ngraph::opset3::Transpose>(data, order_const);

        // WA to test cases with transpose elimination
        auto relu = std::make_shared<ngraph::opset3::Relu>(transpose);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{relu}, ngraph::ParameterVector{data});
    }

    std::shared_ptr<ngraph::Function> get_reference_function(const ngraph::PartialShape & input_shape,
                                                             const std::vector<int64_t> & transpose_order,
                                                             const ReferenceParams & params) {
        if (params.no_changes) {
            return get_initial_function(input_shape, transpose_order);
        }

        auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, input_shape);

        ngraph::Output<ngraph::Node> reshape_dims, last(data);
        if (!params.reshape_value.empty()) {
            reshape_dims = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{params.reshape_value.size()}, params.reshape_value);
        } else {
            auto shape_of = std::make_shared<ngraph::opset3::ShapeOf>(data);
            reshape_dims = std::make_shared<ngraph::opset3::Gather>(shape_of,
                    ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{transpose_order.size()}, transpose_order),
                    ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
        }

        if (!params.is_empty) {
            last = std::make_shared<ngraph::opset3::Reshape>(last, reshape_dims, true);
        }

        last = std::make_shared<ngraph::opset3::Relu>(last);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{last.get_node_shared_ptr()}, ngraph::ParameterVector{data});
    }
};

TEST_P(TransposeToReshapeTests, CompareFunctions) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::TransposeToReshape>();
    manager.run_passes(f);
    f->validate_nodes_and_infer_types();
    ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

#define SAME_FUNCTION    ReferenceParams(true, false)
#define EMPTY_FUNCTION   ReferenceParams(false, true)
#define SHAPE_OF_GATHER  ReferenceParams()

INSTANTIATE_TEST_SUITE_P(KeepTranspose, TransposeToReshapeTests,
        testing::Values(std::make_tuple(InputShape{1, 3, 64, 64},  TransposeOrder{0, 1, 3, 2}, SAME_FUNCTION),
                        std::make_tuple(InputShape{1, 3, 1, 64},   TransposeOrder{2, 0, 3, 1}, SAME_FUNCTION),
                        std::make_tuple(InputShape{1, 3, 1, 3},    TransposeOrder{3, 0, 2, 1}, SAME_FUNCTION),
                        std::make_tuple(InputShape{DYN, 2, 64, 1}, TransposeOrder{1, 0, 3, 2}, SAME_FUNCTION),
                        std::make_tuple(InputShape{DYN, 3},        TransposeOrder{1, 0},       SAME_FUNCTION),
                        std::make_tuple(InputShape{DYN, DYN, 1},   TransposeOrder{2, 1, 0},    SAME_FUNCTION),
                        std::make_tuple(InputShape{DYN, DYN},      TransposeOrder{1, 0},       SAME_FUNCTION)));

INSTANTIATE_TEST_SUITE_P(EliminateTranspose, TransposeToReshapeTests,
        testing::Values(std::make_tuple(InputShape{1, 3, 64, 64}, TransposeOrder{0, 1, 2, 3}, EMPTY_FUNCTION),
                        std::make_tuple(InputShape{1, 1, 1},      TransposeOrder{2, 0, 1},    EMPTY_FUNCTION),
                        std::make_tuple(InputShape{DYN, DYN},     TransposeOrder{0, 1},       EMPTY_FUNCTION)));

INSTANTIATE_TEST_SUITE_P(ReshapeWithConstant, TransposeToReshapeTests,
        testing::Values(std::make_tuple(InputShape{1, 3, 64, 1},   TransposeOrder{0, 1, 3, 2}, ReferenceParams({1, 3, 1, 64})),
                        std::make_tuple(InputShape{1, 3, 1, 64},   TransposeOrder{1, 0, 3, 2}, ReferenceParams({3, 1, 64, 1})),
                        std::make_tuple(InputShape{DYN, DYN, 1},   TransposeOrder{0, 2, 1},    ReferenceParams({0, 1, -1})),
                        std::make_tuple(InputShape{1, 1, DYN},     TransposeOrder{2, 1, 0},    ReferenceParams({-1, 0, 1})),
                        std::make_tuple(InputShape{DYN, 1, 64, 1}, TransposeOrder{1, 0, 3, 2}, ReferenceParams({1, -1, 1, 64}))));

INSTANTIATE_TEST_SUITE_P(ReshapeWithGather, TransposeToReshapeTests,
        testing::Values(std::make_tuple(InputShape{DYN, 1, DYN, 1},   TransposeOrder{1, 0, 3, 2}, SHAPE_OF_GATHER),
                        std::make_tuple(InputShape{1, DYN, DYN, DYN}, TransposeOrder{1, 2, 3, 0}, SHAPE_OF_GATHER)));

#undef SAME_FUNCTION
#undef EMPTY_FUNCTION
#undef SHAPE_OF_GATHER
