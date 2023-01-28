// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/manager.hpp>
#include <openvino/opsets/opset10.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/convert_batch_to_space.hpp>
#include <transformations/op_conversions/convert_space_to_batch.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/runtime/core.hpp"

using namespace std;
using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, BatchToSpaceDecompositionByElements) {
    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{100, 7, 13, 3});
        auto block_shape =
            std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 10, 5, 1});
        auto crops_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto crops_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto batch_to_space = std::make_shared<ngraph::opset3::BatchToSpace>(data, block_shape, crops_begin, crops_end);

        function =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{batch_to_space}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::ConvertBatchToSpace>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{100, 7, 13, 3});

        auto dispresed_shape_1 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {10, 10, 7, 13, 3});
        auto axis_order_1 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 2, 0, 3, 4});
        auto squeezed_order_1 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {10, 70, 13, 3});

        auto reshape_before_1 = std::make_shared<ngraph::opset3::Reshape>(data, dispresed_shape_1, false);
        auto permute_1 = std::make_shared<ngraph::opset3::Transpose>(reshape_before_1, axis_order_1);
        auto reshape_after_1 = std::make_shared<ngraph::opset3::Reshape>(permute_1, squeezed_order_1, false);

        auto dispresed_shape_2 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {5, 2, 70, 13, 3});
        auto axis_order_2 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 2, 3, 0, 4});
        auto squeezed_order_2 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 70, 65, 3});

        auto reshape_before_2 = std::make_shared<ngraph::opset3::Reshape>(reshape_after_1, dispresed_shape_2, false);
        auto permute_2 = std::make_shared<ngraph::opset3::Transpose>(reshape_before_2, axis_order_2);
        auto reshape_after_2 = std::make_shared<ngraph::opset3::Reshape>(permute_2, squeezed_order_2, false);

        auto dispresed_shape_3 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 2, 70, 65, 3});
        auto axis_order_3 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 2, 3, 4, 0});
        auto squeezed_order_3 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 70, 65, 3});

        auto reshape_before_3 = std::make_shared<ngraph::opset3::Reshape>(reshape_after_2, dispresed_shape_3, false);
        auto permute_3 = std::make_shared<ngraph::opset3::Transpose>(reshape_before_3, axis_order_3);
        auto reshape_after_3 = std::make_shared<ngraph::opset3::Reshape>(permute_3, squeezed_order_3, false);

        auto begin = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 0});
        auto end = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 67, 65, 3});
        std::vector<int64_t> begin_mask(4, 0);
        std::vector<int64_t> end_mask(4, 0);
        auto ss = std::make_shared<opset3::StridedSlice>(reshape_after_3, begin, end, begin_mask, end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SpaceToBatchDecompositionByElements) {
    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{2, 64, 64, 3});
        auto block_shape =
            std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 10, 5, 1});
        auto pads_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto pads_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto batch_to_space = std::make_shared<ngraph::opset3::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

        function =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{batch_to_space}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::ConvertSpaceToBatch>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{2, 64, 64, 3});
        auto pads_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto pads_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto pads = std::make_shared<opset3::Pad>(data, pads_begin, pads_end, ngraph::op::PadMode::CONSTANT);

        auto dispresed_shape_1 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {2, 70, 65, 3, 1});
        auto axis_order_1 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {4, 0, 1, 2, 3});
        auto squeezed_order_1 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 70, 65, 3});

        auto reshape_before_1 = std::make_shared<ngraph::opset3::Reshape>(pads, dispresed_shape_1, false);
        auto permute_1 = std::make_shared<ngraph::opset3::Transpose>(reshape_before_1, axis_order_1);
        auto reshape_after_1 = std::make_shared<ngraph::opset3::Reshape>(permute_1, squeezed_order_1, false);

        auto dispresed_shape_2 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {2, 70, 13, 5, 3});
        auto axis_order_2 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {3, 0, 1, 2, 4});
        auto squeezed_order_2 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {10, 70, 13, 3});

        auto reshape_before_2 = std::make_shared<ngraph::opset3::Reshape>(reshape_after_1, dispresed_shape_2, false);
        auto permute_2 = std::make_shared<ngraph::opset3::Transpose>(reshape_before_2, axis_order_2);
        auto reshape_after_2 = std::make_shared<ngraph::opset3::Reshape>(permute_2, squeezed_order_2, false);

        auto dispresed_shape_3 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {10, 7, 10, 13, 3});
        auto axis_order_3 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {2, 0, 1, 3, 4});
        auto squeezed_order_3 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {100, 7, 13, 3});

        auto reshape_before_3 = std::make_shared<ngraph::opset3::Reshape>(reshape_after_2, dispresed_shape_3, false);
        auto permute_3 = std::make_shared<ngraph::opset3::Transpose>(reshape_before_3, axis_order_3);
        auto reshape_after_3 = std::make_shared<ngraph::opset3::Reshape>(permute_3, squeezed_order_3, false);

        auto dispresed_shape_4 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {100, 1, 7, 13, 3});
        auto axis_order_4 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 0, 2, 3, 4});
        auto squeezed_order_4 =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {100, 7, 13, 3});

        auto reshape_before_4 = std::make_shared<ngraph::opset3::Reshape>(reshape_after_3, dispresed_shape_4, false);
        auto permute_4 = std::make_shared<ngraph::opset3::Transpose>(reshape_before_4, axis_order_4);
        auto reshape_after_4 = std::make_shared<ngraph::opset3::Reshape>(permute_4, squeezed_order_4, false);

        function_ref =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after_4}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SpaceToBatchDecomposition) {
    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{2, 64, 64, 3});
        auto block_shape =
            std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 10, 5, 1});
        auto pads_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto pads_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto batch_to_space = std::make_shared<ngraph::opset3::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

        function =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{batch_to_space}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::ConvertSpaceToBatch>(false);
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{2, 64, 64, 3});
        auto pads_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto pads_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto pads = std::make_shared<opset3::Pad>(data, pads_begin, pads_end, ngraph::op::PadMode::CONSTANT);

        auto dispresed_shape =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{7}, {2, 7, 10, 13, 5, 3, 1});
        auto axis_order =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{7}, {2, 4, 6, 0, 1, 3, 5});
        auto squeezed_order = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {100, 7, 13, 3});

        auto reshape_before = std::make_shared<ngraph::opset3::Reshape>(pads, dispresed_shape, false);
        auto permute = std::make_shared<ngraph::opset3::Transpose>(reshape_before, axis_order);
        auto reshape_after = std::make_shared<ngraph::opset3::Reshape>(permute, squeezed_order, false);

        function_ref =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, BatchToSpaceDecomposition) {
    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{100, 7, 13, 3});
        auto block_shape =
            std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 10, 5, 1});
        auto crops_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto crops_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto batch_to_space = std::make_shared<ngraph::opset3::BatchToSpace>(data, block_shape, crops_begin, crops_end);

        function =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{batch_to_space}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::ConvertBatchToSpace>(false);
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{100, 7, 13, 3});

        auto dispresed_shape =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{7}, {10, 5, 1, 2, 7, 13, 3});
        auto axis_order =
            ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{7}, {3, 4, 0, 5, 1, 6, 2});
        auto squeezed_order = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 70, 65, 3});

        auto reshape_before = std::make_shared<ngraph::opset3::Reshape>(data, dispresed_shape, false);
        auto permute = std::make_shared<ngraph::opset3::Transpose>(reshape_before, axis_order);
        auto reshape_after = std::make_shared<ngraph::opset3::Reshape>(permute, squeezed_order, false);

        auto begin = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 0});
        auto end = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 67, 65, 3});
        std::vector<int64_t> begin_mask(4, 0);
        std::vector<int64_t> end_mask(4, 0);
        auto ss = std::make_shared<opset3::StridedSlice>(reshape_after, begin, end, begin_mask, end_mask);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{data});
    }
}

using BatchToSpaceParams = tuple<Shape,            // data_shape
                                 vector<int64_t>,  // block
                                 vector<int64_t>,  // crops_begin
                                 vector<int64_t>,  // crops_end
                                 Shape             // expected_output_shape
                                 >;

using BatchToSpaceDecomposeParams = tuple<BatchToSpaceParams,
                                          bool  // by_elements
                                          >;

class BatchToSpaceDecompositionWithParams : public testing::WithParamInterface<BatchToSpaceDecomposeParams>,
                                            public TransformationTests {};

TEST_P(BatchToSpaceDecompositionWithParams, DynamicInputs) {
    using namespace ov::opset10;
    using namespace ov::pass;

    const auto& params = GetParam();
    Shape data_shape, expected_output_shape;
    vector<int64_t> block, crops_begin, crops_end;
    bool by_elements;
    tie(data_shape, block, crops_begin, crops_end, expected_output_shape) = get<0>(params);
    by_elements = get<1>(params);

    const auto data = make_shared<Parameter>(element::f32, PartialShape::dynamic(data_shape.size()));
    data->get_default_output().get_tensor().add_names({"data"});
    const auto block_shape_p = make_shared<Constant>(element::i64, Shape{block.size()}, block);
    const auto crops_begin_p = make_shared<Constant>(element::i64, Shape{crops_begin.size()}, crops_begin);
    const auto crops_end_p = make_shared<Constant>(element::i64, Shape{crops_end.size()}, crops_end);
    const auto batch_to_space = make_shared<BatchToSpace>(data, block_shape_p, crops_begin_p, crops_end_p);
    const auto f = make_shared<Function>(NodeVector{batch_to_space}, ParameterVector{data});

    Manager m;
    m.register_pass<ConvertBatchToSpace>();
    ASSERT_NO_THROW(m.run_passes(f));
    ASSERT_EQ(count_ops_of_type<BatchToSpace>(f), 0);
    EXPECT_TRUE(f->get_result()->get_input_partial_shape(0).is_dynamic());

    f->reshape({{"data", PartialShape{data_shape}}});
    ASSERT_EQ(f->get_result()->get_input_shape(0), expected_output_shape);
}

static vector<BatchToSpaceParams> batch_to_space_params = {
    {{4, 3}, {1, 2}, {0, 0}, {0, 0}, {2, 6}},
    {{2, 6, 7}, {1, 2, 1}, {0, 0, 0}, {0, 0, 0}, {1, 12, 7}},
    {{6, 5, 7}, {1, 2, 3}, {0, 1, 2}, {0, 1, 2}, {1, 8, 17}},
    {{30, 4, 1, 1}, {1, 5, 3, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 20, 3, 2}},
    {{96, 3, 5, 7, 1}, {1, 4, 3, 2, 1}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {4, 12, 15, 14, 1}},
};

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         BatchToSpaceDecompositionWithParams,
                         ::testing::Combine(::testing::ValuesIn(batch_to_space_params), ::testing::ValuesIn({true, false})));
