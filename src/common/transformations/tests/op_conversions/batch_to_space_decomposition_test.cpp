// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/convert_batch_to_space.hpp"
#include "transformations/op_conversions/convert_space_to_batch.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, BatchToSpaceDecompositionByElements) {
    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{100, 7, 13, 3});
        auto block_shape =
            std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 10, 5, 1});
        auto crops_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto crops_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto batch_to_space = std::make_shared<opset3::BatchToSpace>(data, block_shape, crops_begin, crops_end);

        model = std::make_shared<ov::Model>(NodeVector{batch_to_space}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertBatchToSpace>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{100, 7, 13, 3});

        auto dispresed_shape_1 = opset3::Constant::create(element::i64, Shape{5}, {10, 10, 7, 13, 3});
        auto axis_order_1 = opset3::Constant::create(element::i64, Shape{5}, {1, 2, 0, 3, 4});
        auto squeezed_order_1 = opset3::Constant::create(element::i64, Shape{4}, {10, 70, 13, 3});

        auto reshape_before_1 = std::make_shared<opset3::Reshape>(data, dispresed_shape_1, false);
        auto permute_1 = std::make_shared<opset3::Transpose>(reshape_before_1, axis_order_1);
        auto reshape_after_1 = std::make_shared<opset3::Reshape>(permute_1, squeezed_order_1, false);

        auto dispresed_shape_2 = opset3::Constant::create(element::i64, Shape{5}, {5, 2, 70, 13, 3});
        auto axis_order_2 = opset3::Constant::create(element::i64, Shape{5}, {1, 2, 3, 0, 4});
        auto squeezed_order_2 = opset3::Constant::create(element::i64, Shape{4}, {2, 70, 65, 3});

        auto reshape_before_2 = std::make_shared<opset3::Reshape>(reshape_after_1, dispresed_shape_2, false);
        auto permute_2 = std::make_shared<opset3::Transpose>(reshape_before_2, axis_order_2);
        auto reshape_after_2 = std::make_shared<opset3::Reshape>(permute_2, squeezed_order_2, false);

        auto dispresed_shape_3 = opset3::Constant::create(element::i64, Shape{5}, {1, 2, 70, 65, 3});
        auto axis_order_3 = opset3::Constant::create(element::i64, Shape{5}, {1, 2, 3, 4, 0});
        auto squeezed_order_3 = opset3::Constant::create(element::i64, Shape{4}, {2, 70, 65, 3});

        auto reshape_before_3 = std::make_shared<opset3::Reshape>(reshape_after_2, dispresed_shape_3, false);
        auto permute_3 = std::make_shared<opset3::Transpose>(reshape_before_3, axis_order_3);
        auto reshape_after_3 = std::make_shared<opset3::Reshape>(permute_3, squeezed_order_3, false);

        auto begin = opset3::Constant::create(element::i64, Shape{4}, {0, 3, 1, 0});
        auto end = opset3::Constant::create(element::i64, Shape{4}, {2, 67, 65, 3});
        std::vector<int64_t> begin_mask(4, 0);
        std::vector<int64_t> end_mask(4, 0);
        auto ss = std::make_shared<opset3::StridedSlice>(reshape_after_3, begin, end, begin_mask, end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{ss}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SpaceToBatchDecompositionByElements) {
    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{2, 64, 64, 3});
        auto block_shape =
            std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 10, 5, 1});
        auto pads_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto pads_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto batch_to_space = std::make_shared<opset3::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

        model = std::make_shared<ov::Model>(NodeVector{batch_to_space}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertSpaceToBatch>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{2, 64, 64, 3});
        auto pads_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto pads_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto pads = std::make_shared<opset3::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);

        auto dispresed_shape_1 = opset3::Constant::create(element::i64, Shape{5}, {2, 70, 65, 3, 1});
        auto axis_order_1 = opset3::Constant::create(element::i64, Shape{5}, {4, 0, 1, 2, 3});
        auto squeezed_order_1 = opset3::Constant::create(element::i64, Shape{4}, {2, 70, 65, 3});

        auto reshape_before_1 = std::make_shared<opset3::Reshape>(pads, dispresed_shape_1, false);
        auto permute_1 = std::make_shared<opset3::Transpose>(reshape_before_1, axis_order_1);
        auto reshape_after_1 = std::make_shared<opset3::Reshape>(permute_1, squeezed_order_1, false);

        auto dispresed_shape_2 = opset3::Constant::create(element::i64, Shape{5}, {2, 70, 13, 5, 3});
        auto axis_order_2 = opset3::Constant::create(element::i64, Shape{5}, {3, 0, 1, 2, 4});
        auto squeezed_order_2 = opset3::Constant::create(element::i64, Shape{4}, {10, 70, 13, 3});

        auto reshape_before_2 = std::make_shared<opset3::Reshape>(reshape_after_1, dispresed_shape_2, false);
        auto permute_2 = std::make_shared<opset3::Transpose>(reshape_before_2, axis_order_2);
        auto reshape_after_2 = std::make_shared<opset3::Reshape>(permute_2, squeezed_order_2, false);

        auto dispresed_shape_3 = opset3::Constant::create(element::i64, Shape{5}, {10, 7, 10, 13, 3});
        auto axis_order_3 = opset3::Constant::create(element::i64, Shape{5}, {2, 0, 1, 3, 4});
        auto squeezed_order_3 = opset3::Constant::create(element::i64, Shape{4}, {100, 7, 13, 3});

        auto reshape_before_3 = std::make_shared<opset3::Reshape>(reshape_after_2, dispresed_shape_3, false);
        auto permute_3 = std::make_shared<opset3::Transpose>(reshape_before_3, axis_order_3);
        auto reshape_after_3 = std::make_shared<opset3::Reshape>(permute_3, squeezed_order_3, false);

        auto dispresed_shape_4 = opset3::Constant::create(element::i64, Shape{5}, {100, 1, 7, 13, 3});
        auto axis_order_4 = opset3::Constant::create(element::i64, Shape{5}, {1, 0, 2, 3, 4});
        auto squeezed_order_4 = opset3::Constant::create(element::i64, Shape{4}, {100, 7, 13, 3});

        auto reshape_before_4 = std::make_shared<opset3::Reshape>(reshape_after_3, dispresed_shape_4, false);
        auto permute_4 = std::make_shared<opset3::Transpose>(reshape_before_4, axis_order_4);
        auto reshape_after_4 = std::make_shared<opset3::Reshape>(permute_4, squeezed_order_4, false);

        model_ref = std::make_shared<ov::Model>(NodeVector{reshape_after_4}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SpaceToBatchDecomposition) {
    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{2, 64, 64, 3});
        auto block_shape =
            std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 10, 5, 1});
        auto pads_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto pads_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto batch_to_space = std::make_shared<opset3::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

        model = std::make_shared<ov::Model>(NodeVector{batch_to_space}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertSpaceToBatch>(false);
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{2, 64, 64, 3});
        auto pads_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto pads_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto pads = std::make_shared<opset3::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);

        auto dispresed_shape = opset3::Constant::create(element::i64, Shape{7}, {2, 7, 10, 13, 5, 3, 1});
        auto axis_order = opset3::Constant::create(element::i64, Shape{7}, {2, 4, 6, 0, 1, 3, 5});
        auto squeezed_order = opset3::Constant::create(element::i64, Shape{4}, {100, 7, 13, 3});

        auto reshape_before = std::make_shared<opset3::Reshape>(pads, dispresed_shape, false);
        auto permute = std::make_shared<opset3::Transpose>(reshape_before, axis_order);
        auto reshape_after = std::make_shared<opset3::Reshape>(permute, squeezed_order, false);

        model_ref = std::make_shared<ov::Model>(NodeVector{reshape_after}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, BatchToSpaceDecomposition) {
    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{100, 7, 13, 3});
        auto block_shape =
            std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 10, 5, 1});
        auto crops_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto crops_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto batch_to_space = std::make_shared<opset3::BatchToSpace>(data, block_shape, crops_begin, crops_end);

        model = std::make_shared<ov::Model>(NodeVector{batch_to_space}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertBatchToSpace>(false);
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{100, 7, 13, 3});

        auto dispresed_shape = opset3::Constant::create(element::i64, Shape{7}, {10, 5, 1, 2, 7, 13, 3});
        auto axis_order = opset3::Constant::create(element::i64, Shape{7}, {3, 4, 0, 5, 1, 6, 2});
        auto squeezed_order = opset3::Constant::create(element::i64, Shape{4}, {2, 70, 65, 3});

        auto reshape_before = std::make_shared<opset3::Reshape>(data, dispresed_shape, false);
        auto permute = std::make_shared<opset3::Transpose>(reshape_before, axis_order);
        auto reshape_after = std::make_shared<opset3::Reshape>(permute, squeezed_order, false);

        auto begin = opset3::Constant::create(element::i64, Shape{4}, {0, 3, 1, 0});
        auto end = opset3::Constant::create(element::i64, Shape{4}, {2, 67, 65, 3});
        std::vector<int64_t> begin_mask(4, 0);
        std::vector<int64_t> end_mask(4, 0);
        auto ss = std::make_shared<opset3::StridedSlice>(reshape_after, begin, end, begin_mask, end_mask);
        model_ref = std::make_shared<ov::Model>(NodeVector{ss}, ParameterVector{data});
    }
}

template <typename Op, typename Conversion, typename Params>
void op_convertion_type_test(const Params& params) {
    using namespace ov::opset10;
    using namespace ov::pass;

    const auto by_elements = get<0>(params);
    const auto block_elem_type = get<1>(params);

    const auto data = make_shared<Parameter>(element::f32, Shape{1, 1});
    const auto block_p = Constant::create(block_elem_type, Shape{2}, {1, 1});
    const auto input_2_p = Constant::create(block_elem_type, Shape{2}, {0, 0});
    const auto input_3_p = Constant::create(block_elem_type, Shape{2}, {0, 0});
    const auto bts_or_stb = make_shared<Op>(data, block_p, input_2_p, input_3_p);
    const auto f = make_shared<Model>(NodeVector{bts_or_stb}, ParameterVector{data});

    Manager m;
    m.register_pass<Conversion>(by_elements);
    m.register_pass<ConstantFolding>();
    OV_ASSERT_NO_THROW(m.run_passes(f));
    EXPECT_EQ(f->get_result()->get_input_shape(0), (Shape{1, 1}));
}

using ElementTypeParams = tuple<bool,          // by_elements
                                element::Type  // block element type
                                >;

class BatchToSpaceDecomposition2D : public testing::WithParamInterface<ElementTypeParams>,
                                    public TransformationTests {};

TEST_P(BatchToSpaceDecomposition2D, BlockElemType) {
    op_convertion_type_test<ov::opset10::BatchToSpace, ov::pass::ConvertBatchToSpace>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         BatchToSpaceDecomposition2D,
                         ::testing::Combine(::testing::ValuesIn({false, true}),
                                            ::testing::ValuesIn({element::i32, element::i64})));

class SpaceToBatchDecomposition2D : public testing::WithParamInterface<ElementTypeParams>,
                                    public TransformationTests {};

TEST_P(SpaceToBatchDecomposition2D, BlockElemType) {
    op_convertion_type_test<ov::opset10::SpaceToBatch, ov::pass::ConvertSpaceToBatch>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         SpaceToBatchDecomposition2D,
                         ::testing::Combine(::testing::ValuesIn({false, true}),
                                            ::testing::ValuesIn({element::i32, element::i64})));

template <typename Op, typename Conversion, typename Params>
void op_convertion_test(const Params& params) {
    using namespace ov::opset10;
    using namespace ov::pass;

    const bool by_elements = get<0>(params);
    Shape data_shape;
    Shape expected_output_shape;
    vector<int64_t> block;
    vector<int64_t> input_2;  // crops_begin or pads_begin
    vector<int64_t> input_3;  // crops_end or pads_end
    tie(data_shape, block, input_2, input_3, expected_output_shape) = get<1>(params);

    const auto data = make_shared<Parameter>(element::f32, PartialShape::dynamic(data_shape.size()));
    const auto block_p = Constant::create(element::i64, Shape{block.size()}, block);
    const auto input_2_p = Constant::create(element::i64, Shape{input_2.size()}, input_2);
    const auto input_3_p = Constant::create(element::i64, Shape{input_3.size()}, input_3);
    const auto bts_or_stb = make_shared<Op>(data, block_p, input_2_p, input_3_p);
    const auto f = make_shared<Model>(NodeVector{bts_or_stb}, ParameterVector{data});

    Manager m;
    m.set_per_pass_validation(false);
    m.register_pass<Conversion>(by_elements);
    m.run_passes(f);
    ASSERT_EQ(count_ops_of_type<Op>(f), 0);
    EXPECT_TRUE(f->get_result()->get_input_partial_shape(0).is_dynamic());

    data->set_partial_shape(data_shape);
    f->validate_nodes_and_infer_types();
    ASSERT_EQ(f->get_result()->get_input_shape(0), expected_output_shape);
}

template <typename Params>
string get_test_name(testing::TestParamInfo<Params> obj) {
    const auto& params = obj.param;
    const bool by_elements = get<0>(params);
    const auto& data_shape = get<0>(get<1>(params));

    ostringstream result;
    result << data_shape.size() << "D" << (by_elements ? "_by_elements" : "");
    return result.str();
}

using BatchToSpaceParams = tuple<Shape,            // data_shape
                                 vector<int64_t>,  // block
                                 vector<int64_t>,  // crops_begin
                                 vector<int64_t>,  // crops_end
                                 Shape             // expected_output_shape
                                 >;

using BatchToSpaceDecomposeParams = tuple<bool,  // by_elements
                                          BatchToSpaceParams>;

class BatchToSpaceDecompositionWithParams : public testing::WithParamInterface<BatchToSpaceDecomposeParams>,
                                            public TransformationTests {};

TEST_P(BatchToSpaceDecompositionWithParams, DynamicInputs) {
    op_convertion_test<ov::opset10::BatchToSpace, ov::pass::ConvertBatchToSpace>(GetParam());
}

static vector<BatchToSpaceParams> batch_to_space_params = {
    {{4, 3}, {1, 2}, {0, 0}, {0, 0}, {2, 6}},
    {{6, 5, 7}, {1, 2, 3}, {0, 1, 2}, {0, 1, 2}, {1, 8, 17}},
    {{30, 4, 1, 1}, {1, 5, 3, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 20, 3, 2}},
    {{96, 3, 5, 7, 1}, {1, 4, 3, 2, 1}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {4, 12, 15, 14, 1}},
};

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         BatchToSpaceDecompositionWithParams,
                         ::testing::Combine(::testing::ValuesIn({false, true}),
                                            ::testing::ValuesIn(batch_to_space_params)),
                         get_test_name<BatchToSpaceDecomposeParams>);

using SpaceToBatchParams = tuple<Shape,            // data_shape
                                 vector<int64_t>,  // block
                                 vector<int64_t>,  // pads_begin
                                 vector<int64_t>,  // pads_end
                                 Shape             // expected_output_shape
                                 >;

using SpaceToBatchDecomposeParams = tuple<bool,  // by_elements
                                          SpaceToBatchParams>;

class SpaceToBatchDecompositionWithParams : public testing::WithParamInterface<SpaceToBatchDecomposeParams>,
                                            public TransformationTests {};

TEST_P(SpaceToBatchDecompositionWithParams, DynamicInputs) {
    op_convertion_test<ov::opset10::SpaceToBatch, ov::pass::ConvertSpaceToBatch>(GetParam());
}

static vector<SpaceToBatchParams> space_to_batch_params = {
    {{2, 6}, {1, 2}, {0, 0}, {0, 0}, {4, 3}},
    {{1, 8, 17}, {1, 2, 3}, {0, 1, 2}, {0, 1, 2}, {6, 5, 7}},
    {{1, 20, 3, 2}, {1, 5, 3, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {30, 4, 1, 1}},
    {{4, 12, 15, 14, 1}, {1, 4, 3, 2, 1}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {96, 3, 5, 7, 1}},
};

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         SpaceToBatchDecompositionWithParams,
                         ::testing::Combine(::testing::ValuesIn({false, true}),
                                            ::testing::ValuesIn(space_to_batch_params)),
                         get_test_name<SpaceToBatchDecomposeParams>);
