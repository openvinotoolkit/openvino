// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_nms_gather_path_to_unsigned.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;
using namespace std;

TEST_F(TransformationTestsF, test_convert_to_unsigned_nms_gather_1) {
    // if Convert doesn't exist
    {
        auto boxes = make_shared<opset8::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = make_shared<opset8::NonMaxSuppression>(boxes, scores);

        auto begin = opset8::Constant::create(element::i32, Shape{1}, {3});
        auto end = opset8::Constant::create(element::i32, Shape{1}, {4});
        auto strides = opset8::Constant::create(element::i32, Shape{1}, {1});
        auto ss_node = make_shared<opset8::StridedSlice>(nms->output(0),
                                                         begin,
                                                         end,
                                                         strides,
                                                         vector<int64_t>{1, 0},
                                                         vector<int64_t>{1, 0});

        // squeeze can be represented as reshape
        auto squeeze_node =
            make_shared<opset8::Reshape>(ss_node, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        // usually input to gather data goes after reshape NMS scores
        auto reshape_node =
            make_shared<opset8::Reshape>(scores, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto gather = make_shared<opset8::Gather>(reshape_node,
                                                  squeeze_node,
                                                  opset8::Constant::create(element::i32, Shape{1}, {0}));

        model = make_shared<Model>(NodeVector{gather}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNmsGatherPathToUnsigned>();
    }

    {
        auto boxes = make_shared<opset8::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = make_shared<opset8::NonMaxSuppression>(boxes, scores);

        auto begin = opset8::Constant::create(element::i32, Shape{1}, {3});
        auto end = opset8::Constant::create(element::i32, Shape{1}, {4});
        auto strides = opset8::Constant::create(element::i32, Shape{1}, {1});
        auto ss_node = make_shared<opset8::StridedSlice>(nms->output(0),
                                                         begin,
                                                         end,
                                                         strides,
                                                         vector<int64_t>{1, 0},
                                                         vector<int64_t>{1, 0});

        // squeeze can be represented as reshape
        auto squeeze_node =
            make_shared<opset8::Reshape>(ss_node, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto convert = make_shared<opset8::Convert>(squeeze_node, element::Type_t::u64);
        auto reshape_node =
            make_shared<opset8::Reshape>(scores, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto gather =
            make_shared<opset8::Gather>(reshape_node, convert, opset8::Constant::create(element::i32, Shape{1}, {0}));

        model_ref = make_shared<Model>(NodeVector{gather}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, test_convert_to_unsigned_nms_gather_2) {
    // if Convert already exists
    {
        auto boxes = make_shared<opset8::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = make_shared<opset8::NonMaxSuppression>(boxes, scores);

        auto begin = opset8::Constant::create(element::i32, Shape{1}, {3});
        auto end = opset8::Constant::create(element::i32, Shape{1}, {4});
        auto strides = opset8::Constant::create(element::i32, Shape{1}, {1});
        auto ss_node = make_shared<opset8::StridedSlice>(nms->output(0),
                                                         begin,
                                                         end,
                                                         strides,
                                                         vector<int64_t>{1, 0},
                                                         vector<int64_t>{1, 0});

        // squeeze can be represented as reshape
        auto squeeze_node =
            make_shared<opset8::Reshape>(ss_node, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto convert = make_shared<opset8::Convert>(squeeze_node, element::Type_t::i32);
        // usually input to gather data goes after reshape NMS scores
        auto reshape_node =
            make_shared<opset8::Reshape>(scores, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto gather =
            make_shared<opset8::Gather>(reshape_node, convert, opset8::Constant::create(element::i32, Shape{1}, {0}));

        model = make_shared<Model>(NodeVector{gather}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNmsGatherPathToUnsigned>();
    }

    {
        auto boxes = make_shared<opset8::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = make_shared<opset8::NonMaxSuppression>(boxes, scores);

        auto begin = opset8::Constant::create(element::i32, Shape{1}, {3});
        auto end = opset8::Constant::create(element::i32, Shape{1}, {4});
        auto strides = opset8::Constant::create(element::i32, Shape{1}, {1});
        auto ss_node = make_shared<opset8::StridedSlice>(nms->output(0),
                                                         begin,
                                                         end,
                                                         strides,
                                                         vector<int64_t>{1, 0},
                                                         vector<int64_t>{1, 0});

        // squeeze can be represented as reshape
        auto squeeze_node =
            make_shared<opset8::Reshape>(ss_node, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto convert = make_shared<opset8::Convert>(squeeze_node, element::Type_t::u32);
        auto reshape_node =
            make_shared<opset8::Reshape>(scores, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto gather =
            make_shared<opset8::Gather>(reshape_node, convert, opset8::Constant::create(element::i32, Shape{1}, {0}));

        model_ref = make_shared<Model>(NodeVector{gather}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, test_convert_to_unsigned_nms_gather_with_onnx_slice) {
    // if Convert already exists and Slice is present instead of StridedSlice
    {
        auto boxes = make_shared<opset8::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = make_shared<opset8::NonMaxSuppression>(boxes, scores);

        auto start = opset8::Constant::create(element::i32, Shape{1}, {3});
        auto stop = opset8::Constant::create(element::i32, Shape{1}, {4});
        auto step = opset8::Constant::create(element::i32, Shape{1}, {1});
        auto slice_node = make_shared<opset8::Slice>(nms->output(0), start, stop, step);

        // squeeze can be represented as reshape
        auto squeeze_node =
            make_shared<opset8::Reshape>(slice_node, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto convert = make_shared<opset8::Convert>(squeeze_node, element::Type_t::i32);
        // usually input to gather data goes after reshape NMS scores
        auto reshape_node =
            make_shared<opset8::Reshape>(scores, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto gather =
            make_shared<opset8::Gather>(reshape_node, convert, opset8::Constant::create(element::i32, Shape{1}, {0}));

        model = make_shared<Model>(NodeVector{gather}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNmsGatherPathToUnsigned>();
    }

    {
        auto boxes = make_shared<opset8::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = make_shared<opset8::NonMaxSuppression>(boxes, scores);

        auto start = opset8::Constant::create(element::i32, Shape{1}, {3});
        auto stop = opset8::Constant::create(element::i32, Shape{1}, {4});
        auto step = opset8::Constant::create(element::i32, Shape{1}, {1});
        auto slice_node = make_shared<opset8::Slice>(nms->output(0), start, stop, step);

        // squeeze can be represented as reshape
        auto squeeze_node =
            make_shared<opset8::Reshape>(slice_node, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto convert = make_shared<opset8::Convert>(squeeze_node, element::Type_t::u32);
        auto reshape_node =
            make_shared<opset8::Reshape>(scores, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto gather =
            make_shared<opset8::Gather>(reshape_node, convert, opset8::Constant::create(element::i32, Shape{1}, {0}));

        model_ref = make_shared<Model>(NodeVector{gather}, ParameterVector{boxes, scores});
    }
}

TEST(TransformationTests, test_convert_to_unsigned_nms_gather_3) {
    // if NMS output goes not into Gather indices no converts should be inserted
    auto boxes = make_shared<opset8::Parameter>(element::f32, Shape{1, 1000, 4});
    auto scores = make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 1000});
    auto nms = make_shared<opset8::NonMaxSuppression>(boxes, scores);

    auto gather = make_shared<opset8::Gather>(nms->output(0),
                                              opset8::Constant::create(element::i32, Shape{1}, {2}),
                                              opset8::Constant::create(element::i32, Shape{1}, {0}));

    shared_ptr<Model> f = make_shared<Model>(NodeVector{gather}, ParameterVector{boxes, scores});

    pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::ConvertNmsGatherPathToUnsigned>();
    manager.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Convert>(f), 0);
}

TEST(TransformationTests, test_convert_to_unsigned_nms_gather_with_if_condition) {
    auto boxes = make_shared<opset8::Parameter>(element::f32, PartialShape{1, -1, 4});
    auto scores = make_shared<opset8::Parameter>(element::f32, PartialShape{1, 1, -1});
    auto nms = make_shared<opset8::NonMaxSuppression>(boxes, scores);

    auto gather = make_shared<opset8::Gather>(nms->output(0),
                                              opset8::Constant::create(element::i32, Shape{1}, {2}),
                                              opset8::Constant::create(element::i32, Shape{1}, {0}));

    auto shape_of = make_shared<opset8::ShapeOf>(gather);
    auto gather_shape = make_shared<opset8::Gather>(shape_of,
                                                    opset8::Constant::create(element::i32, Shape{1}, {0}),
                                                    opset8::Constant::create(element::i32, Shape{1}, {0}));
    auto equal = make_shared<opset8::Equal>(gather_shape, opset8::Constant::create(element::i64, Shape{1}, {1}));
    auto if_op = make_shared<opset8::If>(equal);

    auto input_then = make_shared<opset8::Parameter>(element::i32, PartialShape{-1, 1});

    auto start = opset8::Constant::create(element::i32, Shape{1}, {3});
    auto stop = opset8::Constant::create(element::i32, Shape{1}, {4});
    auto step = opset8::Constant::create(element::i32, Shape{1}, {1});
    auto slice = make_shared<opset8::Slice>(input_then, start, stop, step);

    auto then_op_result = make_shared<op::v0::Result>(slice);
    auto body_then_function = make_shared<Model>(NodeVector{then_op_result}, ParameterVector{input_then});

    auto input_else = make_shared<opset8::Parameter>(element::i32, PartialShape{-1, 1});
    auto reshape =
        make_shared<opset8::Reshape>(input_else, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
    auto else_op_result = make_shared<op::v0::Result>(reshape);
    auto body_else_function = make_shared<Model>(NodeVector{else_op_result}, ParameterVector{input_else});

    if_op->set_then_body(body_then_function);
    if_op->set_else_body(body_else_function);
    if_op->set_input(gather, input_then, input_else);

    auto result_if = if_op->set_output(then_op_result, else_op_result);

    auto begin = opset8::Constant::create(element::i32, Shape{1}, {3});
    auto end = opset8::Constant::create(element::i32, Shape{1}, {4});
    auto strides = opset8::Constant::create(element::i32, Shape{1}, {1});
    auto ss_node =
        make_shared<opset8::StridedSlice>(result_if, begin, end, strides, vector<int64_t>{1, 0}, vector<int64_t>{1, 0});

    auto data = make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});
    auto axis = opset8::Constant::create(element::i32, Shape{1}, {0});
    auto target_gather = make_shared<opset8::Gather>(data, ss_node, axis);

    shared_ptr<Model> f = make_shared<Model>(NodeVector{target_gather}, ParameterVector{boxes, scores, data});

    pass::Manager manager;
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertNmsGatherPathToUnsigned>();
    manager.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));

    const auto& ops = f->get_ops();
    const auto& gather_it = find(ops.begin(), ops.end(), target_gather);
    ASSERT_NE(gather_it, ops.end());

    const auto& rti = (*gather_it)->get_rt_info();
    const auto& reverse = rti.find("dontReverseIndices");
    ASSERT_NE(reverse, rti.end());
}
