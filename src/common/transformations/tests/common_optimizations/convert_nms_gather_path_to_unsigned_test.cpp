// Copyright (C) 2018-2023 Intel Corporation
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
    ASSERT_NO_THROW(check_rt_info(f));
    ASSERT_EQ(count_ops_of_type<ov::op::v0::Convert>(f), 0);
}
