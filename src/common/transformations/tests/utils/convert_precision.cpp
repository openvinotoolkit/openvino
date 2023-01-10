// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset5.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>
#include <queue>
#include <string>
#include <transformations/convert_precision.hpp>
#include <transformations/utils/utils.hpp>
#include <vector>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "transformations/common_optimizations/mark_precision_sensitive_shapeof_subgraphs.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

using namespace testing;
using namespace ov;

template <element::Type_t T>
bool has_type(std::shared_ptr<Model> f) {
    for (auto& node : f->get_ordered_ops()) {
        for (auto& input : node->inputs()) {
            if (input.get_element_type() == element::Type(T)) {
                return true;
            }
        }
        for (auto& output : node->outputs()) {
            if (output.get_element_type() == element::Type(T)) {
                return true;
            }
        }
    }
    return false;
}

TEST(TransformationTests, ConvertPrecision_NMS3) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto boxes = std::make_shared<opset3::Parameter>(element::f16, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset3::Parameter>(element::f16, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset3::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset3::Constant::create(element::f16, Shape{}, {0.75});
        auto score_threshold = opset3::Constant::create(element::f16, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset3::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        f = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::Manager manager;

        static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_NMS4) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto boxes = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset4::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset4::Constant::create(element::f16, Shape{}, {0.75});
        auto score_threshold = opset4::Constant::create(element::f16, Shape{}, {0.7});
        auto nms = std::make_shared<opset4::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset4::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        f = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::Manager manager;

        static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_NMS5) {
    std::shared_ptr<Model> f;
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        auto result1 = std::make_shared<opset5::Result>(nms->output(0));
        auto result2 = std::make_shared<opset5::Result>(nms->output(1));
        auto result3 = std::make_shared<opset5::Result>(nms->output(2));
        f = std::make_shared<Model>(ResultVector{result1, result2, result3}, ParameterVector{boxes, scores});
    }

    pass::Manager manager;
    static const precisions_array precisions = {{element::i64, element::i32}, {element::f32, element::f16}};
    manager.register_pass<pass::ConvertPrecision>(precisions);
    manager.run_passes(f);
    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<element::Type_t::f32>(f));
}

TEST(TransformationTests, ConvertPrecision_MatrixNms) {
    std::shared_ptr<Model> f;
    {
        auto boxes = std::make_shared<opset8::Parameter>(element::f16, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset8::Parameter>(element::f16, Shape{1, 1, 1000});
        op::v8::MatrixNms::Attributes attrs;
        attrs.output_type = element::i64;
        auto nms = std::make_shared<opset8::MatrixNms>(boxes, scores, attrs);

        auto result1 = std::make_shared<opset8::Result>(nms->output(0));
        auto result2 = std::make_shared<opset8::Result>(nms->output(1));
        auto result3 = std::make_shared<opset8::Result>(nms->output(2));
        f = std::make_shared<Model>(ResultVector{result1, result2, result3}, ParameterVector{boxes, scores});
    }

    pass::Manager manager;
    static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};
    manager.register_pass<pass::ConvertPrecision>(precisions);
    manager.run_passes(f);
    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_MulticlassNms) {
    std::shared_ptr<Model> f;
    {
        auto boxes = std::make_shared<opset8::Parameter>(element::f16, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset8::Parameter>(element::f16, Shape{1, 1, 1000});
        op::v8::MulticlassNms::Attributes attrs;
        attrs.output_type = element::i64;
        auto nms = std::make_shared<opset8::MulticlassNms>(boxes, scores, attrs);

        auto result1 = std::make_shared<opset8::Result>(nms->output(0));
        auto result2 = std::make_shared<opset8::Result>(nms->output(1));
        auto result3 = std::make_shared<opset8::Result>(nms->output(2));
        f = std::make_shared<Model>(ResultVector{result1, result2, result3}, ParameterVector{boxes, scores});
    }

    pass::Manager manager;
    static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};
    manager.register_pass<pass::ConvertPrecision>(precisions);
    manager.run_passes(f);
    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_ShapeOf) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 4});
        auto shape_of = std::make_shared<opset4::ShapeOf>(input);

        f = std::make_shared<Model>(NodeVector{shape_of}, ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_Range) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto start = std::make_shared<opset4::Parameter>(element::f16, Shape{});
        auto stop = std::make_shared<opset4::Parameter>(element::f16, Shape{});
        auto shift = std::make_shared<opset4::Parameter>(element::f16, Shape{});
        auto range = std::make_shared<opset4::Range>(start, stop, shift, element::i64);

        f = std::make_shared<Model>(NodeVector{range}, ParameterVector{start, stop, shift});

        pass::Manager manager;

        static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_ConstantRelu) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input = opset4::Constant::create(element::f16, Shape{1, 1000, 4}, {0});
        auto relu1 = std::make_shared<opset4::Relu>(input);
        auto relu2 = std::make_shared<opset4::Relu>(relu1);

        f = std::make_shared<Model>(NodeVector{relu2}, ParameterVector{});

        pass::Manager manager;

        static const precisions_array precisions = {{element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_Convert) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 4});
        auto convert = std::make_shared<opset4::Convert>(input, element::i64);

        f = std::make_shared<Model>(NodeVector{convert}, ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_ConvertElimination) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 4});
        auto relu = std::make_shared<opset4::Relu>(input);
        auto convert = std::make_shared<opset4::Convert>(relu, element::f32);

        f = std::make_shared<Model>(NodeVector{convert}, ParameterVector{input});

        pass::Manager manager;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::f16, element::f32}});
        manager.run_passes(f);
        ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1000, 4});
        auto relu = std::make_shared<opset4::Relu>(input);

        f_ref = std::make_shared<Model>(NodeVector{relu}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertPrecision_TopK) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input = std::make_shared<opset3::Parameter>(element::f16, Shape{15, 20, 3});
        auto k = opset3::Constant::create(element::i64, Shape{}, {10});
        auto topk = std::make_shared<opset3::TopK>(input, k, 1, "min", "value", element::i64);

        f = std::make_shared<Model>(OutputVector{topk->output(0), topk->output(1)}, ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_Unique10) {
    std::shared_ptr<Model> model(nullptr);
    {
        auto input = std::make_shared<opset10::Parameter>(element::f16, Shape{15, 20, 3});
        auto unique = std::make_shared<opset10::Unique>(input);

        model = std::make_shared<Model>(unique->outputs(), ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<ov::pass::ConvertPrecision>(precisions);
        manager.run_passes(model);
    }

    ASSERT_EQ(model->outputs().size(), 4);
    EXPECT_EQ(model->outputs()[0].get_element_type(), element::f32);
    EXPECT_EQ(model->outputs()[1].get_element_type(), element::i32);
    EXPECT_EQ(model->outputs()[2].get_element_type(), element::i32);
    EXPECT_EQ(model->outputs()[3].get_element_type(), element::i32);

    EXPECT_EQ(model->get_results().size(), 4);

    EXPECT_FALSE(has_type<element::Type_t::f16>(model));
    EXPECT_FALSE(has_type<element::Type_t::i64>(model));
}

TEST(TransformationTests, ConvertPrecision_NonZero) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{15, 20, 3});
        auto non_zero = std::make_shared<opset4::NonZero>(input, element::i64);

        f = std::make_shared<Model>(OutputVector{non_zero}, ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_Bucketize) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{20});
        auto k = opset4::Constant::create(element::i64, Shape{1}, {10});
        auto b = std::make_shared<opset4::Bucketize>(input, k);

        f = std::make_shared<Model>(OutputVector{b}, ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_Roundings) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto max_int64 = std::numeric_limits<int64_t>::max();
        auto max_int32 = std::numeric_limits<int32_t>::max();

        auto input = std::make_shared<opset1::Parameter>(element::f16, Shape{5, 5, 5, 5});
        auto begin = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset1::Constant::create(element::i64, Shape{4}, {max_int64, max_int64, max_int64, max_int64});
        auto stride = opset1::Constant::create(element::i64, Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto ss = std::make_shared<opset1::StridedSlice>(input, begin, end, stride, begin_mask, end_mask);

        f = std::make_shared<Model>(OutputVector{ss}, ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);

        auto casted_end = std::dynamic_pointer_cast<opset1::Constant>(ss->input_value(2).get_node_shared_ptr());
        ASSERT_TRUE(casted_end != nullptr);
        ASSERT_EQ(casted_end->get_element_type(), element::i32);
        ASSERT_EQ(casted_end->cast_vector<int32_t>(),
                  std::vector<int32_t>({max_int32, max_int32, max_int32, max_int32}));
    }

    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_TIBody) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset4::Parameter>(element::f16, Shape{2, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 128});

        auto Xi = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 128});

        // Body
        auto axis = opset4::Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<opset4::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = opset4::Constant::create(element::f16, Shape{384, 16}, w_val);
        auto R = opset4::Constant::create(element::f16, Shape{384, 128}, r_val);
        auto B = opset4::Constant::create(element::f16, Shape{384}, b_val);

        auto gru_cell = std::make_shared<opset4::GRUCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset4::Result>(gru_cell);
        auto unsqueeze = std::make_shared<opset4::Unsqueeze>(gru_cell, axis);
        auto res_2 = std::make_shared<opset4::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset4::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(tensor_iterator->output(1));
        // auto res_ti_2 = std::make_shared<opset4::Result>(tensor_iterator->output(0));
        f = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager manager;

        static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);

        ASSERT_FALSE(has_type<element::Type_t::f16>(f));
        ASSERT_FALSE(has_type<element::Type_t::i64>(f));
        ASSERT_FALSE(has_type<element::Type_t::f16>(tensor_iterator->get_body()));
        ASSERT_FALSE(has_type<element::Type_t::i64>(tensor_iterator->get_body()));
    }
}

TEST(TransformationTests, ConvertPrecision_Equal) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::f16, Shape{15, 20, 3});
        auto input2 = std::make_shared<opset4::Parameter>(element::f16, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::Equal>(input1, input2);

        f = std::make_shared<Model>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;

        static const precisions_array precisions = {{element::boolean, element::u8}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_NotEqual) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::f16, Shape{15, 20, 3});
        auto input2 = std::make_shared<opset4::Parameter>(element::f16, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::NotEqual>(input1, input2);

        f = std::make_shared<Model>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;

        static const precisions_array precisions = {{element::boolean, element::u8}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_Greater) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::f16, Shape{15, 20, 3});
        auto input2 = std::make_shared<opset4::Parameter>(element::f16, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::Greater>(input1, input2);

        f = std::make_shared<Model>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;

        static const precisions_array precisions = {{element::boolean, element::u8}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_GreaterEqual) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::f16, Shape{15, 20, 3});
        auto input2 = std::make_shared<opset4::Parameter>(element::f16, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::GreaterEqual>(input1, input2);

        f = std::make_shared<Model>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;

        static const precisions_array precisions = {{element::boolean, element::u8}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_Less) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::f16, Shape{15, 20, 3});
        auto input2 = std::make_shared<opset4::Parameter>(element::f16, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::Less>(input1, input2);

        f = std::make_shared<Model>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;

        static const precisions_array precisions = {{element::boolean, element::u8}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_LessEqual) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::f16, Shape{15, 20, 3});
        auto input2 = std::make_shared<opset4::Parameter>(element::f16, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::LessEqual>(input1, input2);

        f = std::make_shared<Model>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;

        static const precisions_array precisions = {{element::boolean, element::u8}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_LogicalAnd) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::boolean, Shape{15, 20, 3});
        auto input2 = std::make_shared<opset4::Parameter>(element::boolean, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::LogicalAnd>(input1, input2);

        f = std::make_shared<Model>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::boolean, element::u8}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_LogicalOr) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::boolean, Shape{15, 20, 3});
        auto input2 = std::make_shared<opset4::Parameter>(element::boolean, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::LogicalOr>(input1, input2);

        f = std::make_shared<Model>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::boolean, element::u8}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_LogicalXor) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::boolean, Shape{15, 20, 3});
        auto input2 = std::make_shared<opset4::Parameter>(element::boolean, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::LogicalXor>(input1, input2);

        f = std::make_shared<Model>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::boolean, element::u8}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_LogicalNot) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::boolean, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::LogicalNot>(input1);

        f = std::make_shared<Model>(OutputVector{node}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::boolean, element::u8}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_Select) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::boolean, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::LogicalNot>(input1);
        auto select = std::make_shared<opset4::Select>(node, input1, input1);

        f = std::make_shared<Model>(OutputVector{select}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::boolean, element::u8}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_TypeRelaxedWithSelect) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::boolean, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::LogicalNot>(input1);
        auto select = std::make_shared<opset4::Select>(node, input1, input1);

        f = std::make_shared<Model>(OutputVector{select}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::boolean, element::i32}});
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::i32, element::i64}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_FALSE(has_type<element::Type_t::i32>(f));
    ASSERT_TRUE(has_type<element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_TypeRelaxed) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::boolean, Shape{15, 20, 3});
        auto select = std::make_shared<opset4::Select>(input1, input1, input1);
        auto type_relaxed = std::make_shared<op::TypeRelaxed<opset4::Select>>(*select,
                                                                              element::TypeVector{},
                                                                              element::TypeVector{element::i64});

        f = std::make_shared<Model>(OutputVector{type_relaxed}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::boolean, element::i32}});
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::i32, element::i64}});
        manager.run_passes(f);

        ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
        ASSERT_FALSE(has_type<element::Type_t::i32>(f));
        ASSERT_TRUE(has_type<element::Type_t::i64>(f));
    }
}

TEST(TransformationTests, ConvertPrecision_Variables) {
    std::shared_ptr<Model> f(nullptr);
    {
        Shape shape{1, 10, 2};
        auto inp = std::make_shared<opset4::Parameter>(element::f16, shape);
        auto m_i = std::make_shared<opset4::Constant>(element::f16, shape, 1);
        auto m_r = std::make_shared<opset4::ReadValue>(m_i, "ID");
        auto sum = std::make_shared<opset4::Add>(inp, m_r);
        auto m_w = std::make_shared<opset4::Assign>(sum, "ID");
        auto mul = std::make_shared<opset4::Multiply>(inp, sum);

        mul->add_control_dependency(m_w);

        f = std::make_shared<Model>(NodeVector{mul}, ParameterVector{inp});

        pass::Manager manager;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::f16, element::f32}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_skip_precision_sensitive) {
    std::shared_ptr<Model> model(nullptr);
    std::shared_ptr<opset10::Interpolate> interpolate(nullptr);
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto sizes = opset10::Constant::create(element::i64, Shape{4}, {1, 3, 288, 512});
        auto scales = opset10::Constant::create(element::f32, Shape{4}, {1.0f, 1.0f, 0.4f, 0.4f});
        opset10::Interpolate::InterpolateAttrs attrs;

        attrs.mode = opset10::Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = opset10::Interpolate::ShapeCalcMode::SCALES;
        attrs.nearest_mode = opset10::Interpolate::NearestMode::FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = opset10::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        interpolate = std::make_shared<opset10::Interpolate>(input, sizes, scales, attrs);
        model = std::make_shared<Model>(NodeVector{interpolate}, ParameterVector{input});

        pass::Manager manager;
        manager.register_pass<pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    ASSERT_TRUE(has_type<element::Type_t::f32>(model));
    ASSERT_TRUE(interpolate->input_value(2).get_element_type() == element::Type_t::f32);
}

TEST(TransformationTests, ConvertPrecision_without_keep_precision_sensitive_in_fp32) {
    // with keep_precision_sensitive_in_fp32 = false all nodes should be converted to f16 even they are marked
    std::shared_ptr<Model> model(nullptr);
    std::shared_ptr<opset10::Interpolate> interpolate(nullptr);
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto sizes = opset10::Constant::create(element::i64, Shape{4}, {1, 3, 288, 512});
        auto scales = opset10::Constant::create(element::f32, Shape{4}, {1.0f, 1.0f, 0.4f, 0.4f});
        opset10::Interpolate::InterpolateAttrs attrs;

        attrs.mode = opset10::Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = opset10::Interpolate::ShapeCalcMode::SCALES;
        attrs.nearest_mode = opset10::Interpolate::NearestMode::FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = opset10::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        interpolate = std::make_shared<opset10::Interpolate>(input, sizes, scales, attrs);
        model = std::make_shared<Model>(NodeVector{interpolate}, ParameterVector{input});
        pass::Manager manager;
        manager.register_pass<pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = false;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    ASSERT_FALSE(has_type<element::Type_t::f32>(model));
    ASSERT_TRUE(interpolate->input_value(2).get_element_type() == element::Type_t::f16);
}

TEST(TransformationTests, ConvertPrecision_check_marking_does_not_leak_in_trivial_case) {
    std::shared_ptr<Model> model(nullptr), model_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto new_shape = std::make_shared<opset10::ShapeOf>(input_2);
        auto reshape = std::make_shared<opset10::Reshape>(input_1, new_shape, false);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 720, 1280});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 720, 1280});
        auto new_shape = std::make_shared<opset10::ShapeOf>(input_2);
        auto reshape = std::make_shared<opset10::Reshape>(input_1, new_shape, false);

        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, ConvertPrecision_whole_shape_subgraph_is_marked_1) {
    std::shared_ptr<Model> model(nullptr), model_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{360, 640});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f32, Shape{720, 1280});
        auto shapeof = std::make_shared<opset10::ShapeOf>(input_2);

        auto convert_to_float = std::make_shared<opset10::Convert>(shapeof, element::f32);
        auto const_denominator = opset10::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset10::Divide>(convert_to_float, const_denominator);
        auto new_shape = std::make_shared<opset10::Convert>(div, element::i64);

        auto reshape = std::make_shared<opset10::Reshape>(input_1, new_shape, false);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f16, Shape{360, 640});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f16, Shape{720, 1280});
        auto shapeof_1 = std::make_shared<opset10::ShapeOf>(input_2);

        auto convert_to_float = std::make_shared<opset10::Convert>(shapeof_1, element::f32);
        auto const_denominator = opset10::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset10::Divide>(convert_to_float, const_denominator);
        auto new_shape = std::make_shared<opset10::Convert>(div, element::i64);

        auto reshape = std::make_shared<opset10::Reshape>(input_1, new_shape, false);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, ConvertPrecision_whole_shape_subgraph_is_marked_2) {
    std::shared_ptr<Model> model(nullptr), model_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 1, 720, 1280});
        auto shapeof_1 = std::make_shared<opset10::ShapeOf>(input_1);
        auto indices = opset10::Constant::create(element::i64, Shape{1}, {3});
        auto gather_axis = opset10::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset10::Gather>(shapeof_1, indices, gather_axis);

        auto convert_to_float = std::make_shared<opset10::Convert>(gather_1, element::f32);
        auto const_denominator = opset10::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset10::Divide>(convert_to_float, const_denominator);
        auto new_dim_size = std::make_shared<opset10::Convert>(div, element::i64);

        auto const_ends = opset10::Constant::create(element::i64, Shape{3}, {-1, -1, -1});
        auto concat_with_ends = std::make_shared<opset10::Concat>(OutputVector{const_ends, new_dim_size}, 0);  // scales

        auto begin = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};
        auto slice = std::make_shared<opset10::StridedSlice>(input_1, begin, concat_with_ends, begin_mask, end_mask);
        auto result = std::make_shared<opset10::Result>(slice);
        model = std::make_shared<Model>(NodeVector{result}, ParameterVector{input_1});

        pass::Manager manager;
        manager.register_pass<pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f16, Shape{1, 1, 720, 1280});
        auto shapeof_1 = std::make_shared<opset10::ShapeOf>(input_1);
        auto indices = opset10::Constant::create(element::i64, Shape{1}, {3});
        auto gather_axis = opset10::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset10::Gather>(shapeof_1, indices, gather_axis);

        auto convert_to_float = std::make_shared<opset10::Convert>(gather_1, element::f32);
        auto const_denominator = opset10::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset10::Divide>(convert_to_float, const_denominator);
        auto new_dim_size = std::make_shared<opset10::Convert>(div, element::i64);

        auto const_ends = opset10::Constant::create(element::i64, Shape{3}, {-1, -1, -1});
        auto concat_with_ends = std::make_shared<opset10::Concat>(OutputVector{const_ends, new_dim_size}, 0);  // scales

        auto begin = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};
        auto slice = std::make_shared<opset10::StridedSlice>(input_1, begin, concat_with_ends, begin_mask, end_mask);
        auto result = std::make_shared<opset10::Result>(slice);
        model_ref = std::make_shared<Model>(NodeVector{result}, ParameterVector{input_1});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, ConvertPrecision_whole_shape_subgraph_is_marked_3) {
    std::shared_ptr<Model> model(nullptr);
    std::shared_ptr<Model> model_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto shapeof_1 = std::make_shared<opset10::ShapeOf>(input_1);
        auto shapeof_2 = std::make_shared<opset10::ShapeOf>(input_2);

        auto const_1 = opset10::Constant::create(element::i64, Shape{2}, {2, 3});
        auto axis_const = opset10::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset10::Gather>(shapeof_2, const_1, axis_const);
        auto convert_1 = std::make_shared<opset10::Convert>(shapeof_1, element::f32);

        auto convert_2 = std::make_shared<opset10::Convert>(gather_1, element::f32);
        auto const_2 = opset10::Constant::create(element::f32, Shape{2}, {512, 512});
        auto div_1 = std::make_shared<opset10::Divide>(const_2, convert_2);
        auto const_3 = opset10::Constant::create(element::f32, Shape{2}, {1, 1});
        auto concat = std::make_shared<opset10::Concat>(OutputVector{const_3, div_1}, 0);  // scales

        auto mul_1 = std::make_shared<opset10::Multiply>(convert_1, concat);
        auto convert_3 = std::make_shared<opset10::Convert>(mul_1, element::i64);  // sizes

        opset10::Interpolate::InterpolateAttrs attrs;
        attrs.mode = opset10::Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = opset10::Interpolate::ShapeCalcMode::SIZES;
        attrs.nearest_mode = opset10::Interpolate::NearestMode::FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = opset10::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        auto interpolate = std::make_shared<opset10::Interpolate>(input_1, convert_3, concat, attrs);

        auto const_4 = opset10::Constant::create(element::f32, Shape{}, {0.1f});
        auto add_1 = std::make_shared<opset10::Add>(input_1, const_4);
        auto result_1 = std::make_shared<opset10::Result>(add_1);
        auto result_2 = std::make_shared<opset10::Result>(interpolate);
        model = std::make_shared<Model>(NodeVector{result_1, result_2}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 720, 1280});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 720, 1280});
        auto shapeof_1 = std::make_shared<opset10::ShapeOf>(input_1);
        auto shapeof_2 = std::make_shared<opset10::ShapeOf>(input_2);

        auto const_1 = opset10::Constant::create(element::i64, Shape{2}, {2, 3});
        auto axis_const = opset10::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset10::Gather>(shapeof_2, const_1, axis_const);
        auto convert_1 = std::make_shared<opset10::Convert>(shapeof_1, element::f32);

        auto convert_2 = std::make_shared<opset10::Convert>(gather_1, element::f32);
        auto const_2 = opset10::Constant::create(element::f32, Shape{2}, {512, 512});
        auto div_1 = std::make_shared<opset10::Divide>(const_2, convert_2);
        auto const_3 = opset10::Constant::create(element::f32, Shape{2}, {1, 1});
        auto concat = std::make_shared<opset10::Concat>(OutputVector{const_3, div_1}, 0);  // scales

        auto mul_1 = std::make_shared<opset10::Multiply>(convert_1, concat);
        auto convert_3 = std::make_shared<opset10::Convert>(mul_1, element::i64);  // sizes

        opset10::Interpolate::InterpolateAttrs attrs;
        attrs.mode = opset10::Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = opset10::Interpolate::ShapeCalcMode::SIZES;
        attrs.nearest_mode = opset10::Interpolate::NearestMode::FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = opset10::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        auto interpolate = std::make_shared<opset10::Interpolate>(input_1, convert_3, concat, attrs);

        auto const_4 = opset10::Constant::create(element::f16, Shape{}, {0.1f});
        auto add_1 = std::make_shared<opset10::Add>(input_1, const_4);
        auto result_1 = std::make_shared<opset10::Result>(add_1);
        auto result_2 = std::make_shared<opset10::Result>(interpolate);
        model_ref = std::make_shared<Model>(NodeVector{result_1, result_2}, ParameterVector{input_1, input_2});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, ConvertCompressedToMixedPrecission_do_not_keep_in_fp32) {
    // negative test: check that without keeping sensitive nodes in FP32 the whole Model is converted to f16
    // including ShapeOf subgraph and we get wrong output shape [1, 3, 287, 511] instead of correct one [1, 3, 288, 512]
    std::shared_ptr<Model> model(nullptr);
    std::shared_ptr<opset10::Interpolate> interpolate(nullptr);
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto sizes = opset10::Constant::create(element::i64, Shape{4}, {1, 3, 288, 512});
        auto scales_const = opset10::Constant::create(element::f32, Shape{4}, {1.0f, 1.0f, 0.4f, 0.4f});

        opset10::Interpolate::InterpolateAttrs attrs;
        attrs.mode = opset10::Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = opset10::Interpolate::ShapeCalcMode::SCALES;
        attrs.nearest_mode = opset10::Interpolate::NearestMode::FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = opset10::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        interpolate = std::make_shared<opset10::Interpolate>(input, sizes, scales_const, attrs);
        model = std::make_shared<Model>(NodeVector{interpolate}, ParameterVector{input});

        pass::Manager manager;
        manager.register_pass<pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = false;  // didn't keep in FP32 intentionally
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    ASSERT_FALSE(has_type<element::Type_t::f32>(model));
    ASSERT_TRUE(interpolate->input_value(2).get_element_type() == element::Type_t::f16);
    ASSERT_TRUE(interpolate->output(0).get_partial_shape() == PartialShape({1, 3, 287, 511}));
}

template <typename From, typename To>
void constant_convert_test(element::Type type_from,
                           element::Type type_to,
                           const std::vector<From>& value,
                           const std::vector<To>& expected) {
    std::shared_ptr<Model> f(nullptr);
    std::string expected_friendly_name;
    size_t size = value.size() * sizeof(From) * 8 / type_from.bitwidth();
    {
        auto c = std::make_shared<opset4::Constant>(type_from, Shape{size}, value.data());
        expected_friendly_name = c->get_friendly_name();
        f = std::make_shared<Model>(NodeVector{c}, ParameterVector{});

        pass::Manager manager;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{type_from, type_to}});
        manager.run_passes(f);
    }
    auto ops = f->get_ordered_ops();
    auto c = std::dynamic_pointer_cast<opset4::Constant>(ops[0]);
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(c->get_friendly_name(), expected_friendly_name);
    std::vector<To> actual;
    try {
        actual = c->cast_vector<To>();
    } catch (...) {
        size_t dst_size = (type_to.bitwidth() * size + 7) / 8;
        actual.assign(c->get_data_ptr<uint8_t>(), c->get_data_ptr<uint8_t>() + dst_size);
    }
    ASSERT_TRUE(actual.size() >= expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        ASSERT_EQ(expected[i], actual[i]);
    }
}

template <typename From, typename To>
void constant_convert_test(element::Type_t type_from, element::Type_t type_to, From value, To expected) {
    std::shared_ptr<Model> f(nullptr);
    std::string expected_friendly_name;
    {
        auto c = std::make_shared<opset4::Constant>(type_from, Shape{}, &value);
        expected_friendly_name = c->get_friendly_name();
        f = std::make_shared<Model>(NodeVector{c}, ParameterVector{});

        pass::Manager manager;
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{type_from, type_to}});
        manager.run_passes(f);
    }
    auto ops = f->get_ordered_ops();
    auto c = std::dynamic_pointer_cast<opset4::Constant>(ops[0]);
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(c->get_friendly_name(), expected_friendly_name);

    auto actual = c->cast_vector<To>()[0];
    ASSERT_EQ(expected, actual);
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I64MinToI32) {
    constant_convert_test(element::Type_t::i64,
                          element::Type_t::i32,
                          std::numeric_limits<int64_t>::min(),
                          std::numeric_limits<int32_t>::min());
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I64MaxToI32) {
    constant_convert_test(element::Type_t::i64,
                          element::Type_t::i32,
                          std::numeric_limits<int64_t>::max(),
                          std::numeric_limits<int32_t>::max());
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U64MinToI32) {
    constant_convert_test(element::Type_t::u64, element::Type_t::i32, std::numeric_limits<uint64_t>::min(), 0);
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U64MaxToI32) {
    constant_convert_test(element::Type_t::u64,
                          element::Type_t::i32,
                          std::numeric_limits<uint64_t>::max(),
                          std::numeric_limits<int32_t>::max());
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U64ToI32) {
    constant_convert_test<uint64_t, int32_t>(element::Type_t::u64, element::Type_t::i32, 42, 42);
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U32MinToI32) {
    constant_convert_test(element::Type_t::u32, element::Type_t::i32, std::numeric_limits<uint32_t>::min(), 0);
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U32MaxToI32) {
    constant_convert_test(element::Type_t::u32,
                          element::Type_t::i32,
                          std::numeric_limits<uint32_t>::max(),
                          std::numeric_limits<int32_t>::max());
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U32ToI32) {
    constant_convert_test(element::Type_t::u32, element::Type_t::i32, 42, 42);
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_BoolToU8) {
    constant_convert_test(element::Type_t::boolean, element::Type_t::u8, true, 1);
    constant_convert_test(element::Type_t::boolean, element::Type_t::u8, false, 0);
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToI8) {
    constant_convert_test<uint8_t, int8_t>(element::u4, element::i8, std::vector<uint8_t>{171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToU8) {
    constant_convert_test<uint8_t, uint8_t>(element::u4, element::u8, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToI8_2) {
    constant_convert_test<uint8_t, int8_t>(element::u4, element::i8, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToU8_96) {
    constant_convert_test<uint8_t, uint8_t>(element::u4, element::u8, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU8) {
    constant_convert_test<uint8_t, uint8_t>(element::i4, element::u8, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI8) {
    constant_convert_test<uint8_t, int8_t>(element::i4, element::i8, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU8_neg) {
    constant_convert_test<uint8_t, uint8_t>(element::i4, element::u8, {171}, {250, 251});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI8_neg) {
    constant_convert_test<uint8_t, int8_t>(element::i4, element::i8, {171}, {-6, -5});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToI32) {
    constant_convert_test<uint8_t, int32_t>(element::u4, element::i32, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToU32) {
    constant_convert_test<uint8_t, uint32_t>(element::u4, element::u32, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU32) {
    constant_convert_test<uint8_t, uint32_t>(element::i4, element::u32, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI32) {
    constant_convert_test<uint8_t, int32_t>(element::i4, element::i32, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU32_neg) {
    constant_convert_test<uint8_t, uint32_t>(element::i4, element::u32, {171}, {4294967290, 4294967291});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI32_neg) {
    constant_convert_test<uint8_t, int32_t>(element::i4, element::i32, {171}, {-6, -5});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToI16) {
    constant_convert_test<uint8_t, int16_t>(element::u4, element::i16, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToU16) {
    constant_convert_test<uint8_t, uint16_t>(element::u4, element::u16, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU16) {
    constant_convert_test<uint8_t, uint16_t>(element::i4, element::u16, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI16) {
    constant_convert_test<uint8_t, int16_t>(element::i4, element::i16, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU16_neg) {
    constant_convert_test<uint8_t, uint16_t>(element::i4, element::u16, {171}, {65530, 65531});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI16_neg) {
    constant_convert_test<uint8_t, int16_t>(element::i4, element::i16, {171}, {-6, -5});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToI64) {
    constant_convert_test<uint8_t, int64_t>(element::u4, element::i64, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToU64) {
    constant_convert_test<uint8_t, int64_t>(element::u4, element::u64, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU64) {
    constant_convert_test<uint8_t, uint64_t>(element::i4, element::u64, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI64) {
    constant_convert_test<uint8_t, int64_t>(element::i4, element::i64, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU64_neg) {
    constant_convert_test<uint8_t, uint64_t>(element::i4,
                                             element::u64,
                                             {171},
                                             {18446744073709551610u, 18446744073709551611u});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI64_neg) {
    constant_convert_test<uint8_t, int64_t>(element::i4, element::i64, {171}, {-6, -5});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U1ToU8) {
    constant_convert_test<uint8_t, uint8_t>(element::u1, element::u8, {171}, {1, 0, 1, 0, 1, 0, 1, 1});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U1ToU4) {
    constant_convert_test<uint8_t, uint8_t>(element::u1,
                                            element::u4,
                                            std::vector<uint8_t>{171},
                                            {1, 0, 1, 0, 1, 0, 1, 1});
}
