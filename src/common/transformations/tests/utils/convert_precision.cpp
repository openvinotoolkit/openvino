// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_precision.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset15.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/common_optimizations/disable_shapeof_constant_folding.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/original_precision_attribute.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;
using namespace std;

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

        static const precisions_map precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

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

        static const precisions_map precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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
    static const precisions_map precisions = {{element::i64, element::i32}, {element::f32, element::f16}};
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertPrecision>(precisions);
    manager.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));
    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<element::Type_t::f32>(f));
}

TEST(TransformationTests, DoubleConvertPrecision_NMS5) {
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
    static const precisions_map precisions1 = {{element::f32, element::f16}};
    static const precisions_map precisions2 = {{element::i64, element::i32}};
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertPrecision>(precisions1);
    manager.register_pass<pass::ConvertPrecision>(precisions2);
    manager.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));
    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<element::Type_t::f32>(f));
}

TEST(TransformationTests, DoubleConvertPrecision_NMS9) {
    std::shared_ptr<Model> f;
    {
        auto boxes = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset9::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset9::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset9::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset9::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset9::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        auto result1 = std::make_shared<opset9::Result>(nms->output(0));
        auto result2 = std::make_shared<opset9::Result>(nms->output(1));
        auto result3 = std::make_shared<opset9::Result>(nms->output(2));
        f = std::make_shared<Model>(ResultVector{result1, result2, result3}, ParameterVector{boxes, scores});
    }

    pass::Manager manager;
    static const precisions_map precisions1 = {{element::f32, element::f16}};
    static const precisions_map precisions2 = {{element::i64, element::i32}};
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertPrecision>(precisions1);
    manager.register_pass<pass::ConvertPrecision>(precisions2);
    manager.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));
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
    static const precisions_map precisions = {{element::i64, element::i32}, {element::f16, element::f32}};
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertPrecision>(precisions);
    manager.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));
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
    static const precisions_map precisions = {{element::i64, element::i32}, {element::f16, element::f32}};
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertPrecision>(precisions);
    manager.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
    ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_Convert_clamp_1) {
    //  Similar to const compression test CompressConstants_compress_to_f16_max_out_of_range_val
    // fp16 out of range should be clamped to [fp16_min, fp16_max]
    std::shared_ptr<Model> model(nullptr), model_ref(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 2});
        auto const_node = opset10::Constant::create(element::f32, Shape{2}, {100000.0f, -100000.0f});
        auto convert = std::make_shared<opset4::Convert>(const_node, element::f16);
        auto add_1 = make_shared<opset10::Add>(input, convert);
        model = std::make_shared<Model>(NodeVector{add_1}, ParameterVector{input});

        pass::Manager manager;
        static const precisions_map precisions = {{element::f32, element::f16}};
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(model);
    }

    {
        auto max_fp16 = static_cast<float>(std::numeric_limits<ov::float16>::max());
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 2});
        auto const_node = opset10::Constant::create(element::f16, Shape{2}, {max_fp16, -max_fp16});
        auto add_1 = make_shared<opset10::Add>(input, const_node);

        model_ref = std::make_shared<Model>(NodeVector{add_1}, ParameterVector{input});
    }
    ASSERT_NO_THROW(check_rt_info(model));
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES)
                        .enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
    const auto res = fc.compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, ConvertPrecision_Convert_clamp_bf16_f16) {
    // fp16 out of range should be clamped to [fp16_min, fp16_max]
    std::shared_ptr<Model> model(nullptr), model_ref(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 3});
        auto const_node = opset10::Constant::create(element::bf16, Shape{3}, {100000.0f, -100000.0f, 10.0f});
        auto convert = std::make_shared<opset4::Convert>(const_node, element::f16);
        auto add_1 = make_shared<opset10::Add>(input, convert);
        model = std::make_shared<Model>(NodeVector{add_1}, ParameterVector{input});

        pass::Manager manager;
        static const precisions_map precisions = {{element::bf16, element::f16}};
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(model);
    }

    {
        auto max_fp16 = static_cast<float>(std::numeric_limits<ov::float16>::max());
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 3});
        auto const_node = opset10::Constant::create(element::f16, Shape{3}, {max_fp16, -max_fp16, 10.0f});
        auto add_1 = make_shared<opset10::Add>(input, const_node);

        model_ref = std::make_shared<Model>(NodeVector{add_1}, ParameterVector{input});
    }
    ASSERT_NO_THROW(check_rt_info(model));
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES)
                        .enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
    const auto res = fc.compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
TEST(TransformationTests, ConvertPrecision_Convert_clamp_2) {
#else
// Ticket: CVS-122397
TEST(TransformationTests, DISABLED_ConvertPrecision_Convert_clamp_2) {
#endif
    //  Similar to const compression test CompressConstants_compress_to_f16_max_out_of_range_val
    // fp16 out of range should be clamped to [fp16_min, fp16_max]
    std::shared_ptr<Model> model(nullptr), model_ref(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1000, 2});
        auto const_node_1 = opset10::Constant::create(element::i32, Shape{2}, {100000, -100000});
        auto convert_f32 = std::make_shared<opset4::Convert>(const_node_1, element::f32);
        auto const_node_2 = opset10::Constant::create(element::f32, Shape{1}, {1.0f});

        auto add_1 = make_shared<opset10::Add>(convert_f32, const_node_2);
        auto add_2 = make_shared<opset10::Add>(input, add_1);
        model = std::make_shared<Model>(NodeVector{add_2}, ParameterVector{input});

        pass::Manager manager;
        static const precisions_map precisions = {{element::f32, element::f16}};
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.register_pass<pass::ConstantFolding>();
        manager.run_passes(model);
    }

    {
        auto max_fp16 = static_cast<float>(std::numeric_limits<ov::float16>::max());
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 2});
        auto const_node = opset10::Constant::create(element::f16, Shape{2}, {max_fp16, -max_fp16});
        auto add_1 = make_shared<opset10::Add>(input, const_node);

        model_ref = std::make_shared<Model>(NodeVector{add_1}, ParameterVector{input});
    }

    ASSERT_NO_THROW(check_rt_info(model));
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES)
                        .enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
    const auto res = fc.compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
TEST(TransformationTests, ConvertPrecision_Convert_clamp_int32) {
#else
// Ticket: CVS-122397
TEST(TransformationTests, DISABLED_ConvertPrecision_Convert_clamp_int32) {
#endif
    // int32 values will be converted to float16, but during CF evaluate is calculated in float32
    // const_1[i32] -> convert_to_f16[f16] -> some_foldable_op[f16] -> ...
    // cont_1_converted_to_f16[f16] -> some_foldable_op[f16] -> ...
    // but during CF the subgraph above is evaluated in f32 and then again is cast to f16.
    // therefore we should ensure that clamp still takes place if in intermediate calculation overflow happens

    std::shared_ptr<Model> model(nullptr), model_ref(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1000, 2});
        auto const_node_1 = opset10::Constant::create(element::i32, Shape{2}, {100000, -100000});
        auto convert_f32 = std::make_shared<opset4::Convert>(const_node_1, element::f32);
        auto const_node_2 = opset10::Constant::create(element::f32, Shape{1}, {1.0f});

        auto add_1 = make_shared<opset10::Add>(convert_f32, const_node_2);
        auto add_2 = make_shared<opset10::Add>(input, add_1);
        model = std::make_shared<Model>(NodeVector{add_2}, ParameterVector{input});

        pass::Manager manager;
        static const precisions_map precisions = {{element::f32, element::f16}};
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.register_pass<pass::ConstantFolding>();
        manager.run_passes(model);
    }

    {
        auto max_fp16 = static_cast<float>(std::numeric_limits<ov::float16>::max());
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 2});
        auto const_node = opset10::Constant::create(element::f16, Shape{2}, {max_fp16, -max_fp16});
        auto add_1 = make_shared<opset10::Add>(input, const_node);

        model_ref = std::make_shared<Model>(NodeVector{add_1}, ParameterVector{input});
    }

    ASSERT_NO_THROW(check_rt_info(model));
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES)
                        .enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
    const auto res = fc.compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, ConvertPrecision_ConvertElimination) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 4});
        auto relu = std::make_shared<opset4::Relu>(input);
        auto convert = std::make_shared<opset4::Convert>(relu, element::f32);

        f = std::make_shared<Model>(NodeVector{convert}, ParameterVector{input});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f16, element::f32}});
        manager.run_passes(f);
        ASSERT_FALSE(has_type<element::Type_t::f16>(f));
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1000, 4});
        auto relu = std::make_shared<opset4::Relu>(input);

        f_ref = std::make_shared<Model>(NodeVector{relu}, ParameterVector{input});
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ConvertPrecision>(precisions);
        manager.run_passes(model);
    }
    OV_ASSERT_NO_THROW(check_rt_info(model));
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

        static const precisions_map precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);

        auto casted_end = ov::as_type_ptr<opset1::Constant>(ss->input_value(2).get_node_shared_ptr());
        ASSERT_TRUE(casted_end != nullptr);
        ASSERT_EQ(casted_end->get_element_type(), element::i32);
        ASSERT_EQ(casted_end->cast_vector<int32_t>(),
                  std::vector<int32_t>({max_int32, max_int32, max_int32, max_int32}));
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::boolean, element::u8}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::boolean, element::u8}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::boolean, element::u8}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::boolean, element::u8}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::boolean, element::u8}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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

        static const precisions_map precisions = {{element::boolean, element::u8}, {element::f16, element::f32}};

        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::boolean, element::u8}});
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::boolean, element::u8}});
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::boolean, element::u8}});
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::boolean, element::u8}});
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<element::Type_t::u8>(f));

    std::shared_ptr<ov::op::TypeRelaxedBase> tr;
    for (const auto& node : f->get_ordered_ops())
        if (auto op = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(node))
            tr = op;
    ASSERT_TRUE(tr != nullptr);
    ASSERT_EQ(tr->get_origin_input_type(0), element::boolean);
    ASSERT_EQ(tr->get_origin_input_type(1), element::undefined);
}

TEST(TransformationTests, ConvertPrecision_Select) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::boolean, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::LogicalNot>(input1);
        auto select = std::make_shared<opset4::Select>(node, input1, input1);

        f = std::make_shared<Model>(OutputVector{select}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::boolean, element::u8}});
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
    ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_Select_Relaxed) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::boolean, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::LogicalNot>(input1);
        auto select = std::make_shared<opset4::Select>(node, input1, input1);

        f = std::make_shared<Model>(OutputVector{select}, ParameterVector{input1});

        // Explicitly setting the element type of a node to a different one to
        // test the appearance of TypeRelaxed within Select
        node->set_output_type(0, ov::element::u8, node->get_output_partial_shape(0));

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::u8, element::boolean}});
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
    ASSERT_FALSE(has_type<element::Type_t::u8>(f));
    ASSERT_TRUE(has_type<element::Type_t::boolean>(f));
    int counter = 0;
    for (const auto& node : f->get_ordered_ops())
        if (std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(node))
            ++counter;
    ASSERT_EQ(counter, 1);
}

TEST(TransformationTests, ConvertPrecision_TypeRelaxedWithSelect) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset4::Parameter>(element::boolean, Shape{15, 20, 3});
        auto node = std::make_shared<opset4::LogicalNot>(input1);
        auto select = std::make_shared<opset4::Select>(node, input1, input1);

        f = std::make_shared<Model>(OutputVector{select}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::boolean, element::i32}});
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::i32, element::i64}});
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::boolean, element::i32}});
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::i32, element::i64}});
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_FALSE(has_type<element::Type_t::boolean>(f));
        ASSERT_FALSE(has_type<element::Type_t::i32>(f));
        ASSERT_TRUE(has_type<element::Type_t::i64>(f));
    }
}

TEST(TransformationTests, ConvertPrecision_SearchSorted) {
    std::shared_ptr<Model> f(nullptr);
    {
        auto search_sorted_input = opset15::Constant::create(ov::element::i64, {5}, {1, 2, 3, 4, 5});
        auto indices = std::make_shared<opset15::Parameter>(ov::element::i64, Shape{3});
        auto search_sorted = std::make_shared<opset15::SearchSorted>(search_sorted_input, indices);

        auto less_input = opset15::Constant::create(ov::element::i64, {3}, {4, 5, 6});
        auto less = std::make_shared<opset15::Less>(search_sorted, less_input);

        f = std::make_shared<Model>(OutputVector{less}, ParameterVector{indices});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::i64, element::i32}});
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
    ASSERT_FALSE(has_type<element::Type_t::i64>(f));
    ASSERT_TRUE(has_type<element::Type_t::i32>(f));
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
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f16, element::f32}});
        manager.run_passes(f);
    }
    OV_ASSERT_NO_THROW(check_rt_info(f));
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
        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }
    OV_ASSERT_NO_THROW(check_rt_info(model));
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
        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = false;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }
    OV_ASSERT_NO_THROW(check_rt_info(model));
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
        manager.register_pass<pass::DisableShapeOfConstantFolding>();

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
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
                        .enable(FunctionsComparator::CONST_VALUES)
                        .enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
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
        manager.register_pass<pass::DisableShapeOfConstantFolding>();

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
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
                        .enable(FunctionsComparator::CONST_VALUES)
                        .enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
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
        manager.register_pass<pass::DisableShapeOfConstantFolding>();

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
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
                        .enable(FunctionsComparator::CONST_VALUES)
                        .enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
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
        manager.register_pass<pass::DisableShapeOfConstantFolding>();

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
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
                        .enable(FunctionsComparator::CONST_VALUES)
                        .enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
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
        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = false;  // didn't keep in FP32 intentionally
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }
    OV_ASSERT_NO_THROW(check_rt_info(model));
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
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{type_from, type_to}});
        manager.run_passes(f);
    }
    auto ops = f->get_ordered_ops();
    auto c = ov::as_type_ptr<opset4::Constant>(ops[0]);
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
        EXPECT_EQ(expected[i], actual[i]) << "Elements with index " << i << " are not equal.";
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
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{type_from, type_to}});
        manager.run_passes(f);
    }
    auto ops = f->get_ordered_ops();
    auto c = ov::as_type_ptr<opset4::Constant>(ops[0]);
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
                                            {0, 1, 0, 1, 0, 1, 1, 1});
}

TEST(TransformationTests, ConvertPrecision_keep_precission_sensitive_fp32_with_exp) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset10::Exp>(input_1);
        auto input_2 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset10::ReduceSum>(exp_1, reduction_axes);

        auto factor_const = opset10::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset10::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset10::Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<opset10::MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto input_1_decompressed = make_shared<opset10::Convert>(input_1, element::f32);
        auto exp_1 = make_shared<opset10::Exp>(input_1_decompressed);
        auto input_2 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset10::ReduceSum>(exp_1, reduction_axes);

        auto factor_const = opset10::Constant::create(element::f32, Shape{1}, {-1});
        auto mul_1 = make_shared<opset10::Multiply>(reduce_sum_1, factor_const);
        auto mul_1_compressed = make_shared<opset10::Convert>(mul_1, element::f16);
        auto matmul_1 = make_shared<opset10::MatMul>(mul_1_compressed, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecision_keep_precission_sensitive_fp32_with_reducemean) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset10::Exp>(input_1);
        auto input_2 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset10::ReduceSum>(exp_1, reduction_axes);

        auto factor_const = opset10::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset10::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset10::Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<opset10::MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto input_1_decompressed = make_shared<opset10::Convert>(input_1, element::f32);
        auto exp_1 = make_shared<opset10::Exp>(input_1_decompressed);
        auto input_2 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_mean_1 = make_shared<opset10::ReduceMean>(exp_1, reduction_axes);

        auto factor_const = opset10::Constant::create(element::f32, Shape{1}, {-1});
        auto mul_1 = make_shared<opset10::Multiply>(reduce_mean_1, factor_const);
        auto mul_1_compressed = make_shared<opset10::Convert>(mul_1, element::f16);
        auto matmul_1 = make_shared<opset10::MatMul>(mul_1_compressed, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecision_reducesum_without_exp) {
    // ReduceSum without Exp is not a precision sensitive case, the whole Model should be cast into f16,
    // no nodes should be marked and no Converts should be added
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    {
        auto input_1 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset10::ReduceSum>(input_1, reduction_axes);

        auto factor_const = opset10::Constant::create(element::f32, Shape{1}, {-1});
        auto mul_1 = make_shared<opset10::Multiply>(reduce_sum_1, factor_const);
        auto matmul_1 = make_shared<opset10::MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset10::ReduceSum>(input_1, reduction_axes);

        auto factor_const = opset10::Constant::create(element::f16, Shape{1}, {-1});
        auto mul_1 = make_shared<opset10::Multiply>(reduce_sum_1, factor_const);
        auto matmul_1 = make_shared<opset10::MatMul>(mul_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecision_keep_precission_sensitive_fp32_t2t_subgraph) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    // subgraph from t2t-vit-7
    {
        auto input_1 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3136, 32});
        auto input_2 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3136, 32});
        auto input_3 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3136, 64, 1});
        auto input_4 = make_shared<opset10::Parameter>(element::f32, Shape{128, 64});
        auto exp_1 = make_shared<opset10::Exp>(input_1);
        auto exp_2 = make_shared<opset10::Exp>(input_2);

        auto factor_1 = opset10::Constant::create(element::f32, Shape{1}, {0.5});  // add decompression
        auto mul_1 = make_shared<opset10::Multiply>(exp_1, factor_1);
        auto factor_2 = opset10::Constant::create(element::f32, Shape{1}, {0.5});
        auto mul_2 = make_shared<opset10::Multiply>(exp_2, factor_2);

        auto const_unsqueeze_1 = opset10::Constant::create(element::i64, Shape{4}, {1, 3136, 1, 32});
        auto unsqueeze_1 = make_shared<opset10::Reshape>(mul_1, const_unsqueeze_1, false);

        auto const_unsqueeze_2 = opset10::Constant::create(element::i64, Shape{4}, {1, 3136, 1, 32});
        auto unsqueeze_2 = make_shared<opset10::Reshape>(mul_2, const_unsqueeze_1, false);
        auto reduction_axes_1 = opset10::Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum_1 = make_shared<opset10::ReduceSum>(mul_2, reduction_axes_1, true);
        auto mul_3 = make_shared<opset10::Multiply>(reduce_sum_1, mul_1);
        auto mul_4 = make_shared<opset10::Multiply>(input_3, unsqueeze_2);

        auto reduction_axes_2 = opset10::Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum_2 = make_shared<opset10::ReduceSum>(mul_4, reduction_axes_2);
        auto reduction_axes_3 = opset10::Constant::create(element::i64, Shape{1}, {2});
        auto reduce_sum_3 = make_shared<opset10::ReduceSum>(mul_3, reduction_axes_3, true);

        auto broadcast_to_shape = opset10::Constant::create(element::i64, Shape{3}, {1, 1, 1});
        auto broadcast =
            make_shared<opset10::Broadcast>(reduce_sum_3, broadcast_to_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto tile_shape = opset10::Constant::create(element::i64, Shape{3}, {1, 1, 64});
        auto tile = make_shared<opset10::Tile>(broadcast, tile_shape);
        auto eps_const = opset10::Constant::create(element::f32, Shape{1}, {1.e-10});
        auto add_1 = make_shared<opset10::Add>(tile, eps_const);

        auto const_unsqueeze_3 = opset10::Constant::create(element::i64, Shape{4}, {1, 1, 64, 32});
        auto unsqueeze_3 = make_shared<opset10::Reshape>(reduce_sum_2, const_unsqueeze_3, false);
        auto mul_5 = make_shared<opset10::Multiply>(unsqueeze_1, unsqueeze_3);

        auto reduction_axes_4 = opset10::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_4 = make_shared<opset10::ReduceSum>(mul_5, reduction_axes_4);

        auto div_1 = make_shared<opset10::Divide>(reduce_sum_4, add_1);
        auto matmul_1 = make_shared<opset10::MatMul>(div_1, input_4, false, true);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2, input_3, input_4});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3136, 32});
        auto input_2 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3136, 32});
        auto input_3 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3136, 64, 1});
        auto input_4 = make_shared<opset10::Parameter>(element::f16, Shape{128, 64});
        auto input_1_decompressed = make_shared<opset10::Convert>(input_1, element::f32);
        auto input_2_decompressed = make_shared<opset10::Convert>(input_2, element::f32);
        auto input_3_decompressed = make_shared<opset10::Convert>(input_3, element::f32);

        auto exp_1 = make_shared<opset10::Exp>(input_1_decompressed);
        auto exp_2 = make_shared<opset10::Exp>(input_2_decompressed);

        auto factor_1 = opset10::Constant::create(element::f32, Shape{1}, {0.5});
        auto mul_1 = make_shared<opset10::Multiply>(exp_1, factor_1);
        auto factor_2 = opset10::Constant::create(element::f32, Shape{1}, {0.5});
        auto mul_2 = make_shared<opset10::Multiply>(exp_2, factor_2);

        auto const_unsqueeze_1 = opset10::Constant::create(element::i64, Shape{4}, {1, 3136, 1, 32});
        auto unsqueeze_1 = make_shared<opset10::Reshape>(mul_1, const_unsqueeze_1, false);

        auto const_unsqueeze_2 = opset10::Constant::create(element::i64, Shape{4}, {1, 3136, 1, 32});
        auto unsqueeze_2 = make_shared<opset10::Reshape>(mul_2, const_unsqueeze_2, false);
        auto reduction_axes_1 = opset10::Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum_1 = make_shared<opset10::ReduceSum>(mul_2, reduction_axes_1, true);
        auto mul_3 = make_shared<opset10::Multiply>(reduce_sum_1, mul_1);
        auto mul_4 = make_shared<opset10::Multiply>(input_3_decompressed, unsqueeze_2);

        auto reduction_axes_2 = opset10::Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum_2 = make_shared<opset10::ReduceSum>(mul_4, reduction_axes_2);
        auto reduction_axes_3 = opset10::Constant::create(element::i64, Shape{1}, {2});
        auto reduce_sum_3 = make_shared<opset10::ReduceSum>(mul_3, reduction_axes_3, true);

        auto broadcast_to_shape = opset10::Constant::create(element::i64, Shape{3}, {1, 1, 1});
        auto broadcast =
            make_shared<opset10::Broadcast>(reduce_sum_3, broadcast_to_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto tile_shape = opset10::Constant::create(element::i64, Shape{3}, {1, 1, 64});
        auto tile = make_shared<opset10::Tile>(broadcast, tile_shape);
        auto eps_const = opset10::Constant::create(element::f32, Shape{1}, {1.e-10});
        auto add_1 = make_shared<opset10::Add>(tile, eps_const);

        auto const_unsqueeze_3 = opset10::Constant::create(element::i64, Shape{4}, {1, 1, 64, 32});
        auto unsqueeze_3 = make_shared<opset10::Reshape>(reduce_sum_2, const_unsqueeze_3, false);
        auto mul_5 = make_shared<opset10::Multiply>(unsqueeze_1, unsqueeze_3);

        auto reduction_axes_4 = opset10::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_4 = make_shared<opset10::ReduceSum>(mul_5, reduction_axes_4);

        auto div_1 = make_shared<opset10::Divide>(reduce_sum_4, add_1);
        auto div_compressed = make_shared<opset10::Convert>(div_1, element::f16);
        auto matmul_1 = make_shared<opset10::MatMul>(div_compressed, input_4, false, true);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2, input_3, input_4});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecision_DivisionByZeroMinimalPattern) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    const float eps_value = 1.0e-12f;
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset10::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset10::Add>(input_2, eps_const);
        auto divide = std::make_shared<opset10::Divide>(input_1, add);
        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f16, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset10::Parameter>(element::f16, PartialShape::dynamic(3));
        auto input_1_decompressed = make_shared<opset10::Convert>(input_1, element::f32);
        auto input_2_decompressed = make_shared<opset10::Convert>(input_2, element::f32);

        auto eps_const = opset10::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset10::Add>(input_2_decompressed, eps_const);
        auto divide = std::make_shared<opset10::Divide>(input_1_decompressed, add);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecision_PowWithNegativeExponent) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    const float eps_value = 1.0e-12f;
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset10::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset10::Add>(input_2, eps_const);
        auto pow_exp_const = opset10::Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<opset10::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset10::Multiply>(input_1, pow);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f16, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset10::Parameter>(element::f16, PartialShape::dynamic(3));
        auto input_1_decompressed = make_shared<opset10::Convert>(input_1, element::f32);
        auto input_2_decompressed = make_shared<opset10::Convert>(input_2, element::f32);

        auto eps_const = opset10::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset10::Add>(input_2_decompressed, eps_const);
        auto pow_exp_const = opset10::Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<opset10::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset10::Multiply>(input_1_decompressed, pow);

        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecision_exp_through_unsqueeze) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset10::Exp>(input_1);
        auto input_2 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});

        auto unsqueeze_axes = opset10::Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze_1 = make_shared<opset10::Unsqueeze>(exp_1, unsqueeze_axes);
        auto reduce_sum_1 = make_shared<opset10::ReduceSum>(unsqueeze_1, reduction_axes);

        auto factor_const = opset10::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset10::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset10::Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<opset10::MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto input_1_decompressed = make_shared<opset10::Convert>(input_1, element::f32);
        auto exp_1 = make_shared<opset10::Exp>(input_1_decompressed);
        auto input_2 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 224, 224});

        auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});

        auto unsqueeze_axes = opset10::Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze_1 = make_shared<opset10::Unsqueeze>(exp_1, unsqueeze_axes);
        auto reduce_sum_1 = make_shared<opset10::ReduceSum>(unsqueeze_1, reduction_axes);

        auto factor_const = opset10::Constant::create(element::f32, Shape{1}, {-1});
        auto mul_1 = make_shared<opset10::Multiply>(reduce_sum_1, factor_const);
        auto mul_1_compressed = make_shared<opset10::Convert>(mul_1, element::f16);
        auto matmul_1 = make_shared<opset10::MatMul>(mul_1_compressed, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecision_disable_for_quantized_nodes_1) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset10::Exp>(input_1);

        auto in_low = op::v0::Constant::create(element::f32, Shape{}, {0.f});
        auto in_high = op::v0::Constant::create(element::f32, Shape{}, {5.f});
        auto out_low = op::v0::Constant::create(element::f32, Shape{}, {2.f});
        auto out_high = op::v0::Constant::create(element::f32, Shape{}, {4.f});
        auto fq_1 = make_shared<opset10::FakeQuantize>(exp_1, in_low, in_high, out_low, out_high, 256);

        auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset10::ReduceSum>(fq_1, reduction_axes);

        auto fq_2 = make_shared<opset10::FakeQuantize>(reduce_sum_1, in_low, in_high, out_low, out_high, 256);
        auto matmul_1 = make_shared<opset10::MatMul>(fq_2, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset10::Exp>(input_1);

        auto in_low = op::v0::Constant::create(element::f16, Shape{}, {0.f});
        auto in_high = op::v0::Constant::create(element::f16, Shape{}, {5.f});
        auto out_low = op::v0::Constant::create(element::f16, Shape{}, {2.f});
        auto out_high = op::v0::Constant::create(element::f16, Shape{}, {4.f});
        auto fq_1 = make_shared<opset10::FakeQuantize>(exp_1, in_low, in_high, out_low, out_high, 256);

        auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset10::ReduceSum>(fq_1, reduction_axes);

        auto fq_2 = make_shared<opset10::FakeQuantize>(reduce_sum_1, in_low, in_high, out_low, out_high, 256);
        auto matmul_1 = make_shared<opset10::MatMul>(fq_2, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecision_disable_for_quantized_nodes_2) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset10::Exp>(input_1);

        auto in_low = op::v0::Constant::create(element::f32, Shape{}, {0.f});
        auto in_high = op::v0::Constant::create(element::f32, Shape{}, {5.f});
        auto out_low = op::v0::Constant::create(element::f32, Shape{}, {2.f});
        auto out_high = op::v0::Constant::create(element::f32, Shape{}, {4.f});
        auto fq_1 = make_shared<opset10::FakeQuantize>(exp_1, in_low, in_high, out_low, out_high, 256);

        auto unsqueeze_axes = opset10::Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze_1 = make_shared<opset10::Unsqueeze>(fq_1, unsqueeze_axes);

        auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset10::ReduceSum>(unsqueeze_1, reduction_axes);

        auto fq_2 = make_shared<opset10::FakeQuantize>(reduce_sum_1, in_low, in_high, out_low, out_high, 256);
        auto matmul_1 = make_shared<opset10::MatMul>(fq_2, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset10::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset10::Exp>(input_1);

        auto in_low = op::v0::Constant::create(element::f16, Shape{}, {0.f});
        auto in_high = op::v0::Constant::create(element::f16, Shape{}, {5.f});
        auto out_low = op::v0::Constant::create(element::f16, Shape{}, {2.f});
        auto out_high = op::v0::Constant::create(element::f16, Shape{}, {4.f});
        auto fq_1 = make_shared<opset10::FakeQuantize>(exp_1, in_low, in_high, out_low, out_high, 256);

        auto unsqueeze_axes = opset10::Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze_1 = make_shared<opset10::Unsqueeze>(fq_1, unsqueeze_axes);

        auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset10::ReduceSum>(unsqueeze_1, reduction_axes);

        auto fq_2 = make_shared<opset10::FakeQuantize>(reduce_sum_1, in_low, in_high, out_low, out_high, 256);
        auto matmul_1 = make_shared<opset10::MatMul>(fq_2, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecisionExplicitConvertsForParameterAndResult) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto param_1 = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto sin = make_shared<opset10::Sin>(param_1);
        sin->set_friendly_name("sine");
        sin->get_output_tensor(0).add_names({"sine:0"});
        auto result_sin = make_shared<opset10::Result>(sin);
        model = make_shared<Model>(result_sin, ParameterVector{param_1});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = false;
        bool convert_input_output_precision = false;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f64, element::f32}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32,
                                                      convert_input_output_precision);
        manager.run_passes(model);
    }

    {
        auto param_1 = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto converted_param = make_shared<opset10::Convert>(param_1, element::f32);
        auto sin = make_shared<opset10::Sin>(converted_param);
        auto converted_sin = make_shared<opset10::Convert>(sin, element::f64);
        converted_sin->get_output_tensor(0).add_names({"sine:0"});
        auto result_sin = make_shared<opset10::Result>(converted_sin);
        model_ref = make_shared<Model>(result_sin, ParameterVector{param_1});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;

    const auto& results = model->get_results();
    ASSERT_EQ("sine", results[0]->get_input_node_ptr(0)->get_friendly_name());
}

TEST(TransformationTests, ConvertPrecisionExplicitConvertsMultiParam) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto param_1 = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto convert_1 = make_shared<opset10::Convert>(param_1, element::f32);

        auto param_2 = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto convert_2 = make_shared<opset10::Convert>(param_2, element::i64);

        auto param_3 = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto param_4 = make_shared<opset10::Parameter>(element::i64, Shape{3});

        auto add = make_shared<opset10::Add>(convert_2, param_4);
        auto mul = make_shared<opset10::Multiply>(param_1, param_3);
        auto sin = make_shared<opset10::Sin>(convert_1);

        add->set_friendly_name("add");
        add->get_output_tensor(0).add_names({"add:0"});
        mul->set_friendly_name("mul");
        mul->get_output_tensor(0).add_names({"mul:0"});
        sin->set_friendly_name("sine");
        sin->get_output_tensor(0).add_names({"sine:0"});

        auto result_add = make_shared<opset10::Result>(add);
        auto result_mul = make_shared<opset10::Result>(mul);
        auto result_sin = make_shared<opset10::Result>(sin);

        model = make_shared<Model>(ResultVector{result_add, result_mul, result_sin},
                                   ParameterVector{param_1, param_2, param_3, param_4});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = false;
        bool convert_input_output_precision = false;
        manager.register_pass<pass::ConvertPrecision>(
            precisions_map{{element::f64, element::f32}, {element::i64, element::i32}},
            empty_type_to_fuse_map,
            keep_precision_sensitive_in_fp32,
            convert_input_output_precision);
        manager.run_passes(model);
    }

    {
        auto param_1 = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto convert_1 = make_shared<opset10::Convert>(param_1, element::f32);

        auto param_2 = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto convert_2 = make_shared<opset10::Convert>(param_2, element::i32);

        auto param_3 = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto convert_3 = make_shared<opset10::Convert>(param_3, element::f32);
        auto param_4 = make_shared<opset10::Parameter>(element::i64, Shape{3});
        auto convert_4 = make_shared<opset10::Convert>(param_4, element::i32);

        auto add = make_shared<opset10::Add>(convert_2, convert_4);
        auto converted_add = make_shared<opset10::Convert>(add, element::i64);
        auto convert_1_2 = make_shared<opset10::Convert>(param_1, element::f32);
        auto mul = make_shared<opset10::Multiply>(convert_1_2, convert_3);
        auto converted_mul = make_shared<opset10::Convert>(mul, element::f64);
        auto sin = make_shared<opset10::Sin>(convert_1);

        converted_add->get_output_tensor(0).add_names({"add:0"});
        converted_mul->get_output_tensor(0).add_names({"mul:0"});
        sin->get_output_tensor(0).add_names({"sine:0"});

        auto result_add = make_shared<opset10::Result>(converted_add);
        auto result_mul = make_shared<opset10::Result>(converted_mul);
        auto result_sin = make_shared<opset10::Result>(sin);

        model_ref = make_shared<Model>(ResultVector{result_add, result_mul, result_sin},
                                       ParameterVector{param_1, param_2, param_3, param_4});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;

    const auto& results = model->get_results();
    ASSERT_EQ("add", results[0]->get_input_node_ptr(0)->get_friendly_name());
    ASSERT_EQ("mul", results[1]->get_input_node_ptr(0)->get_friendly_name());
    ASSERT_EQ("sine", results[2]->get_input_node_ptr(0)->get_friendly_name());
}

TEST(TransformationTests, ConvertPrecisionExplicitConvertsSingleNodeMultipleOutputs) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto param_1 = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto axis = opset10::Constant::create(element::i32, Shape{}, {0});
        auto split = make_shared<opset10::Split>(param_1, axis, 3);
        split->set_friendly_name("split");
        split->get_output_tensor(0).add_names({"split:0"});
        split->get_output_tensor(1).add_names({"split:1"});
        split->get_output_tensor(2).add_names({"split:2"});
        OPENVINO_SUPPRESS_DEPRECATED_START
        ov::descriptor::set_ov_tensor_legacy_name(split->get_output_tensor(0), "legacy_split:0");
        ov::descriptor::set_ov_tensor_legacy_name(split->get_output_tensor(1), "legacy_split:1");
        ov::descriptor::set_ov_tensor_legacy_name(split->get_output_tensor(2), "legacy_split:2");
        OPENVINO_SUPPRESS_DEPRECATED_END
        model = make_shared<Model>(split->outputs(), ParameterVector{param_1});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = false;
        bool convert_input_output_precision = false;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f64, element::f32}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32,
                                                      convert_input_output_precision);
        manager.run_passes(model);
    }

    {
        auto param_1 = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto convert_1 = make_shared<opset10::Convert>(param_1, element::f32);
        auto axis = opset10::Constant::create(element::i32, Shape{}, {0});
        auto split = make_shared<opset10::Split>(convert_1, axis, 3);

        auto convert_split_0 = make_shared<opset10::Convert>(split->output(0), element::f64);
        auto convert_split_1 = make_shared<opset10::Convert>(split->output(1), element::f64);
        auto convert_split_2 = make_shared<opset10::Convert>(split->output(2), element::f64);
        convert_split_0->get_output_tensor(0).add_names({"split:0"});
        convert_split_1->get_output_tensor(0).add_names({"split:1"});
        convert_split_2->get_output_tensor(0).add_names({"split:2"});
        model_ref =
            make_shared<Model>(NodeVector{convert_split_0, convert_split_1, convert_split_2}, ParameterVector{param_1});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;

    const auto& results = model->get_results();
    ASSERT_EQ("split.0", results[0]->get_input_node_ptr(0)->get_friendly_name());
    ASSERT_EQ("split.1", results[1]->get_input_node_ptr(0)->get_friendly_name());
    ASSERT_EQ("split.2", results[2]->get_input_node_ptr(0)->get_friendly_name());
    OPENVINO_SUPPRESS_DEPRECATED_START
    ASSERT_EQ("legacy_split:0", ov::descriptor::get_ov_tensor_legacy_name(results[0]->get_input_tensor(0)));
    ASSERT_EQ("legacy_split:1", ov::descriptor::get_ov_tensor_legacy_name(results[1]->get_input_tensor(0)));
    ASSERT_EQ("legacy_split:2", ov::descriptor::get_ov_tensor_legacy_name(results[2]->get_input_tensor(0)));
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST(TransformationTests, ConvertPrecisionExplicitConvertsMultiSubgraphs) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto cond = make_shared<opset10::Parameter>(element::boolean, Shape{});
        auto param_1 = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto param_2 = make_shared<opset10::Parameter>(element::f64, Shape{3});

        auto if_op = make_shared<opset10::If>(cond);

        auto param_1_then = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto param_2_then = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto add = make_shared<opset10::Add>(param_1_then, param_2_then);
        auto result_then = make_shared<opset10::Result>(add);
        auto then_body = make_shared<Model>(result_then, ParameterVector{param_1_then, param_2_then});

        auto param_1_else = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto param_2_else = make_shared<opset10::Parameter>(element::f64, Shape{3});

        auto trip_count = op::v0::Constant::create(element::i32, Shape{}, {2});
        auto term_cond = op::v0::Constant::create(element::boolean, Shape{}, {true});
        auto loop = make_shared<opset10::Loop>(trip_count, term_cond);

        auto param_1_loop = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto param_2_loop = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto mul = make_shared<opset10::Multiply>(param_1_loop, param_2_loop);
        auto result_mul = make_shared<opset10::Result>(mul);
        auto result_cond = make_shared<opset10::Result>(term_cond);
        auto loop_body =
            make_shared<Model>(ResultVector{result_cond, result_mul}, ParameterVector{param_1_loop, param_2_loop});

        loop->set_function(loop_body);
        loop->set_special_body_ports({-1, 0});
        loop->set_merged_input(param_1_loop, param_1_else, result_mul);

        auto result_else = make_shared<opset10::Result>(loop->get_iter_value(result_mul));
        auto else_body = make_shared<Model>(result_else, ParameterVector{param_1_else, param_2_else});

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(param_1, param_1_then, param_1_else);
        if_op->set_input(param_2, param_2_then, param_2_else);
        auto result = if_op->set_output(result_then, result_else);

        result.get_node()->set_friendly_name("if_result");
        result.add_names({"if_result:0"});
        model = make_shared<Model>(OutputVector{result}, ParameterVector{cond, param_1, param_2});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = false;
        bool convert_input_output_precision = false;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f64, element::f32}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32,
                                                      convert_input_output_precision);
        manager.run_passes(model);
    }

    {
        auto cond = make_shared<opset10::Parameter>(element::boolean, Shape{});
        auto param_1 = make_shared<opset10::Parameter>(element::f64, Shape{3});
        auto param_2 = make_shared<opset10::Parameter>(element::f64, Shape{3});

        auto if_op = make_shared<opset10::If>(cond);

        auto param_1_then = make_shared<opset10::Parameter>(element::f32, Shape{3});
        auto param_2_then = make_shared<opset10::Parameter>(element::f32, Shape{3});
        auto add = make_shared<opset10::Add>(param_1_then, param_2_then);
        auto result_then = make_shared<opset10::Result>(add);
        auto then_body = make_shared<Model>(result_then, ParameterVector{param_1_then, param_2_then});

        auto param_1_else = make_shared<opset10::Parameter>(element::f32, Shape{3});
        auto param_2_else = make_shared<opset10::Parameter>(element::f32, Shape{3});

        auto trip_count = op::v0::Constant::create(element::i32, Shape{}, {2});
        auto term_cond = op::v0::Constant::create(element::boolean, Shape{}, {true});
        auto loop = make_shared<opset10::Loop>(trip_count, term_cond);

        auto param_1_loop = make_shared<opset10::Parameter>(element::f32, Shape{3});
        auto param_2_loop = make_shared<opset10::Parameter>(element::f32, Shape{3});
        auto mul = make_shared<opset10::Multiply>(param_1_loop, param_2_loop);
        auto result_mul = make_shared<opset10::Result>(mul);
        auto result_cond = make_shared<opset10::Result>(term_cond);
        auto loop_body =
            make_shared<Model>(ResultVector{result_cond, result_mul}, ParameterVector{param_1_loop, param_2_loop});

        loop->set_function(loop_body);
        loop->set_special_body_ports({-1, 0});
        loop->set_merged_input(param_1_loop, param_1_else, result_mul);

        auto result_else = make_shared<opset10::Result>(loop->get_iter_value(result_mul));
        auto else_body = make_shared<Model>(result_else, ParameterVector{param_1_else, param_2_else});

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        auto convert_1 = make_shared<opset10::Convert>(param_1, element::f32);
        auto convert_2 = make_shared<opset10::Convert>(param_2, element::f32);
        if_op->set_input(convert_1, param_1_then, param_1_else);
        if_op->set_input(convert_2, param_2_then, param_2_else);
        auto result = if_op->set_output(result_then, result_else);
        auto converted_result = make_shared<opset10::Convert>(result, element::f64);
        converted_result->get_output_tensor(0).add_names({"if_result:0"});

        model_ref = make_shared<Model>(converted_result, ParameterVector{cond, param_1, param_2});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;

    const auto& results = model->get_results();
    ASSERT_EQ("if_result", results[0]->get_input_node_ptr(0)->get_friendly_name());
}

TEST(TransformationTests, align_mixed_fp16_fp32_with_parameter_for_shape_1) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto shape_input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});

        auto upscale_const = ov::op::v0::Constant::create(element::f32, Shape{1}, {2.0f});
        auto mul_1 = make_shared<ov::op::v1::Multiply>(shape_input, upscale_const);
        auto axis_const = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto final_float_shape = make_shared<ov::op::v1::ReduceProd>(mul_1, axis_const, true);
        auto final_int_shape = make_shared<ov::op::v0::Convert>(final_float_shape, element::i64);
        auto reshape_1 = make_shared<ov::op::v1::Reshape>(input_1, final_int_shape, false);

        model = make_shared<Model>(NodeVector{reshape_1}, ParameterVector{input_1, shape_input});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<ov::op::v0::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto shape_input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});

        // even for FP16 compressed model shape subgraph should be kept in fp32
        auto upscale_const = ov::op::v0::Constant::create(element::f32, Shape{1}, {2.0f});
        auto mul_1 = make_shared<ov::op::v1::Multiply>(shape_input, upscale_const);
        auto axis_const = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto final_float_shape = make_shared<ov::op::v1::ReduceProd>(mul_1, axis_const, true);
        auto final_int_shape = make_shared<ov::op::v0::Convert>(final_float_shape, element::i64);
        auto reshape_1 = make_shared<ov::op::v1::Reshape>(input_1, final_int_shape, false);

        model_ref = make_shared<Model>(NodeVector{reshape_1}, ParameterVector{input_1, shape_input});
    }
    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, align_mixed_fp16_fp32_with_parameter_for_shape_2) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto shape_input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});

        auto upscale_const = ov::op::v0::Constant::create(element::f32, Shape{1}, {2.0f});
        auto mul_1 = make_shared<ov::op::v1::Multiply>(shape_input, upscale_const);
        auto axis_const = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto final_float_shape = make_shared<ov::op::v1::ReduceProd>(mul_1, axis_const, true);
        auto final_int_shape = make_shared<ov::op::v0::Convert>(final_float_shape, element::i64);
        auto reshape_1 = make_shared<ov::op::v1::Reshape>(input_1, final_int_shape, false);

        model = make_shared<Model>(NodeVector{reshape_1}, ParameterVector{input_1, shape_input});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        const bool convert_input_output_precision = false;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32,
                                                      convert_input_output_precision);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto convert_to_f16 = make_shared<ov::op::v0::Convert>(input_1, element::f16);
        auto shape_input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});

        // even for FP16 compressed model shape subgraph should be kept in fp32
        auto upscale_const = ov::op::v0::Constant::create(element::f32, Shape{1}, {2.0f});
        auto mul_1 = make_shared<ov::op::v1::Multiply>(shape_input, upscale_const);
        auto axis_const = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto final_float_shape = make_shared<ov::op::v1::ReduceProd>(mul_1, axis_const, true);
        auto final_int_shape = make_shared<ov::op::v0::Convert>(final_float_shape, element::i64);
        auto reshape_1 = make_shared<ov::op::v1::Reshape>(convert_to_f16, final_int_shape, false);
        auto convert_to_f32 = make_shared<ov::op::v0::Convert>(reshape_1, element::f32);

        model_ref = make_shared<Model>(NodeVector{convert_to_f32}, ParameterVector{input_1, shape_input});
    }
    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecision_assign_read_value_preserve_orig_types) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{10, 10}, ov::element::f32, "variable_name"});

        auto input = make_shared<opset10::Parameter>(element::f32, Shape{10, 10});
        auto read_value = make_shared<opset10::ReadValue>(input, variable);

        auto some_value = opset10::Constant::create(element::f32, Shape{1}, {2});
        auto mul = make_shared<opset10::Multiply>(read_value, some_value);
        auto res = make_shared<opset10::Result>(mul);
        auto assign = make_shared<opset10::Assign>(mul, variable);

        model = make_shared<Model>(ResultVector{res}, SinkVector{assign}, ParameterVector{input});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        bool convert_input_output_precision = false;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32,
                                                      convert_input_output_precision);
        manager.run_passes(model);
    }

    {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{10, 10}, ov::element::f32, "variable_name"});

        auto input = make_shared<opset10::Parameter>(element::f32, Shape{10, 10});
        auto convert_1 = make_shared<opset10::Convert>(input, element::f16);
        auto convert_2 = make_shared<opset10::Convert>(convert_1, element::f32);

        auto read_value = make_shared<opset10::ReadValue>(convert_2, variable);
        auto convert_3 = make_shared<opset10::Convert>(read_value, element::f16);

        auto some_value = opset10::Constant::create(element::f16, Shape{1}, {2});
        auto mul = make_shared<opset10::Multiply>(convert_3, some_value);
        auto convert_4 = make_shared<opset10::Convert>(mul, element::f32);
        auto res = make_shared<opset10::Result>(convert_4);

        auto convert_5 = make_shared<opset10::Convert>(mul, element::f32);
        auto assign = make_shared<opset10::Assign>(convert_5, variable);

        model_ref = make_shared<Model>(ResultVector{res}, SinkVector{assign}, ParameterVector{input});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecision_assign_read_value_change_variable_type) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{10, 10}, ov::element::f32, "variable_name"});

        auto input = make_shared<opset10::Parameter>(element::f32, Shape{10, 10});
        auto read_value = make_shared<opset10::ReadValue>(input, variable);

        auto some_value = opset10::Constant::create(element::f32, Shape{1}, {2});
        auto mul = make_shared<opset10::Multiply>(read_value, some_value);
        auto res = make_shared<opset10::Result>(mul);
        auto assign = make_shared<opset10::Assign>(mul, variable);

        model = make_shared<Model>(ResultVector{res}, SinkVector{assign}, ParameterVector{input});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        bool convert_input_output_precision = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32,
                                                      convert_input_output_precision);
        manager.run_passes(model);
    }

    {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{10, 10}, ov::element::f16, "variable_name"});

        auto input = make_shared<opset10::Parameter>(element::f16, Shape{10, 10});
        auto read_value = make_shared<opset10::ReadValue>(input, variable);

        auto some_value = opset10::Constant::create(element::f16, Shape{1}, {2});
        auto mul = make_shared<opset10::Multiply>(read_value, some_value);
        auto res = make_shared<opset10::Result>(mul);
        auto assign = make_shared<opset10::Assign>(mul, variable);

        model_ref = make_shared<Model>(ResultVector{res}, SinkVector{assign}, ParameterVector{input});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecision_assign_read_value_preserve_orig_types_as_rt_attribute) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{10, 10}, ov::element::f32, "variable_name"});

        auto input = make_shared<opset10::Parameter>(element::f32, Shape{10, 10});
        auto read_value = make_shared<opset10::ReadValue>(input, variable);

        auto some_value = opset10::Constant::create(element::f32, Shape{1}, {2});
        auto mul = make_shared<opset10::Multiply>(read_value, some_value);
        auto res = make_shared<opset10::Result>(mul);
        auto assign = make_shared<opset10::Assign>(mul, variable);

        model = make_shared<Model>(ResultVector{res}, SinkVector{assign}, ParameterVector{input});

        type_to_fuse_map empty_type_to_fuse_map = {};
        bool keep_precision_sensitive_in_fp32 = true;
        bool convert_input_output_precision = false;
        bool store_original_precision_as_rt_attribute = true;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      empty_type_to_fuse_map,
                                                      keep_precision_sensitive_in_fp32,
                                                      convert_input_output_precision,
                                                      store_original_precision_as_rt_attribute);
        manager.run_passes(model);
        EXPECT_EQ(ov::get_original_precision(read_value), element::f32);
        EXPECT_EQ(ov::get_original_precision(assign), element::f32);
    }

    {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{10, 10}, ov::element::f16, "variable_name"});

        auto input = make_shared<opset10::Parameter>(element::f32, Shape{10, 10});
        auto convert_1 = make_shared<opset10::Convert>(input, element::f16);
        auto read_value = make_shared<opset10::ReadValue>(convert_1, variable);

        auto some_value = opset10::Constant::create(element::f16, Shape{1}, {2});
        auto mul = make_shared<opset10::Multiply>(read_value, some_value);
        auto convert_2 = make_shared<opset10::Convert>(mul, element::f32);
        auto res = make_shared<opset10::Result>(convert_2);
        auto assign = make_shared<opset10::Assign>(mul, variable);

        model_ref = make_shared<Model>(ResultVector{res}, SinkVector{assign}, ParameterVector{input});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}
