// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>
#include <vector>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

template<ngraph::element::Type_t T>
bool has_type(std::shared_ptr<ngraph::Function> f) {
    for (auto & node : f->get_ordered_ops()) {
        for (auto & input : node->inputs()) {
            if (input.get_element_type() == element::Type(T)) {
                return true;
            }
        }
        for (auto & output : node->outputs()) {
            if (output.get_element_type() == element::Type(T)) {
                return true;
            }
        }
    }
    return false;
}

TEST(TransformationTests, ConvertPrecision_NMS3) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto boxes = std::make_shared<opset3::Parameter>(element::f16, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset3::Parameter>(element::f16, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset3::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset3::Constant::create(element::f16, Shape{}, {0.75});
        auto score_threshold = opset3::Constant::create(element::f16, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, opset3::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::f16, ngraph::element::f32);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_NMS4) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto boxes = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset4::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset4::Constant::create(element::f16, Shape{}, {0.75});
        auto score_threshold = opset4::Constant::create(element::f16, Shape{}, {0.7});
        auto nms = std::make_shared<opset4::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, opset4::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::f16, ngraph::element::f32);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_ShapeOf) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 4});
        auto shape_of = std::make_shared<opset4::ShapeOf>(input);

        f = std::make_shared<Function>(NodeVector{shape_of}, ParameterVector{input});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::f16, ngraph::element::f32);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_Convert) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 4});
        auto convert = std::make_shared<opset4::Convert>(input, element::i64);

        f = std::make_shared<Function>(NodeVector{convert}, ParameterVector{input});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::f16, ngraph::element::f32);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_ConvertElimination) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 4});
        auto relu = std::make_shared<opset4::Relu>(input);
        auto convert = std::make_shared<opset4::Convert>(relu, element::f32);

        f = std::make_shared<Function>(NodeVector{convert}, ParameterVector{input});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::f16, ngraph::element::f32);
        manager.run_passes(f);
        ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1000, 4});
        auto relu = std::make_shared<opset4::Relu>(input);

        f_ref = std::make_shared<Function>(NodeVector{relu}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertPrecision_TopK) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset3::TopK>(input, k, 1, "min", "value", ngraph::element::i64);

        f = std::make_shared<Function>(OutputVector{topk->output(0), topk->output(1)}, ParameterVector{input});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::f16, ngraph::element::f32);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_NonZero) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto non_zero = std::make_shared<ngraph::opset4::NonZero>(input, ngraph::element::i64);

        f = std::make_shared<Function>(OutputVector{non_zero}, ParameterVector{input});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::f16, ngraph::element::f32);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_Bucketize) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{20});
        auto k = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
        auto b = std::make_shared<ngraph::opset4::Bucketize>(input, k);

        f = std::make_shared<Function>(OutputVector{b}, ParameterVector{input});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::f16, ngraph::element::f32);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_Roundings) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto max_int64 = std::numeric_limits<int64_t>::max();
        auto max_int32 = std::numeric_limits<int32_t>::max();

        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{5, 5, 5, 5});
        auto begin = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {max_int64, max_int64, max_int64, max_int64});
        auto stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto ss = std::make_shared<ngraph::opset1::StridedSlice>(input, begin, end, stride, begin_mask, end_mask);

        f = std::make_shared<Function>(OutputVector{ss}, ParameterVector{input});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::f16, ngraph::element::f32);
        manager.run_passes(f);

        auto casted_end = std::dynamic_pointer_cast<opset1::Constant>(ss->input_value(2).get_node_shared_ptr());
        ASSERT_TRUE(casted_end != nullptr);
        ASSERT_EQ(casted_end->get_element_type(), element::i32);
        ASSERT_EQ(casted_end->cast_vector<int32_t>(), std::vector<int32_t>({max_int32, max_int32, max_int32, max_int32}));
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
}