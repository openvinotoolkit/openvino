// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <transformations/common_optimizations/pad_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


TEST(TransformationTests, PadFusionAvgPoolExcludePad) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<opset5::AvgPool>(pad, Strides{1, 1},
                                                          Shape{0, 0}, Shape{0, 0},
                                                          Shape{4, 4}, true, op::RoundingType::FLOOR);
        f = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::PadFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto avg_pool = std::make_shared<opset5::AvgPool>(data, Strides{1, 1},
                                                          Shape{1, 1}, Shape{2, 2}, Shape{4, 4},
                                                          false, op::RoundingType::FLOOR, op::PadType::EXPLICIT);
        f_ref = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, PadFusionAvgPoolDontExcludePad) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<opset5::AvgPool>(pad, Strides{1, 1},
                                                          Shape{0, 0}, Shape{1, 1},
                                                          Shape{4, 4}, false, op::RoundingType::FLOOR);
        f = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::PadFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto avg_pool = std::make_shared<opset5::AvgPool>(data, Strides{1, 1},
                                                          Shape{1, 1}, Shape{3, 3}, Shape{4, 4},
                                                          false, op::RoundingType::FLOOR, op::PadType::EXPLICIT);
        f_ref = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, PadFusionMaxPool) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto max_pool = std::make_shared<opset5::MaxPool>(pad, Strides{1, 1},
                                                          Shape{0, 0}, Shape{1, 1}, Shape{4, 4});
        f = std::make_shared<Function>(NodeVector{max_pool}, ParameterVector{data});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::PadFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto max_pool = std::make_shared<opset5::MaxPool>(data, Strides{1, 1},
                                                          Shape{1, 1}, Shape{3, 3}, Shape{4, 4},
                                                          op::RoundingType::FLOOR, op::PadType::EXPLICIT);
        f_ref = std::make_shared<Function>(NodeVector{max_pool}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, PadFusionConvolution) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::PadFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(data, filters, Strides{1, 1},
                                                          CoordinateDiff{1, 1}, CoordinateDiff{3, 3}, Shape{1, 1},
                                                          op::PadType::EXPLICIT);
        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, PadFusionConvolutionBackpropData) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<opset5::ConvolutionBackpropData>(pad, filters, Strides{1, 1},
                                                                      CoordinateDiff{4, 4}, CoordinateDiff{3, 3}, Shape{1, 1});

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::PadFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<opset5::ConvolutionBackpropData>(data, filters, Strides{1, 1},
                                                                      CoordinateDiff{3, 3}, CoordinateDiff{1, 1}, Shape{1, 1});

        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, PadFusionGroupConvolution) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<opset5::GroupConvolution>(pad, filters, Strides{1, 1},
                                                               CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::PadFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<opset5::GroupConvolution>(data, filters, Strides{1, 1},
                                                               CoordinateDiff{1, 1}, CoordinateDiff{3, 3}, Shape{1, 1},
                                                               op::PadType::EXPLICIT);
        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, PadFusionGroupConvolutionBackpropData) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 3, 1});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<opset5::GroupConvolutionBackpropData>(pad, filters, Strides{1, 1},
                                                               CoordinateDiff{3, 2}, CoordinateDiff{4, 3}, Shape{1, 1});
        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::PadFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<opset5::GroupConvolutionBackpropData>(data, filters, Strides{1, 1},
                                                                           CoordinateDiff{2, 1}, CoordinateDiff{1, 2}, Shape{1, 1});
        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativePadFusionNonConstantPadMode) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::REFLECT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::PadFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::REFLECT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativePadFusionNonZeroPadValue) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = opset5::Constant::create(element::i32, Shape{}, {2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::PadFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = opset5::Constant::create(element::i32, Shape{}, {2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativePadFusionPadForBatchSize) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {1, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = opset5::Constant::create(element::i32, Shape{}, {0});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::PadFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {1, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = opset5::Constant::create(element::i32, Shape{}, {0});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<opset5::Convolution>(pad, filters, Strides{1, 1},
                                                          CoordinateDiff{0, 0}, CoordinateDiff{1, 1}, Shape{1, 1});
        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativePadFusionAvgPoolExcludePadNonZeroPads) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<opset5::AvgPool>(pad, Strides{1, 1},
                                                          Shape{0, 0}, Shape{1, 1},
                                                          Shape{4, 4}, true, op::RoundingType::FLOOR);
        f = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::PadFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::i32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<opset5::AvgPool>(pad, Strides{1, 1},
                                                          Shape{0, 0}, Shape{1, 1},
                                                          Shape{4, 4}, true, op::RoundingType::FLOOR);
        f_ref = std::make_shared<Function>(NodeVector{avg_pool}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativePadFusionConvolutionBackpropDataTooSmallPad) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<opset5::ConvolutionBackpropData>(pad, filters, Strides{1, 1},
                                                                      CoordinateDiff{1, 1}, CoordinateDiff{1, 1}, Shape{1, 1});

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::PadFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<opset5::Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<opset5::ConvolutionBackpropData>(pad, filters, Strides{1, 1},
                                                                      CoordinateDiff{1, 1}, CoordinateDiff{1, 1}, Shape{1, 1});

        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
