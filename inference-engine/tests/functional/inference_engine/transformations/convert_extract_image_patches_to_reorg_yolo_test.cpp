// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/convert_extract_image_patches_to_reorg_yolo.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertExtractImagePatchesToReorgYoloTests1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 10, 10});

        auto sizes = ngraph::Shape{5, 5};
        auto strides = ngraph::Strides{5, 5};
        auto rates = ngraph::Shape{1, 1};
        ngraph::op::PadType auto_pad = ngraph::op::PadType::VALID;

        auto eip = std::make_shared<ngraph::opset3::ExtractImagePatches>(input, sizes, strides, rates, auto_pad);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{eip}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertExtractImagePatchesToReorgYolo>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 10, 10});
        auto strides = ngraph::Strides{5, 5};
        auto reorg_yolo = std::make_shared<ngraph::opset3::ReorgYolo>(input, strides);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reorg_yolo}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertExtractImagePatchesToReorgYoloTestsNegative1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32,
            ngraph::PartialShape{1, 3, ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()});

        auto sizes = ngraph::Shape{5, 5};
        auto strides = ngraph::Strides{5, 5};
        auto rates = ngraph::Shape{1, 1};
        ngraph::op::PadType auto_pad = ngraph::op::PadType::VALID;

        auto eip = std::make_shared<ngraph::opset3::ExtractImagePatches>(input, sizes, strides, rates, auto_pad);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{eip}, ngraph::ParameterVector{input});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertExtractImagePatchesToReorgYolo>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32,
            ngraph::PartialShape{1, 3, ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()});

        auto sizes = ngraph::Shape{5, 5};
        auto strides = ngraph::Strides{5, 5};
        auto rates = ngraph::Shape{1, 1};
        ngraph::op::PadType auto_pad = ngraph::op::PadType::VALID;

        auto eip = std::make_shared<ngraph::opset3::ExtractImagePatches>(input, sizes, strides, rates, auto_pad);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{eip}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertExtractImagePatchesToReorgYoloTestsNegative2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 10, 10});

        auto sizes = ngraph::Shape{5, 5};
        auto strides = ngraph::Strides{5, 5};
        auto rates = ngraph::Shape{1, 1};
        ngraph::op::PadType auto_pad = ngraph::op::PadType::SAME_LOWER;

        auto eip = std::make_shared<ngraph::opset3::ExtractImagePatches>(input, sizes, strides, rates, auto_pad);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{eip}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertExtractImagePatchesToReorgYolo>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 10, 10});

        auto sizes = ngraph::Shape{5, 5};
        auto strides = ngraph::Strides{5, 5};
        auto rates = ngraph::Shape{1, 1};
        ngraph::op::PadType auto_pad = ngraph::op::PadType::SAME_LOWER;

        auto eip = std::make_shared<ngraph::opset3::ExtractImagePatches>(input, sizes, strides, rates, auto_pad);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{eip}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertExtractImagePatchesToReorgYoloTestsNegative3) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 10, 10});

        auto sizes = ngraph::Shape{3, 3};
        auto strides = ngraph::Strides{5, 5};
        auto rates = ngraph::Shape{1, 1};
        ngraph::op::PadType auto_pad = ngraph::op::PadType::VALID;

        auto eip = std::make_shared<ngraph::opset3::ExtractImagePatches>(input, sizes, strides, rates, auto_pad);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{eip}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertExtractImagePatchesToReorgYolo>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 10, 10});

        auto sizes = ngraph::Shape{3, 3};
        auto strides = ngraph::Strides{5, 5};
        auto rates = ngraph::Shape{1, 1};
        ngraph::op::PadType auto_pad = ngraph::op::PadType::VALID;

        auto eip = std::make_shared<ngraph::opset3::ExtractImagePatches>(input, sizes, strides, rates, auto_pad);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{eip}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
