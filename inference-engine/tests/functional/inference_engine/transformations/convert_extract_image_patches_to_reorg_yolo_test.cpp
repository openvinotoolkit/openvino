// Copyright (C) 2020 Intel Corporation
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
#include <vector>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph_ops/crop_ie.hpp>

#include <transformations/convert_extract_image_patches_to_reorg_yolo.hpp>
#include <ngraph/op/reshape.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include "ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertExtractImagePatchesToReorgYoloTests) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 10, 10 });

        std::vector<int64_t> sizes = { 5, 5 };
        std::vector<int64_t> strides = { 5, 5 };
        std::vector<int64_t> rates = { 1, 1 };
        ngraph::op::PadType auto_pad = ngraph::op::PadType::VALID;


        auto eip = std::make_shared<ngraph::opset3::ExtractImagePatches>(input, sizes, strides, rates, auto_pad);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ eip }, ngraph::ParameterVector{ input });
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertExtractImagePatchesToReorgYolo().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 10, 10 });

        std::vector<int64_t> stride = { 5, 5 };

        auto reorg_yolo = std::make_shared<ngraph::opset3::ReorgYolo>(input, stride);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reorg_yolo }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertExtractImagePatchesToReorgYoloTestsNegative) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 10, 10 });

        std::vector<int64_t> sizes = { 5, 5 };
        std::vector<int64_t> strides = { 5, 5 };
        std::vector<int64_t> rates = { 1, 1 };
        ngraph::op::PadType auto_pad = ngraph::op::PadType::SAME_LOWER;


        auto eip = std::make_shared<ngraph::opset3::ExtractImagePatches>(input, sizes, strides, rates, auto_pad);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ eip }, ngraph::ParameterVector{ input });
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertExtractImagePatchesToReorgYolo().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 10, 10 });

        std::vector<int64_t> stride = { 5, 5 };

        auto reorg_yolo = std::make_shared<ngraph::opset3::ReorgYolo>(input, stride);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reorg_yolo }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
