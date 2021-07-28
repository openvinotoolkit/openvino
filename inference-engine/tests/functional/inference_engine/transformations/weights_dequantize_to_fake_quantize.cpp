// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/init_node_info.hpp>
#include <precomp.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

struct FQ_as_Mul_Sub_dequantize {
    int8_t min_int, max_int;
    float zp, scale;
    float o_low, o_high;
    size_t levels;
};

class TranslateNewWeightFormatToOldOne : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<FQ_as_Mul_Sub_dequantize, ngraph::element::Type>> {
public:
    std::shared_ptr<ngraph::Function> f, f_ref;

    void SetUp() override {
        const auto& test_case = std::get<0>(GetParam());
        const auto& float_element_type = std::get<1>(GetParam());

        std::vector<int8_t> weights{test_case.min_int, test_case.max_int};
        {
            auto i_weights = std::make_shared<ngraph::opset6::Constant>(ngraph::element::i8, ngraph::Shape{weights.size()}, weights);

            auto f_weights = std::make_shared<ngraph::opset6::Convert>(i_weights, float_element_type);

            auto zp = std::make_shared<ngraph::opset6::Constant>(float_element_type, ngraph::Shape{}, std::vector<float>{test_case.zp});
            auto subtract_zp = std::make_shared<ngraph::opset6::Subtract>(f_weights, zp);

            auto scale = std::make_shared<ngraph::opset6::Constant>(float_element_type, ngraph::Shape{}, std::vector<float>{test_case.scale});

            ngraph::NodeVector output;
            if (test_case.zp == 0)
                output.push_back(std::make_shared<ngraph::opset6::Multiply>(f_weights, scale));
            else
                output.push_back(std::make_shared<ngraph::opset6::Multiply>(subtract_zp, scale));

            f = std::make_shared<ngraph::Function>(output, ngraph::ParameterVector{});
        }

        {
            auto i_weights = std::make_shared<ngraph::opset6::Constant>(ngraph::element::i8, ngraph::Shape{weights.size()}, weights);

            auto f_weights = std::make_shared<ngraph::opset6::Convert>(i_weights, float_element_type);

            auto i_low = std::make_shared<ngraph::opset6::Constant>(
                    float_element_type, ngraph::Shape{}, std::vector<float>{static_cast<float>(test_case.min_int)});
            auto i_high = std::make_shared<ngraph::opset6::Constant>(
                    float_element_type, ngraph::Shape{}, std::vector<float>{static_cast<float>(test_case.max_int)});
            auto o_low = std::make_shared<ngraph::opset6::Constant>(float_element_type, ngraph::Shape{}, std::vector<float>{test_case.o_low});
            auto o_high = std::make_shared<ngraph::opset6::Constant>(float_element_type, ngraph::Shape{}, std::vector<float>{test_case.o_high});

            auto fq = std::make_shared<ngraph::opset6::FakeQuantize>(f_weights, i_low, i_high, o_low, o_high, test_case.levels);

            f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fq}, ngraph::ParameterVector{});
        }
    }
};

TEST_P(TranslateNewWeightFormatToOldOne, ReshapeMatMul) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::WeightsDequantizeToFakeQuantize>();
    manager.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_SUITE_P(NGraph, TranslateNewWeightFormatToOldOne, testing::Combine(
        testing::Values(
            FQ_as_Mul_Sub_dequantize{-128, 127, 1, 2, (-128 - 1) * 2, (127 - 1) * 2, 256},
            FQ_as_Mul_Sub_dequantize{-127, 127, 1, 2, (-127 - 1) * 2, (127 - 1) * 2, 255},
            FQ_as_Mul_Sub_dequantize{-128, 127, 0, 2, (-128 - 0) * 2, (127 - 0) * 2, 256},
            FQ_as_Mul_Sub_dequantize{-127, 127, 0, 2, (-127 - 0) * 2, (127 - 0) * 2, 255}),
        testing::Values(
            ngraph::element::f32,
            ngraph::element::f16)));
