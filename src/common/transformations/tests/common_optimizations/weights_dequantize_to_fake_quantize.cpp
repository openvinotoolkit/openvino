// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/pass/manager.hpp>
#include <precomp.hpp>
#include <string>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace ov;
using namespace testing;

struct FQ_as_Mul_Sub_dequantize {
    int8_t min_int, max_int;
    float zp, scale;
    float o_low, o_high;
    size_t levels;
};

class TranslateNewWeightFormatToOldOne
    : public ov::test::TestsCommon,
      public testing::WithParamInterface<std::tuple<FQ_as_Mul_Sub_dequantize, element::Type>> {
public:
    std::shared_ptr<ov::Model> f, f_ref;

    void SetUp() override {
        const auto& test_case = std::get<0>(GetParam());
        const auto& float_element_type = std::get<1>(GetParam());

        std::vector<int8_t> weights{test_case.min_int, test_case.max_int};
        {
            auto i_weights = std::make_shared<opset6::Constant>(element::i8, Shape{weights.size()}, weights);

            auto f_weights = std::make_shared<opset6::Convert>(i_weights, float_element_type);

            auto zp = std::make_shared<opset6::Constant>(float_element_type, Shape{}, std::vector<float>{test_case.zp});
            auto subtract_zp = std::make_shared<opset6::Subtract>(f_weights, zp);

            auto scale =
                std::make_shared<opset6::Constant>(float_element_type, Shape{}, std::vector<float>{test_case.scale});

            NodeVector output;
            if (test_case.zp == 0)
                output.push_back(std::make_shared<opset6::Multiply>(f_weights, scale));
            else
                output.push_back(std::make_shared<opset6::Multiply>(subtract_zp, scale));

            f = std::make_shared<ov::Model>(output, ParameterVector{});
        }

        {
            auto i_weights = std::make_shared<opset6::Constant>(element::i8, Shape{weights.size()}, weights);

            auto f_weights = std::make_shared<opset6::Convert>(i_weights, float_element_type);

            auto i_low = std::make_shared<opset6::Constant>(float_element_type,
                                                            Shape{},
                                                            std::vector<float>{static_cast<float>(test_case.min_int)});
            auto i_high = std::make_shared<opset6::Constant>(float_element_type,
                                                             Shape{},
                                                             std::vector<float>{static_cast<float>(test_case.max_int)});
            auto o_low =
                std::make_shared<opset6::Constant>(float_element_type, Shape{}, std::vector<float>{test_case.o_low});
            auto o_high =
                std::make_shared<opset6::Constant>(float_element_type, Shape{}, std::vector<float>{test_case.o_high});

            auto fq = std::make_shared<opset6::FakeQuantize>(f_weights, i_low, i_high, o_low, o_high, test_case.levels);

            f_ref = std::make_shared<ov::Model>(NodeVector{fq}, ParameterVector{});
        }
    }
};

TEST_P(TranslateNewWeightFormatToOldOne, ReshapeMatMul) {
    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    pass::Manager m;
    m.register_pass<ov::pass::InitUniqueNames>(unh);
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::WeightsDequantizeToFakeQuantize>();
    m.register_pass<ov::pass::CheckUniqueNames>(unh);
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    auto fc = FunctionsComparator::no_default()
                  .enable(FunctionsComparator::NODES)
                  .enable(FunctionsComparator::PRECISIONS)
                  .enable(FunctionsComparator::CONST_VALUES);
    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

INSTANTIATE_TEST_SUITE_P(
    NGraph,
    TranslateNewWeightFormatToOldOne,
    testing::Combine(testing::Values(FQ_as_Mul_Sub_dequantize{-128, 127, 1, 2, (-128 - 1) * 2, (127 - 1) * 2, 256},
                                     FQ_as_Mul_Sub_dequantize{-127, 127, 1, 2, (-127 - 1) * 2, (127 - 1) * 2, 255},
                                     FQ_as_Mul_Sub_dequantize{-128, 127, 0, 2, (-128 - 0) * 2, (127 - 0) * 2, 256},
                                     FQ_as_Mul_Sub_dequantize{-127, 127, 0, 2, (-127 - 0) * 2, (127 - 0) * 2, 255}),
                     testing::Values(element::f32, element::f16)));
