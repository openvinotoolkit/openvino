// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ngraph;
namespace SubgraphTestsDefinitions {
class ConvDWConv : virtual public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto precision = ov::element::f32;
        ov::test::InputShape input_shape{{}, {{1, 32, 112, 112}}};
        init_input_shapes({input_shape});


        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, shape));
        }
        auto conv_weights = ngraph::builder::makeConstant(precision, std::vector<size_t>{32, 32, 1, 1}, std::vector<float>{}, true);
        auto conv = ngraph::builder::makeConvolution(params[0],
                                                     conv_weights,
                                                     precision,
                                                     std::vector<size_t>{1, 1},
                                                     std::vector<size_t>{1, 1},
                                                     ov::CoordinateDiff{0, 0},
                                                     ov::CoordinateDiff{0, 0},
                                                     std::vector<size_t>{1, 1},
                                                     ngraph::op::PadType::EXPLICIT,
                                                     32,
                                                     true);

        auto dw_conv_weights = ngraph::builder::makeConstant(precision, std::vector<size_t>{32, 1, 1, 3, 3}, std::vector<float>{}, true);
        auto dw_conv = ngraph::builder::makeGroupConvolution(conv,
                                                             dw_conv_weights,
                                                             precision,
                                                             std::vector<size_t>{1, 1},
                                                             ov::CoordinateDiff{1, 1},
                                                             ov::CoordinateDiff{1, 1},
                                                             std::vector<size_t>{1, 1},
                                                             ngraph::op::PadType::EXPLICIT);
        auto bias_const = ngraph::builder::makeConstant(precision, {1, 32 , 1, 1}, std::vector<float>{}, true);
        auto bias = std::make_shared<ov::opset10::Add>(dw_conv, bias_const);
        function = std::make_shared<ov::Model>(bias, params, "ConvDWConv");
    }
};

TEST_F(ConvDWConv, smoke_CompareWithRefs) {
    run();
}

TEST_F(ConvDWConv, smoke_CompareWithRefs_FP16) {
    if (!(ov::with_cpu_x86_avx512_core_fp16() || ov::with_cpu_x86_avx512_core_amx_fp16())) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }
    configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});
    run();
}

} // namespace SubgraphTestsDefinitions
