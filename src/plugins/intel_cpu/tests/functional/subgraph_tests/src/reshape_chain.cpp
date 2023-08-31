// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/ov_subgraph.hpp>
#include <ngraph_functions/builders.hpp>
#include <openvino/pass/serialize.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

using namespace ov::test;

namespace SubgraphTestsDefinitions {

class ReshapeChain : public SubgraphBaseTest {
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        InputShape inputShapes{{-1, -1, -1, -1}, {{10, 20, 30, 40}, {16, 24, 16, 24}, {4, 8, 12, 16}}};

        init_input_shapes({inputShapes});
        auto ngPrc = ngraph::element::f32;
        const auto secondInPrc = ngraph::element::Type_t::i32;
        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, shape));
        }
        auto reshapeParam1 = ngraph::builder::makeConstant<int>(secondInPrc, {3}, {0, 0, -1});
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(inputParams.front(), reshapeParam1, true);
        auto reshapeParam2 = ngraph::builder::makeConstant<int>(secondInPrc, {2}, {0, -1});
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(reshape1, reshapeParam2, true);
        auto reshapeParam3 = ngraph::builder::makeConstant<int>(secondInPrc, {1}, {-1});
        auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(reshape2, reshapeParam3, true);
        auto reshapeParam4 = ngraph::builder::makeConstant<int>(secondInPrc, {2}, {4, -1});
        auto reshape4 = std::make_shared<ngraph::opset1::Reshape>(reshape3, reshapeParam4, true);

        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(reshape4)};
        function = std::make_shared<ngraph::Function>(results, inputParams, "reshapeChain");
    }
};

TEST_F(ReshapeChain, smoke_ReshapeChain) {
    run();
}

TEST_F(ReshapeChain, smoke_ReshapeChain_FP16) {
    if (!(ov::with_cpu_x86_avx512_core_fp16() || ov::with_cpu_x86_avx512_core_amx_fp16())) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }
    configuration.insert({ov::hint::inference_precision.name(), "f16"});

    run();
}


} // namespace SubgraphTestsDefinitions
