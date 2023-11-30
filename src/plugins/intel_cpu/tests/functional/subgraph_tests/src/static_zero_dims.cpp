// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "functional_test_utils/skip_tests_config.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ov::test;

namespace SubgraphTestsDefinitions {

class StaticZeroDims : public SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        InputShape inputShapes{{}, {{7, 4}}};

        init_input_shapes({inputShapes});

        auto ngPrc = ngraph::element::f32;
        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, shape));
        }
        auto splitAxisOp = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{}, std::vector<int64_t>{0});
        std::vector<int> splitLenght = {1, 0, 6};
        auto splitLengthsOp = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i32, ngraph::Shape{splitLenght.size()}, splitLenght);
        auto varSplit = std::make_shared<ngraph::opset3::VariadicSplit>(inputParams[0], splitAxisOp, splitLengthsOp);

        auto relu1 = std::make_shared<ngraph::opset5::Relu>(varSplit->output(0));

        auto numInRoi = ngraph::builder::makeConstant(ngPrc, {0}, std::vector<float>{}, false);
        auto expDet = std::make_shared<ov::op::v6::ExperimentalDetectronTopKROIs>(varSplit->output(1), numInRoi, 10);
        auto relu2 = std::make_shared<ngraph::opset5::Relu>(expDet);

        auto relu3 = std::make_shared<ngraph::opset5::Relu>(varSplit->output(2));

        ngraph::NodeVector results{relu1, relu2, relu3};
        function = std::make_shared<ngraph::Function>(results, inputParams, "StaticZeroDims");
    }

    void compare(const std::vector<ov::Tensor> &expected, const std::vector<ov::Tensor> &actual) override {
        ASSERT_EQ(expected.size(), actual.size());
        for (size_t i = 0; i < expected.size(); i++) {
            // skip second output tensor because it's output ExperimentalDetectronTopKROIs: input dims [0, 4]
            // so according to spec output values undefined
            if (i == 1) {
                continue;
            }
            ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold);
        }
    }
};

TEST_F(StaticZeroDims, smoke_CompareWithRefs) {
    run();
}

TEST_F(StaticZeroDims, smoke_CompareWithRefs_FP16) {
    if (!(ov::with_cpu_x86_avx512_core_fp16() || ov::with_cpu_x86_avx512_core_amx_fp16())) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }
    configuration.insert({ov::hint::inference_precision.name(), ov::element::f16.to_string()});

    run();
}

} // namespace SubgraphTestsDefinitions
