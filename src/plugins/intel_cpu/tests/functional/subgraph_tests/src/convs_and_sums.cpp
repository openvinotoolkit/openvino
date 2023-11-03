// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ngraph;
using ngraph::helpers::EltwiseTypes;

namespace SubgraphTestsDefinitions {

/* We can't fuse EltwiseAdd several times into one convolution

   FQ1    FQ2
     \   /
      ADD1      CONV1 [canBeExecutedInInt8]
        \      /
         \    /
          ADD2         CONV2 [canBeExecutedInInt8]
             \        /
              \      /
                ADD3
                 |
                RELU
                 |
               RESULT
*/

class ConvsAndSums : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision = InferenceEngine::Precision::FP32;

        targetDevice = ov::test::utils::DEVICE_CPU;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, 512, 32}),
                                   std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, 128, 32})};

        auto FQ = ngraph::builder::makeFakeQuantize(params[1], ngPrc, 256, {}, {-2.8215785026550293}, {2.799535036087036},
                                                      {-2.8215785026550293}, {2.799535036087036});
        auto FQ_0 = ngraph::builder::makeFakeQuantize(params[1], ngPrc, 256, {}, {-5.031249523162842}, {4.991942882537842},
                                                    {-5.031249523162842}, {4.991942882537842});

        auto Add_0 = ngraph::builder::makeEltwise(FQ_0, FQ, EltwiseTypes::ADD);

        auto FQ_1 = ngraph::builder::makeFakeQuantize(params[0], ngPrc, 256, {}, {-2.122633457183838}, {2.106050491333008},
                                                      {-2.122633457183838}, {2.106050491333008});

        auto Const = ngraph::builder::makeConstant(ngPrc, {128, 512, 1}, std::vector<float>{-0.0512377955019474}, false);
        auto FQ_2 = ngraph::builder::makeFakeQuantize(Const, ngPrc, 255, {128, 1, 1}, {-0.56387859582901}, {0.56387859582901},
                                                      {-0.56387859582901}, {0.56387859582901});

        auto Conv = std::make_shared<ngraph::opset1::Convolution>(FQ_1, FQ_2, Strides{1}, CoordinateDiff{0}, CoordinateDiff{0}, Strides{1});

        auto Add = ngraph::builder::makeEltwise(Add_0, Conv, EltwiseTypes::ADD);

        auto FQ_11 = ngraph::builder::makeFakeQuantize(params[0], ngPrc, 256, {}, {-3.2050728797912598}, {3.1800332069396973},
                                                      {-3.2050728797912598}, {3.1800332069396973});

        auto Const_ = ngraph::builder::makeConstant(ngPrc, {128, 512, 1}, std::vector<float>{-0.001183388871140778}, false);
        auto FQ_22 = ngraph::builder::makeFakeQuantize(Const_, ngPrc, 255, {128, 1, 1}, {-0.325547456741333}, {0.325547456741333},
                                                      {-0.325547456741333}, {0.325547456741333});

        auto Conv2 = std::make_shared<ngraph::opset1::Convolution>(FQ_11, FQ_22, Strides{1}, CoordinateDiff{0}, CoordinateDiff{0}, Strides{1});
        auto Add2 = ngraph::builder::makeEltwise(Add, Conv2, EltwiseTypes::ADD);
        auto relu3 = ngraph::builder::makeActivation(Add2, ngPrc, ngraph::helpers::ActivationTypes::Relu);

        auto result = std::make_shared<ngraph::opset1::Result>(relu3);
        function = std::make_shared<ngraph::Function>(result, params, "SimpleNet");
    }
};

TEST_F(ConvsAndSums, smoke_CompareWithRefs) {
    Run();
}

} // namespace SubgraphTestsDefinitions
