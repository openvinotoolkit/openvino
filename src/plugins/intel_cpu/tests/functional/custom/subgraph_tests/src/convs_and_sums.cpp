// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace test {

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

class ConvsAndSums : virtual public SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        ov::element::Type netPrecision = ov::element::f32;

        targetDevice = ov::test::utils::DEVICE_CPU;

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, ov::Shape{1, 512, 32}),
                                   std::make_shared<ov::op::v0::Parameter>(netPrecision, ov::Shape{1, 128, 32})};

        auto FQ = ov::test::utils::make_fake_quantize(params[1],
                                                    netPrecision,
                                                    256,
                                                    {},
                                                    {-2.8215785026550293},
                                                    {2.799535036087036},
                                                    {-2.8215785026550293},
                                                    {2.799535036087036});
        auto FQ_0 = ov::test::utils::make_fake_quantize(params[1],
                                                      netPrecision,
                                                      256,
                                                      {},
                                                      {-5.031249523162842},
                                                      {4.991942882537842},
                                                      {-5.031249523162842},
                                                      {4.991942882537842});

        auto Add_0 = ov::test::utils::make_eltwise(FQ_0, FQ, ov::test::utils::EltwiseTypes::ADD);

        auto FQ_1 = ov::test::utils::make_fake_quantize(params[0],
                                                      netPrecision,
                                                      256,
                                                      {},
                                                      {-2.122633457183838},
                                                      {2.106050491333008},
                                                      {-2.122633457183838},
                                                      {2.106050491333008});

        auto Const = ov::op::v0::Constant::create(netPrecision, {128, 512, 1}, std::vector<float>{-0.0512377955019474});
        auto FQ_2 = ov::test::utils::make_fake_quantize(Const,
                                                      netPrecision,
                                                      255,
                                                      {128, 1, 1},
                                                      {-0.56387859582901},
                                                      {0.56387859582901},
                                                      {-0.56387859582901},
                                                      {0.56387859582901});

        auto Conv = std::make_shared<ov::op::v1::Convolution>(FQ_1,
                                                              FQ_2,
                                                              Strides{1},
                                                              CoordinateDiff{0},
                                                              CoordinateDiff{0},
                                                              Strides{1});

        auto Add = ov::test::utils::make_eltwise(Add_0, Conv, ov::test::utils::EltwiseTypes::ADD);

        auto FQ_11 = ov::test::utils::make_fake_quantize(params[0],
                                                       netPrecision,
                                                       256,
                                                       {},
                                                       {-3.2050728797912598},
                                                       {3.1800332069396973},
                                                       {-3.2050728797912598},
                                                       {3.1800332069396973});

        auto Const_ = ov::op::v0::Constant::create(netPrecision,
                                                   ov::Shape{128, 512, 1},
                                                   std::vector<float>{-0.001183388871140778});
        auto FQ_22 = ov::test::utils::make_fake_quantize(Const_,
                                                       netPrecision,
                                                       255,
                                                       {128, 1, 1},
                                                       {-0.325547456741333},
                                                       {0.325547456741333},
                                                       {-0.325547456741333},
                                                       {0.325547456741333});

        auto Conv2 = std::make_shared<ov::op::v1::Convolution>(FQ_11,
                                                               FQ_22,
                                                               Strides{1},
                                                               CoordinateDiff{0},
                                                               CoordinateDiff{0},
                                                               Strides{1});
        auto Add2 = ov::test::utils::make_eltwise(Add, Conv2, ov::test::utils::EltwiseTypes::ADD);
        auto relu3 = ov::test::utils::make_activation(Add2, netPrecision, ov::test::utils::ActivationTypes::Relu);

        auto result = std::make_shared<ov::op::v0::Result>(relu3);
        function = std::make_shared<ov::Model>(result, params, "SimpleNet");

        abs_threshold = 9e-4;
    }
};

TEST_F(ConvsAndSums, smoke_CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
