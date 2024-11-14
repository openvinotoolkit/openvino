// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

class StaticZeroDims : public SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        InputShape inputShapes{{}, {{7, 4}}};

        init_input_shapes({inputShapes});

        auto ngPrc = ov::element::f32;
        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, shape));
        }
        auto splitAxisOp = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
        std::vector<int> splitLenght = {1, 0, 6};
        auto splitLengthsOp = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{splitLenght.size()}, splitLenght);
        auto varSplit = std::make_shared<ov::op::v1::VariadicSplit>(inputParams[0], splitAxisOp, splitLengthsOp);

        auto relu1 = std::make_shared<ov::op::v0::Relu>(varSplit->output(0));

        auto numInRoi = ov::op::v0::Constant::create(ngPrc, {0}, std::vector<float>{});
        auto expDet = std::make_shared<ov::op::v6::ExperimentalDetectronTopKROIs>(varSplit->output(1), numInRoi, 10);
        auto relu2 = std::make_shared<ov::op::v0::Relu>(expDet);

        auto relu3 = std::make_shared<ov::op::v0::Relu>(varSplit->output(2));

        ov::NodeVector results{relu1, relu2, relu3};
        function = std::make_shared<ov::Model>(results, inputParams, "StaticZeroDims");
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

}  // namespace test
}  // namespace ov
