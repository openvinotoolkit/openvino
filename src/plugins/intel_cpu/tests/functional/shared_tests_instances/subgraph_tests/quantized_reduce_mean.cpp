// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

#include <tuple>
#include <vector>

namespace ov {
namespace test {

typedef std::tuple<std::vector<size_t>,  // input shape
                   std::vector<size_t>   // axes
                   > QuantizedReduceMeanParams;

class QuantizedReduceMeanTest : public testing::WithParamInterface<QuantizedReduceMeanParams>,
                       public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QuantizedReduceMeanParams>& obj) {
        std::vector<size_t> input_shape;
        std::vector<size_t> axes;
        std::tie(input_shape, axes) = obj.param;
        std::ostringstream result;
        result << "inputShape=" << ov::test::utils::vec2str(input_shape) << "_";
        result << "axes=" << ov::test::utils::vec2str(axes);
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        std::vector<size_t> input_shape;
        std::vector<size_t> axes;
        std::vector<size_t> axes_shape;
        std::tie(input_shape, axes) = this->GetParam();
        axes_shape.push_back(axes.size());

        auto dataNode = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(input_shape));

        std::vector<size_t> dataFqConstShapes(input_shape.size(), 1);
        size_t constDataSize = ov::shape_size(dataFqConstShapes);
        std::vector<float> inputLowData(constDataSize), inputHighData(constDataSize), outputLowData(constDataSize), outputHighData(constDataSize);
        for (size_t i = 0; i < constDataSize; i++) {
            inputLowData[i] = 0;
            inputHighData[i] = 255;
            outputLowData[i] = 0;
            outputHighData[i] = 255;
        }
        auto dataFqNode = ov::test::utils::make_fake_quantize(
             dataNode, ov::element::f32, 256, dataFqConstShapes, inputLowData, inputHighData, outputLowData, outputHighData);


        auto axesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape(axes_shape), axes);

        const auto reduce_mean = std::make_shared<ov::op::v1::ReduceMean>(dataFqNode, axesNode, true);

        ov::ParameterVector inputs;
        inputs.push_back(dataNode);
        ov::ResultVector outputs;
        outputs.push_back(std::make_shared<ov::op::v0::Result>(reduce_mean));
        function = std::make_shared<ov::Model>(outputs, inputs);
    }
};

TEST_P(QuantizedReduceMeanTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<std::vector<size_t>> input_shape = {{1, 2, 3, 3}};
const std::vector<std::vector<size_t>> axes = {{2, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_QuantizedReduceMean,
                         QuantizedReduceMeanTest,
                         testing::Combine(
                            ::testing::ValuesIn(input_shape),
                            ::testing::ValuesIn(axes)),
                         QuantizedReduceMeanTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
