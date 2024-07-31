// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

#include <tuple>
#include <vector>

namespace ov {
namespace test {

typedef std::tuple<ov::element::Type,    // input precision
                   std::vector<size_t>,  // input shape
                   std::vector<size_t>,  // axes
                   bool                  // quantized
                   > IntegerReduceMeanParams;

// IntegerReduceMeanTest covers the two rounding scenarios in ReduceMean with integer inputs.
// Scenario 1: ReduceMean has both input and output precisions to be integers from the original model, so rounding to zero should
//             be done before converting intermediate floating point value to integer. Covered by test suite smoke_ReduceMeanIntegerInput.
// Scenario 2: Integer inputs of ReduceMean are resulted from quantization, then such rounding should not be done, in order to maintain
//             accuracy. Coverd by test suite smoke_ReduceMeanQuantized.
class IntegerReduceMeanTest : public testing::WithParamInterface<IntegerReduceMeanParams>,
                       public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<IntegerReduceMeanParams>& obj) {
        ov::element::Type input_precision;
        std::vector<size_t> input_shape;
        std::vector<size_t> axes;
        bool quantized;
        std::tie(input_precision, input_shape, axes, quantized) = obj.param;
        std::ostringstream result;
        result << "inputPrecision=" << input_precision.to_string() << "_";
        result << "inputShape=" << ov::test::utils::vec2str(input_shape) << "_";
        result << "axes=" << ov::test::utils::vec2str(axes) << "_";
        if (quantized)
            result << "quantized=true";
        else
            result << "quantized=false";
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::element::Type input_precision;
        std::vector<size_t> input_shape;
        std::vector<size_t> axes;
        std::vector<size_t> axes_shape;
        bool quantized;
        std::tie(input_precision, input_shape, axes, quantized) = this->GetParam();
        axes_shape.push_back(axes.size());

        auto dataNode = std::make_shared<ov::op::v0::Parameter>(input_precision, ov::Shape(input_shape));
        auto axesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape(axes_shape), axes);

        std::shared_ptr<ov::op::v1::ReduceMean> reduce_mean;
        if (quantized) {
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
                dataNode, input_precision, 256, dataFqConstShapes, inputLowData, inputHighData, outputLowData, outputHighData);
            reduce_mean = std::make_shared<ov::op::v1::ReduceMean>(dataFqNode, axesNode, true);
        } else {
            reduce_mean = std::make_shared<ov::op::v1::ReduceMean>(dataNode, axesNode, true);
        }

        ov::ParameterVector inputs;
        inputs.push_back(dataNode);
        ov::ResultVector outputs;
        outputs.push_back(std::make_shared<ov::op::v0::Result>(reduce_mean));
        function = std::make_shared<ov::Model>(outputs, inputs);
    }
};

TEST_P(IntegerReduceMeanTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<ov::element::Type> input_precision = {ov::element::f32};
const std::vector<ov::element::Type> integer_input_precision = {ov::element::i32, ov::element::i8, ov::element::u8};
const std::vector<std::vector<size_t>> input_shape = {{1, 2, 3, 3}};
const std::vector<std::vector<size_t>> axes = {{2, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_ReduceMeanQuantized,
                         IntegerReduceMeanTest,
                         testing::Combine(
                            ::testing::ValuesIn(input_precision),
                            ::testing::ValuesIn(input_shape),
                            ::testing::ValuesIn(axes),
                            ::testing::Values(true)),
                         IntegerReduceMeanTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReduceMeanIntegerInput,
                         IntegerReduceMeanTest,
                         testing::Combine(
                            ::testing::ValuesIn(integer_input_precision),
                            ::testing::ValuesIn(input_shape),
                            ::testing::ValuesIn(axes),
                            ::testing::Values(false)),
                         IntegerReduceMeanTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
