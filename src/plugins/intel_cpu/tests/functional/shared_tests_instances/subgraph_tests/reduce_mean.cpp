// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

#include <tuple>
#include <vector>

namespace ov {
namespace test {

typedef std::tuple<ov::element::Type,    // input precision
                   std::vector<size_t>,  // input shape
                   std::vector<size_t>   // axes
                   > ReduceMeanParams;

class ReduceMeanTest : public testing::WithParamInterface<ReduceMeanParams>,
                       public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReduceMeanParams>& obj) {
        ov::element::Type input_precision;
        std::vector<size_t> input_shape;
        std::vector<size_t> axes;
        std::tie(input_precision, input_shape, axes) = obj.param;
        std::ostringstream result;
        result << "inputPrecision=" << input_precision.to_string() << "_";
        result << "inputShape=" << ov::test::utils::vec2str(input_shape) << "_";
        result << "axes=" << ov::test::utils::vec2str(axes);
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::element::Type input_precision;
        std::vector<size_t> input_shape;
        std::vector<size_t> axes;
        std::vector<size_t> axes_shape;
        std::tie(input_precision, input_shape, axes) = this->GetParam();
        axes_shape.push_back(axes.size());

        auto dataNode = std::make_shared<ov::op::v0::Parameter>(input_precision, ov::Shape(input_shape));
        auto axesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape(axes_shape), axes);

        const auto reduce_mean = std::make_shared<ov::op::v1::ReduceMean>(dataNode, axesNode, true);

        ov::ParameterVector inputs;
        inputs.push_back(dataNode);
        ov::ResultVector outputs;
        outputs.push_back(std::make_shared<ov::op::v0::Result>(reduce_mean));
        function = std::make_shared<ov::Model>(outputs, inputs);
    }
};

TEST_P(ReduceMeanTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<ov::element::Type> input_precision = {ov::element::i32, ov::element::i8, ov::element::u8};
const std::vector<std::vector<size_t>> input_shape = {{1, 2, 3, 3}};
const std::vector<std::vector<size_t>> axes = {{2, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_ReduceMean,
                         ReduceMeanTest,
                         testing::Combine(
                            ::testing::ValuesIn(input_precision),
                            ::testing::ValuesIn(input_shape),
                            ::testing::ValuesIn(axes)),
                         ReduceMeanTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
