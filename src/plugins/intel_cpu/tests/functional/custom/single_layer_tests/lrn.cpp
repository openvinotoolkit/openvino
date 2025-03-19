// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
using LRNParams = std::tuple<
    ElementType,           // data precision
    InputShape,            // data shape
    double,                // alpha
    double,                // beta
    double,                // bias
    size_t,                // size
    std::vector<int64_t>>; // axes to reduction

class LRNLayerCPUTest : public testing::WithParamInterface<LRNParams>, public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LRNParams> obj) {
        ElementType inputPrecision;
        InputShape inputShapes;
        double alpha, beta, bias;
        size_t size;
        std::vector<int64_t> axes;
        std::tie(inputPrecision, inputShapes, alpha, beta, bias, size, axes) = obj.param;

        std::ostringstream result;
        result << inputPrecision << "_" << "IS=" << ov::test::utils::partialShape2str({ inputShapes.first }) << "_" << "TS=(";
        for (const auto& shape : inputShapes.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }

        result << ")_alpha=" << alpha << "_beta=" << beta << "_bias=" << bias << "_size=" << size << "_axes=" << ov::test::utils::vec2str(axes);
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        ElementType inputPrecision;
        InputShape inputShapes;
        double alpha, beta, bias;
        size_t size;
        std::vector<int64_t> axes;
        std::tie(inputPrecision, inputShapes, alpha, beta, bias, size, axes) = this->GetParam();

        init_input_shapes({ inputShapes });
        selectedType = makeSelectedTypeStr("ref_any", inputPrecision);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inputPrecision, shape));
        }
        auto axesNode = ov::op::v0::Constant::create(ov::element::i32, { axes.size() }, axes);
        auto lrn = std::make_shared<ov::op::v0::LRN>(params[0], axesNode, alpha, beta, bias, size);
        function = makeNgraphFunction(inputPrecision, params, lrn, "LRN");
        if (inputPrecision == ov::element::f32) {
            abs_threshold = 5e-3;
        }
    }
};

TEST_P(LRNLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "LRN");
}

const std::vector<ElementType> inputPrecisions = {
    ov::element::f32,
};

const std::vector<std::vector<std::int64_t>> axes = {
    { 1 },
    { 2, 3 },
    { 3, 2 },
    { 1, 2, 3 }
};

const std::vector<double> alpha = { 9.9e-05 };
const std::vector<double> beta = { 2. };
const std::vector<double> bias = { 1. };
const std::vector<size_t> size = { 5ul };

const std::vector<InputShape> inputShapes = {
    InputShape{{}, {{10, 10, 3, 8}}},
    InputShape{
        // dynamic
        {-1, -1, -1, -1},
        // static
        {{15, 5, 7, 8}, {10, 10, 3, 8}, {1, 3, 5, 5}, {10, 10, 3, 8}}
    },
    InputShape{
        // dynamic
        {{1, 15}, {3, 10}, {3, 7}, {5, 8}},
        // static
        {{15, 5, 7, 8}, {10, 10, 3, 8}, {1, 3, 5, 5}, {10, 10, 3, 8}}
    },
    InputShape{
        // dynamic
        {{1, 15}, 3, 5, 5},
        // static
        {{2, 3, 5, 5}, {1, 3, 5, 5}, {3, 3, 5, 5}}
    },
};

const auto testCases = ::testing::Combine(
    ::testing::ValuesIn(inputPrecisions),
    ::testing::ValuesIn(inputShapes),
    ::testing::ValuesIn(alpha),
    ::testing::ValuesIn(beta),
    ::testing::ValuesIn(bias),
    ::testing::ValuesIn(size),
    ::testing::ValuesIn(axes)
);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, LRNLayerCPUTest, testCases, LRNLayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
