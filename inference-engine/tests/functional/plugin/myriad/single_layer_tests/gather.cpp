// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/gather.hpp"
#include <vpu/private_plugin_config.hpp>

using namespace LayerTestsDefinitions;

namespace {

using GatherParams = std::tuple<
    std::vector<size_t>,                 // Indices shape
    std::pair<std::vector<size_t>, int>, // Input shapes and axis
    InferenceEngine::Precision           // Network precision
>;

const std::vector<std::vector<std::size_t>> indicesShapes = {
    {},
    {5},
    {10, 5},
    {1, 128, 1},
    {15, 4, 20, 5},
};

const std::vector<std::pair<std::vector<std::size_t>, int>> inputShapes = {
    {{6, 12, 10, 24}, -4},
    {{6, 12, 10, 24}, -3},

    {{3052, 768},     -2},
    {{6, 12, 10, 24}, -2},

    {{10},            -1},
    {{3052, 768},     -1},
    {{6, 12, 10, 24}, -1},

    {{10},             0},
    {{3052, 768},      0},
    {{6, 12, 10, 24},  0},

    {{3052, 768},      1},
    {{6, 12, 10, 24},  1},

    {{6, 12, 10, 24},  2},
    {{6, 12, 10, 24},  3},
};

const std::vector<InferenceEngine::Precision> networkPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
};

class MyriadGatherLayerTest : public testing::WithParamInterface<GatherParams>, public GatherLayerTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherParams>& obj) {
        std::vector<size_t> indicesShape;
        std::pair<std::vector<size_t>, int> inputShapeAndAxis;
        InferenceEngine::Precision netPrecision;
        std::tie(indicesShape, inputShapeAndAxis, netPrecision) = obj.param;
        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShapeAndAxis.first) << "_";
        result << "axis=" << inputShapeAndAxis.second << "_";
        result << "indicesShape=" << CommonTestUtils::vec2str(indicesShape) << "_";
        result << "IP=" << netPrecision;
        return result.str();
    }

protected:
    void SetUp() override {
        configuration[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
        GatherLayerTestBase::SetUp(generateParams(GetParam()));
    }

private:
    static gatherParamsTuple generateParams(const GatherParams& params) {
        const auto& indicesShape     = std::get<0>(params);
        const auto& inputShape       = std::get<1>(params).first;
        const auto& axis             = std::get<1>(params).second;
        const auto& networkPrecision = std::get<2>(params);
        const auto& inputPrecision   = InferenceEngine::Precision::UNSPECIFIED;
        const auto& outputPrecision  = InferenceEngine::Precision::UNSPECIFIED;
        const auto& inputLayout      = InferenceEngine::Layout::ANY;
        const auto& outputLayout     = InferenceEngine::Layout::ANY;

        return std::make_tuple(
            generateIndices(indicesShape, inputShape, axis),
            indicesShape,
            axis,
            inputShape,
            networkPrecision,
            inputPrecision,
            outputPrecision,
            inputLayout,
            outputLayout,
            CommonTestUtils::DEVICE_MYRIAD);
    }

    static std::vector<int> generateIndices(const std::vector<size_t>& indicesShape, const std::vector<size_t>& inputShape, int axis) {
        axis = axis < 0 ? axis + static_cast<int>(inputShape.size()) : axis;

        std::vector<int> indices(indicesShape.empty() ? 1 : CommonTestUtils::getTotal(indicesShape));
        CommonTestUtils::fill_data_random(indices.data(), indices.size(), inputShape[axis]);
        return indices;
    }
};

TEST_P(MyriadGatherLayerTest, accuracy) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Gather,
    MyriadGatherLayerTest,
    testing::Combine(
        testing::ValuesIn(indicesShapes),
        testing::ValuesIn(inputShapes),
        testing::ValuesIn(networkPrecisions)),
    MyriadGatherLayerTest::getTestCaseName);

}  // namespace
