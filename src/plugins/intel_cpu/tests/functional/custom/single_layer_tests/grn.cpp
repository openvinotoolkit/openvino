// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {

using GRNCPUTestParams = typename std::tuple<ov::element::Type,  // Network type
                                             ov::element::Type,  // Input type
                                             ov::element::Type,  // Output type
                                             InputShape,         // Input shape
                                             float,              // Bias
                                             std::string>;       // Device name

class GRNLayerCPUTest : public testing::WithParamInterface<GRNCPUTestParams>,
                        virtual public SubgraphBaseTest,
                        public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GRNCPUTestParams> obj) {
        ov::element::Type netPrecision;
        ov::element::Type inPrc, outPrc;
        InputShape inputShape;
        float bias;
        std::string targetDevice;

        std::tie(netPrecision, inPrc, outPrc, inputShape, bias, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=";
        for (const auto& item : inputShape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "netPRC=" << netPrecision.get_type_name() << "_";
        result << "inPRC=" << inPrc.get_type_name() << "_";
        result << "outPRC=" << outPrc.get_type_name() << "_";
        result << "bias=" << bias << "_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        ov::element::Type netPrecision;
        ov::element::Type inPrc, outPrc;
        InputShape inputShape;
        float bias;

        std::tie(netPrecision, inPrc, outPrc, inputShape, bias, targetDevice) = GetParam();

        init_input_shapes({inputShape});

        ov::ParameterVector paramsIn;
        for (auto&& shape : inputDynamicShapes)
            paramsIn.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));

        const auto grn = std::make_shared<ov::op::v0::GRN>(paramsIn[0], bias);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(grn)};
        function = std::make_shared<ov::Model>(results, paramsIn, "Grn");
    }
};

TEST_P(GRNLayerCPUTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<ov::element::Type> netPrecisions = {ov::element::bf16, ov::element::f16, ov::element::f32};

const std::vector<float> biases = {1e-6f, 0.33f, 1.1f, 2.25f, 100.25f};

const std::vector<InputShape> dataInputStaticShapes = {{{}, {{16, 24}}}, {{}, {{3, 16, 24}}}, {{}, {{1, 3, 30, 30}}}};

const std::vector<InputShape> dataInputDynamicShapes = {{{-1, -1}, {{5, 17}, {10, 3}}},
                                                        {{3, {10, 12}, -1}, {{3, 12, 25}, {3, 10, 10}}},
                                                        {{2, -1, -1, {5, 10}}, {{2, 17, 20, 7}, {2, 10, 12, 5}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GRNCPUStatic,
                         GRNLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::ValuesIn(dataInputStaticShapes),
                                            ::testing::ValuesIn(biases),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GRNLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GRNCPUDynamic,
                         GRNLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::ValuesIn(dataInputDynamicShapes),
                                            ::testing::ValuesIn(biases),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GRNLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
