// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "openvino/util/common_util.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace test {
typedef std::tuple<Shape,                                              // Input shape
                   element::Type,                                      // Input precision
                   std::pair<std::vector<float>, std::vector<float>>,  // ScaleShift scales and shifts
                   std::vector<std::vector<float>>,                    // Quantize intervals
                   std::string                                         // Device name
                   >
    FuseScaleShiftAndQuantizeTuple;

class FuseScaleShiftAndFakeQuantizeTest : public testing::WithParamInterface<FuseScaleShiftAndQuantizeTuple>,
                                          virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FuseScaleShiftAndQuantizeTuple>& obj) {
        Shape inputShape;
        element::Type inputPrecision;
        std::pair<std::vector<float>, std::vector<float>> scaleShift;
        std::vector<std::vector<float>> quantizeIntervals;
        std::string targetName;
        std::tie(inputShape, inputPrecision, scaleShift, quantizeIntervals, targetName) = obj.param;
        std::ostringstream results;

        results << "IS=" << inputShape << "_InPRC=" << inputPrecision
                << "_Scale=" << ov::util::vector_to_string(scaleShift.first)
                << "_Shift=" << ov::util::vector_to_string(scaleShift.second) << "_Intervals=";
        for (const auto& vecInt : quantizeIntervals) {
            results << ov::util::vector_to_string(vecInt) << ",";
        }

        results << "targetDevice=" << targetName;

        return results.str();
    }

protected:
    void SetUp() override {
        Shape inputShape;
        element::Type inputPrecision;
        std::pair<std::vector<float>, std::vector<float>> scaleShift;
        std::vector<std::vector<float>> quantizeIntervals;
        std::tie(inputShape, inputPrecision, scaleShift, quantizeIntervals, targetDevice) = this->GetParam();

        const auto param = std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputShape);
        Shape constShape = Shape(inputShape.size(), 1);
        constShape[1] = scaleShift.second.size();
        const auto subtract = std::make_shared<ov::op::v1::Subtract>(
            param,
            std::make_shared<ov::op::v0::Constant>(inputPrecision, constShape, scaleShift.second));
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(
            param,
            std::make_shared<ov::op::v0::Constant>(inputPrecision, constShape, scaleShift.first));
        Shape inConstShape = Shape(inputShape.size(), 1);
        inConstShape[1] = quantizeIntervals[0].size();
        const auto quantize = ov::test::utils::make_fake_quantize(multiply,
                                                                inputPrecision,
                                                                256,
                                                                inConstShape,
                                                                quantizeIntervals[0],
                                                                quantizeIntervals[1],
                                                                quantizeIntervals[2],
                                                                quantizeIntervals[3]);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(quantize)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "FuseScaleShiftAndQuantize");
        if (inputPrecision == element::f32) {
            abs_threshold = 2e-7;
        }
    }
};

TEST_P(FuseScaleShiftAndFakeQuantizeTest, CompareWithRefs) {
    run();
}

namespace {
std::vector<Shape> inputShapes{{1, 4, 16, 16},
                               {8, 4, 16, 16},
                               {1, 4, 16, 16, 16},
                               {8, 4, 16, 16, 16},
                               {1, 4, 16, 16, 16, 16},
                               {8, 4, 16, 16, 16, 16}};

std::vector<std::pair<std::vector<float>, std::vector<float>>> scaleShifts{
    {{30.f}, {17.f}},       // actually fused in LPT
    {{-30.f}, {0.f}},       // fused with crop bound invert
    {{-17.f}, {12.f}},      // fused with crop bound invert
    {{-1.23e-44f}, {0.f}},  // fused with denormal handling
    {{0.f}, {0.f}},         // not fused
    {{0.f}, {18.f}},        // not fused
};

std::vector<std::vector<std::vector<float>>> quantizes{
    {{-1.f}, {5.f}, {-5.f}, {1.f}},
    {{2.f}, {4.f}, {-4.f}, {-2.f}},
    {{-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
    {{0.f}, {2.55f}, {0.f}, {2.55f}},
};

INSTANTIATE_TEST_SUITE_P(smoke_FuseScaleShiftAndFakeQuantize,
                         FuseScaleShiftAndFakeQuantizeTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(scaleShifts),
                                            ::testing::ValuesIn(quantizes),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         FuseScaleShiftAndFakeQuantizeTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
