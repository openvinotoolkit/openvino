// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {
using extractImagePatchesParams = typename std::tuple<InputShape,        // input shape
                                                      ElementType,       // Network precision
                                                      ov::Shape,         // kernel size
                                                      ov::Strides,       // strides
                                                      ov::Shape,         // rates
                                                      ov::op::PadType>;  // pad type

class ExtractImagePatchesLayerCPUTest : public testing::WithParamInterface<extractImagePatchesParams>,
                                        virtual public SubgraphBaseTest,
                                        public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<extractImagePatchesParams> obj) {
        InputShape inputShapes;
        ElementType inputPrecision;
        ov::Shape kernelSize;
        ov::Strides strides;
        ov::Shape rates;
        ov::op::PadType padType;
        std::tie(inputShapes, inputPrecision, kernelSize, strides, rates, padType) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << inputPrecision << "_"
               << "IS=" << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
        result << "TS=";
        result << "(";
        for (const auto& targetShape : inputShapes.second) {
            result << ov::test::utils::vec2str(targetShape) << "_";
        }

        result << ")_"
               << "kernelSize=" << kernelSize << "_strides=" << strides << "_rates=" << rates << "_padType=" << padType;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        InputShape inputShapes;
        ElementType inputPrecision;
        ov::Shape kernelSize;
        ov::Strides strides;
        ov::Shape rates;
        ov::op::PadType padType;
        std::tie(inputShapes, inputPrecision, kernelSize, strides, rates, padType) = this->GetParam();

        selectedType = makeSelectedTypeStr("ref_any", inputPrecision);
        if (inputPrecision == ElementType::bf16) {
            rel_threshold = 1e-2;
        }

        init_input_shapes({inputShapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inputPrecision, shape));
        }
        auto extImgPatches =
            std::make_shared<ov::op::v3::ExtractImagePatches>(params[0], kernelSize, strides, rates, padType);
        function = makeNgraphFunction(inputPrecision, params, extImgPatches, "ExtractImagePatches");
    }
};

TEST_P(ExtractImagePatchesLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "ExtractImagePatches");
}

namespace {
const std::vector<InputShape> inputShapes = {
    InputShape{{}, {{2, 3, 13, 37}}},
    InputShape{// dynamic
               {-1, -1, -1, -1},
               // static
               {{2, 3, 13, 37}, {6, 4, 14, 14}, {8, 12, 15, 16}, {2, 3, 13, 37}}},
    InputShape{// dynamic
               {{5, 15}, {6, 17}, {10, 15}, {13, 16}},
               // static
               {{5, 17, 10, 15}, {15, 10, 12, 13}, {10, 10, 15, 16}, {5, 17, 10, 15}}},
};

const std::vector<ElementType> inputPrecisions = {ElementType::i8, ElementType::bf16, ElementType::f32};

const std::vector<ov::Shape> kSizes = {{1, 5}, {3, 4}, {3, 1}};

const std::vector<ov::Strides> strides = {{1, 2}, {2, 2}, {2, 1}};

const std::vector<ov::Shape> rates = {{1, 3}, {3, 3}, {3, 1}};

const std::vector<ov::op::PadType> autoPads = {ov::op::PadType::VALID,
                                               ov::op::PadType::SAME_UPPER,
                                               ov::op::PadType::SAME_LOWER};

const auto params = ::testing::Combine(::testing::ValuesIn(inputShapes),
                                       ::testing::ValuesIn(inputPrecisions),
                                       ::testing::ValuesIn(kSizes),
                                       ::testing::ValuesIn(strides),
                                       ::testing::ValuesIn(rates),
                                       ::testing::ValuesIn(autoPads));

INSTANTIATE_TEST_SUITE_P(smoke_ExtractImagePatches_CPU,
                         ExtractImagePatchesLayerCPUTest,
                         params,
                         ExtractImagePatchesLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
