// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace InferenceEngine;
using namespace ngraph;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using fqSpecificParams = std::tuple<int64_t,                  // 'data' input low bounds
                                    int64_t,                  // 'data' input high bounds
                                    std::vector<float>,       // output low
                                    std::vector<float>,       // output high
                                    std::vector<SizeVector>,  // 'range' inputs shapes
                                    size_t>;                  // levels

using fqLayerTestParamsSet = std::tuple<fqSpecificParams,
                                        SizeVector,                                        // 'data' input shape
                                        Precision,                                         // input precision
                                        std::pair<std::vector<float>, std::vector<float>>, // il and ih values
                                        bool,                                              // should be decomposed
                                        CPUSpecificParams>;

class FakeQuantizeLayerCPUTest : public testing::WithParamInterface<fqLayerTestParamsSet>,
                                 virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<fqLayerTestParamsSet> obj) {
        fqSpecificParams fqParams;
        SizeVector inDataShape;
        Precision inPrec;
        std::pair<std::vector<float>, std::vector<float>> inputRangesValues;
        bool shouldBeDecomposed;
        CPUSpecificParams cpuParams;
        std::tie(fqParams, inDataShape, inPrec, inputRangesValues, shouldBeDecomposed, cpuParams) = obj.param;

        int64_t inDataLowBounds, inDataHighBounds;
        std::vector<float> inputLow, inputHigh, outputLow, outputHigh;
        std::vector<SizeVector> inRangesShapes;
        size_t levels;
        inputLow = inputRangesValues.first;
        inputHigh = inputRangesValues.second;
        std::tie(inDataLowBounds, inDataHighBounds, outputLow, outputHigh, inRangesShapes, levels) = fqParams;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inDataShape) << "_";
        result << "inPrec=" << inPrec.name() << "_";

        std::string rs = "";
        for (size_t i = 0; i < inRangesShapes.size(); i++) {
            rs += CommonTestUtils::vec2str(inRangesShapes[i]) + "_";
        }
        result << "RS=" << rs;
        result << "LOW_BOUNDS=" << inDataLowBounds << "_";
        result << "HIGH_BOUNDS=" << inDataHighBounds << "_";
        result << "IL=" << CommonTestUtils::vec2str(inputLow) << "_";
        result << "IH=" << CommonTestUtils::vec2str(inputHigh) << "_";
        result << "OL=" << CommonTestUtils::vec2str(outputLow) << "_";
        result << "OH=" << CommonTestUtils::vec2str(outputHigh) << "_";
        result << "LEVELS=" << levels;

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

    void Infer() override {
        inferRequest = executableNetwork.CreateInferRequest();
        inputs.clear();

        const InputsDataMap &inDataMap = cnnNetwork.getInputsInfo();
        auto input = inDataMap.begin();

        Blob::Ptr blob = FuncTestUtils::createAndFillBlob(input->second->getTensorDesc(), inDataHighBounds - inDataLowBounds, inDataLowBounds);
        inferRequest.SetBlob(input->second->name(), blob);
        inputs.push_back(blob);

        inferRequest.Infer();
    }

protected:
    std::string layerName;

    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        fqSpecificParams fqParams;
        SizeVector inDataShape;
        Precision inPrec;
        std::pair<std::vector<float>, std::vector<float>> inputRangesValues;
        bool shouldBeDecomposed;
        CPUSpecificParams cpuParams;
        std::tie(fqParams, inDataShape, inPrec, inputRangesValues, shouldBeDecomposed, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::vector<SizeVector> inRangesShapes;
        size_t levels;
        std::vector<std::vector<float>> rangesBounds(RANGES_INPUT_NUMBER);
        rangesBounds[0] = inputRangesValues.first;
        rangesBounds[1] = inputRangesValues.second;
        std::tie(inDataLowBounds, inDataHighBounds, rangesBounds[2], rangesBounds[3], inRangesShapes, levels) = fqParams;

        auto ngInPrec = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
        ParameterVector params = builder::makeParams(ngInPrec, {inDataShape});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<opset5::Parameter>(params));

        auto il = builder::makeConstant(ngInPrec, inRangesShapes[0], rangesBounds[0], rangesBounds[0].empty());
        auto ih = builder::makeConstant(ngInPrec, inRangesShapes[1], rangesBounds[1], rangesBounds[1].empty());
        auto ol = builder::makeConstant(ngInPrec, inRangesShapes[2], rangesBounds[2], rangesBounds[2].empty());
        auto oh = builder::makeConstant(ngInPrec, inRangesShapes[3], rangesBounds[3], rangesBounds[3].empty());
        auto fq = std::make_shared<opset5::FakeQuantize>(paramOuts[0], il, ih, ol, oh, levels);

        layerName = shouldBeDecomposed ? "" : "FakeQuantize";

        if (selectedType.empty()) {
           selectedType = getPrimitiveType() + "_" + inPrec.name();
        }

        fq->get_rt_info() = getCPUInfo();

        function = std::make_shared<Function>(fq, params, "FakeQuantizeCPU");
    }

private:
    const size_t RANGES_INPUT_NUMBER = 4;

    int64_t inDataLowBounds, inDataHighBounds;
};

TEST_P(FakeQuantizeLayerCPUTest, CompareWithRefs) {
    Run();

    CheckPluginRelatedResults(executableNetwork, layerName);
}


const std::vector<size_t> levels = {16, 255, 256};

int64_t dataLowBounds{-10}, dataHighBounds{10};

const std::vector<std::pair<std::vector<float>, std::vector<float>>> input_ranges = {
    {{0.0f}, {5.f}},
    {{-10.0f}, {-5.f}}
};

const std::vector<float> outputLow{5.0f}, outputHigh{25.0f};

namespace fqImpl {

std::vector<CPUSpecificParams> memForm4D_jit = {
        CPUSpecificParams({nchw}, {nchw}, {}, {}),
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {})
};

const std::vector<std::vector<SizeVector>> rangesShapes4D_jit = {
    {{1, 5, 1, 1}, {1, 5, 1, 1}, {1, 5, 1, 1}, {1, 5, 1, 1}},
    {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}
};

const auto specificParams4D_jit = ::testing::Combine(::testing::Values(dataLowBounds),
                                                     ::testing::Values(dataHighBounds),
                                                     ::testing::Values(outputLow),
                                                     ::testing::Values(outputHigh),
                                                     ::testing::ValuesIn(rangesShapes4D_jit),
                                                     ::testing::ValuesIn(levels));
const auto testParams4D_jit = ::testing::Combine(specificParams4D_jit,
                                                 ::testing::Values(SizeVector{4, 5, 6, 7}),
                                                 ::testing::Values(Precision::FP32),
                                                 ::testing::ValuesIn(input_ranges),
                                                 ::testing::Values(false),
                                                 ::testing::ValuesIn(filterCPUSpecificParams(memForm4D_jit)));
INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizeLayerCPUTest_4D_jit, FakeQuantizeLayerCPUTest, testParams4D_jit, FakeQuantizeLayerCPUTest::getTestCaseName);


std::vector<CPUSpecificParams> memForm4D_ref = {
        CPUSpecificParams({nchw}, {nchw}, {"ref_FP32"}, {"ref_FP32"})
};

const std::vector<std::vector<SizeVector>> rangesShapes4D_ref = {
    {{4, 1, 1, 1}, {4, 1, 1, 1}, {4, 1, 1, 1}, {4, 1, 1, 1}}
};

const auto specificParams4D_ref = ::testing::Combine(::testing::Values(dataLowBounds),
                                                     ::testing::Values(dataHighBounds),
                                                     ::testing::Values(outputLow),
                                                     ::testing::Values(outputHigh),
                                                     ::testing::ValuesIn(rangesShapes4D_ref),
                                                     ::testing::ValuesIn(levels));
const auto testParams4D_ref = ::testing::Combine(specificParams4D_ref,
                                                 ::testing::Values(SizeVector{4, 5, 6, 7}),
                                                 ::testing::Values(Precision::FP32),
                                                 ::testing::ValuesIn(input_ranges),
                                                 ::testing::Values(false),
                                                 ::testing::ValuesIn(memForm4D_ref));
INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizeLayerCPUTest_4D_ref, FakeQuantizeLayerCPUTest, testParams4D_ref, FakeQuantizeLayerCPUTest::getTestCaseName);


std::vector<CPUSpecificParams> memForm5D_jit = {
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {}),
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {})
};

const std::vector<std::vector<SizeVector>> rangesShapes5D_jit = {
    {{1, 4, 1, 1, 1}, {1, 4, 1, 1, 1}, {1, 4, 1, 1, 1}, {1, 4, 1, 1, 1}},
    {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}
};

const auto specificParams5D_jit = ::testing::Combine(::testing::Values(dataLowBounds),
                                                     ::testing::Values(dataHighBounds),
                                                     ::testing::Values(outputLow),
                                                     ::testing::Values(outputHigh),
                                                     ::testing::ValuesIn(rangesShapes5D_jit),
                                                     ::testing::ValuesIn(levels));
const auto testParams5D_jit = ::testing::Combine(specificParams5D_jit,
                                                 ::testing::Values(SizeVector{3, 4, 5, 6, 7}),
                                                 ::testing::Values(Precision::FP32),
                                                 ::testing::ValuesIn(input_ranges),
                                                 ::testing::Values(false),
                                                 ::testing::ValuesIn(filterCPUSpecificParams(memForm5D_jit)));

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizeLayerCPUTest_5D_jit, FakeQuantizeLayerCPUTest, testParams5D_jit, FakeQuantizeLayerCPUTest::getTestCaseName);


std::vector<CPUSpecificParams> memForm5D_ref = {
        CPUSpecificParams({ncdhw}, {ncdhw}, {"ref_FP32"}, {"ref_FP32"})
};

const std::vector<std::vector<SizeVector>> rangesShapes5D_ref = {
    {{3, 1, 1, 1, 1}, {3, 1, 1, 1, 1}, {3, 1, 1, 1, 1}, {3, 1, 1, 1, 1}}
};

const auto specificParams5D_ref = ::testing::Combine(::testing::Values(dataLowBounds),
                                                     ::testing::Values(dataHighBounds),
                                                     ::testing::Values(outputLow),
                                                     ::testing::Values(outputHigh),
                                                     ::testing::ValuesIn(rangesShapes5D_ref),
                                                     ::testing::ValuesIn(levels));
const auto testParams5D_ref = ::testing::Combine(specificParams5D_ref,
                                                 ::testing::Values(SizeVector{3, 4, 5, 6, 7}),
                                                 ::testing::Values(Precision::FP32),
                                                 ::testing::ValuesIn(input_ranges),
                                                 ::testing::Values(false),
                                                 ::testing::ValuesIn(memForm5D_ref));

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizeLayerCPUTest_5D_ref, FakeQuantizeLayerCPUTest, testParams5D_ref, FakeQuantizeLayerCPUTest::getTestCaseName);

} // namespace fqImpl

const std::vector<SizeVector> dataShapes = {
    {4, 5, 6, 7},
    {3, 4, 5, 6, 7},
    {2, 3, 4, 5, 6, 7},
};

const std::vector<std::vector<SizeVector>> rangesShapes = {
    {{4, 5, 6, 7}, {4, 5, 6, 7}, {4, 5, 6, 7}, {4, 5, 6, 7}},
    {{1, 5, 1, 1}, {1, 1, 6, 7}, {1, 1, 6, 7}, {1, 1, 6, 7}},
    {{1, 1, 6, 7}, {1, 1, 6, 7}, {1, 1, 6, 7}, {1, 1, 6, 7}},
    {{1, 1, 6, 7}, {1, 1, 6, 7}, {1, 1, 1, 1}, {1, 1, 1, 1}},
    {{1, 1, 6, 1}, {1, 5, 6, 7}, {1, 1, 6, 1}, {1, 1, 6, 1}}
};

namespace fqDecompos {

const auto specificParams = ::testing::Combine(::testing::Values(dataLowBounds),
                                               ::testing::Values(dataHighBounds),
                                               ::testing::Values(outputLow),
                                               ::testing::Values(outputHigh),
                                               ::testing::ValuesIn(rangesShapes),
                                               ::testing::ValuesIn(levels));
const auto testParams = ::testing::Combine(specificParams,
                                           ::testing::ValuesIn(dataShapes),
                                           ::testing::Values(Precision::FP32),
                                           ::testing::ValuesIn(input_ranges),
                                           ::testing::Values(true),
                                           ::testing::Values(CPUSpecificParams{}));

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizeLayerCPUTest_Decompos, FakeQuantizeLayerCPUTest, testParams, FakeQuantizeLayerCPUTest::getTestCaseName);

} // namespace fqDecompos

} // namespace CPULayerTestsDefinitions
