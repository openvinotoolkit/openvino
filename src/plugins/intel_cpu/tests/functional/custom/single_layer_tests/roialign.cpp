// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
using ROIAlignShapes = std::vector<InputShape>;

using ROIAlignSpecificParams =  std::tuple<
        int,                                                 // bin's column count
        int,                                                 // bin's row count
        float,                                               // scale for given region considering actual input size
        int,                                                 // pooling ratio
        std::string,                                         // pooling mode
        std::string,                                         // aligned mode
        ROIAlignShapes
>;

using ROIAlignLayerTestParams = std::tuple<
        ROIAlignSpecificParams,
        ElementType,                    // Net precision
        ov::test::TargetDevice   // Device name
>;

using ROIAlignLayerCPUTestParamsSet = std::tuple<
        ROIAlignLayerTestParams,
        CPUSpecificParams>;

class ROIAlignLayerCPUTest : public testing::WithParamInterface<ROIAlignLayerCPUTestParamsSet>,
                             public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ROIAlignLayerCPUTestParamsSet> obj) {
        ROIAlignLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        ElementType netPrecision;
        ROIAlignSpecificParams roiPar;
        std::tie(roiPar, netPrecision, td) = basicParamsSet;

        int pooledH;
        int pooledW;
        float spatialScale;
        int samplingRatio;
        std::string mode;
        std::string alignedMode;
        ROIAlignShapes inputShapes;
        std::tie(pooledH, pooledW, spatialScale, samplingRatio, mode, alignedMode, inputShapes) = roiPar;
        std::ostringstream result;

        result << netPrecision << "_IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({ shape.first }) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            for (const auto& targetShape : shape.second) {
                result << ov::test::utils::vec2str(targetShape) << "_";
            }
            result << ")_";
        }

        result << "pooledH=" << pooledH << "_";
        result << "pooledW=" << pooledW << "_";
        result << "spatialScale=" << spatialScale << "_";
        result << "samplingRatio=" << samplingRatio << "_";
        result << mode << "_";
        result << alignedMode << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        ov::Tensor data_tensor;
        const auto& dataPrecision = funcInputs[0].get_element_type();
        const auto& dataShape = targetInputStaticShapes.front();
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = 10;
        in_data.resolution = 1000;
        data_tensor = ov::test::utils::create_and_fill_tensor(dataPrecision, dataShape, in_data);

        const auto& coordsET = funcInputs[1].get_element_type();
        auto coordsTensor = ov::Tensor{ coordsET, targetInputStaticShapes[1] };
        if (coordsET == ElementType::f32) {
            auto coordsTensorData = static_cast<float*>(coordsTensor.data());
            for (size_t i = 0; i < coordsTensor.get_size(); i += 4) {
                coordsTensorData[i] = 1.f;
                coordsTensorData[i + 1] = 1.f;
                coordsTensorData[i + 2] = 19.f;
                coordsTensorData[i + 3] = 19.f;
            }
        } else if (coordsET == ElementType::bf16) {
            auto coordsTensorData = static_cast<std::int16_t*>(coordsTensor.data());
            for (size_t i = 0; i < coordsTensor.get_size(); i += 4) {
                coordsTensorData[i] = static_cast<std::int16_t>(ov::bfloat16(1.f).to_bits());
                coordsTensorData[i + 1] = static_cast<std::int16_t>(ov::bfloat16(1.f).to_bits());
                coordsTensorData[i + 2] = static_cast<std::int16_t>(ov::bfloat16(19.f).to_bits());
                coordsTensorData[i + 3] = static_cast<std::int16_t>(ov::bfloat16(19.f).to_bits());
            }
        } else {
            OPENVINO_THROW("roi align. Unsupported precision: ", coordsET);
        }

        auto roisIdxTensor = ov::Tensor{ funcInputs[2].get_element_type(), targetInputStaticShapes[2] };
        auto roisIdxTensorData = static_cast<std::int32_t*>(roisIdxTensor.data());
        std::int32_t batchIdx = 0;
        for (size_t i = 0; i < roisIdxTensor.get_size(); i++) {
            roisIdxTensorData[i] = batchIdx;
            batchIdx = (batchIdx + 1) % targetInputStaticShapes[0][0];
        }

        inputs.insert({ funcInputs[0].get_node_shared_ptr(), data_tensor });
        inputs.insert({ funcInputs[1].get_node_shared_ptr(), coordsTensor });
        inputs.insert({ funcInputs[2].get_node_shared_ptr(), roisIdxTensor });
    }

    void SetUp() override {
        ROIAlignLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        ROIAlignSpecificParams roiAlignParams;
        ElementType inputPrecision;
        std::tie(roiAlignParams, inputPrecision, targetDevice) = basicParamsSet;

        int pooledH;
        int pooledW;
        float spatialScale;
        int samplingRatio;
        std::string mode;
        std::string alignedMode;
        ROIAlignShapes inputShapes;
        std::tie(pooledH, pooledW, spatialScale, samplingRatio, mode, alignedMode, inputShapes) = roiAlignParams;

        init_input_shapes(inputShapes);

        ov::ParameterVector float_params;
        for (auto&& shape : { inputDynamicShapes[0], inputDynamicShapes[1] }) {
            float_params.push_back(std::make_shared<ov::op::v0::Parameter>(inputPrecision, shape));
        }
        auto int_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes[2]);
        auto pooling_mode = ov::EnumNames<ov::op::v9::ROIAlign::PoolingMode>::as_enum(mode);
        auto aligned_mode = ov::EnumNames<ov::op::v9::ROIAlign::AlignedMode>::as_enum(alignedMode);

        auto roialign = std::make_shared<ov::op::v9::ROIAlign>(float_params[0],
                                                               float_params[1],
                                                               int_param,
                                                               pooledH,
                                                               pooledW,
                                                               samplingRatio,
                                                               spatialScale,
                                                               pooling_mode,
                                                               aligned_mode);

        selectedType = makeSelectedTypeStr(selectedType, inputPrecision);
        if (inputPrecision == ElementType::bf16) {
            rel_threshold = 1e-2;
        }

        ov::ParameterVector params{ float_params[0], float_params[1], int_param };
        function = makeNgraphFunction(inputPrecision, params, roialign, "ROIAlign");
    }
};

TEST_P(ROIAlignLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "ROIAlign");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (ov::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, nc, x}, {nchw}, {"jit_avx512"}, {"jit_avx512"}});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, nc, x}, {nhwc}, {"jit_avx512"}, {"jit_avx512"}});
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, nc, x}, {nChw16c}, {"jit_avx512"}, {"jit_avx512"}});
    } else if (ov::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, nc, x}, {nchw}, {"jit_avx2"}, {"jit_avx2"}});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, nc, x}, {nhwc}, {"jit_avx2"}, {"jit_avx2"}});
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, nc, x}, {nChw8c}, {"jit_avx2"}, {"jit_avx2"}});
    } else if (ov::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, nc, x}, {nchw}, {"jit_sse42"}, {"jit_sse42"}});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, nc, x}, {nhwc}, {"jit_sse42"}, {"jit_sse42"}});
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, nc, x}, {nChw8c}, {"jit_sse42"}, {"jit_sse42"}});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw, nc, x}, {nchw}, {"ref"}, {"ref"}});
    }
    return resCPUParams;
}

const std::vector<ElementType> netPrecisions = {
        ElementType::f32,
        ElementType::bf16
};

const std::vector<int> spatialBinXVector = { 2 };

const std::vector<int> spatialBinYVector = { 2 };

const std::vector<float> spatialScaleVector = { 1.0f };

const std::vector<int> poolingRatioVector = { 7 };

const std::vector<std::string> modeVector = {
        "avg",
        "max"
};

const std::vector<std::string> alignedModeVector = {
        "asymmetric",
        "half_pixel_for_nn",
        "half_pixel"
};

const std::vector<ROIAlignShapes> inputShapeVector = {
    ROIAlignShapes{{{}, {{ 2, 22, 20, 20 }}}, {{}, {{2, 4}}}, {{}, {{2}}}},
    ROIAlignShapes{{{}, {{ 2, 18, 20, 20 }}}, {{}, {{2, 4}}}, {{}, {{2}}}},
    ROIAlignShapes{{{}, {{ 2, 4, 20, 20 }}}, {{}, {{2, 4}}}, {{}, {{2}}}},
    ROIAlignShapes{{{}, {{ 2, 4, 20, 40 }}}, {{}, {{2, 4}}}, {{}, {{2}}}},
    ROIAlignShapes{{{}, {{ 10, 1, 20, 20 }}}, {{}, {{2, 4}}}, {{}, {{2}}}},
    ROIAlignShapes{{{}, {{ 2, 18, 20, 20 }}}, {{}, {{1, 4}}}, {{}, {{1}}}},
    ROIAlignShapes{{{}, {{ 2, 4, 20, 20 }}}, {{}, {{1, 4}}}, {{}, {{1}}}},
    ROIAlignShapes{{{}, {{ 2, 4, 20, 40 }}}, {{}, {{1, 4}}}, {{}, {{1}}}},
    ROIAlignShapes{{{}, {{ 10, 1, 20, 20 }}}, {{}, {{1, 4}}}, {{}, {{1}}}},
    ROIAlignShapes{
        {{-1, -1, -1, -1}, {{ 10, 1, 20, 20 }, { 2, 4, 20, 20 }, { 2, 18, 20, 20 }}},
        {{-1, 4}, {{1, 4}, {2, 4}, {1, 4}}},
        {{-1}, {{1}, {2}, {1}}}
    },
    ROIAlignShapes{
        {{{2, 10}, { 1, 5 }, -1, -1}, {{ 2, 1, 20, 20 }, { 10, 5, 30, 20 }, { 4, 4, 40, 40 }}},
        {{-1, 4}, {{2, 4}, {2, 4}, {1, 4}}},
        {{-1}, {{2}, {2}, {1}}}
    },
    ROIAlignShapes{
        {{{2, 10}, {1, 18}, {10, 30}, {15, 25}}, {{ 10, 1, 10, 15 }, { 2, 4, 20, 20 }, { 7, 18, 30, 25 }}},
        {{{1, 2}, 4}, {{1, 4}, {2, 4}, {1, 4}}},
        {{{1, 2}}, {{1}, {2}, {1}}}
    },
};

const auto roiAlignParams = ::testing::Combine(
        ::testing::ValuesIn(spatialBinXVector),       // bin's column count
        ::testing::ValuesIn(spatialBinYVector),       // bin's row count
        ::testing::ValuesIn(spatialScaleVector),      // scale for given region considering actual input size
        ::testing::ValuesIn(poolingRatioVector),      // pooling ratio for bin
        ::testing::ValuesIn(modeVector),              // pooling mode
        ::testing::ValuesIn(alignedModeVector),       // aligned mode
        ::testing::ValuesIn(inputShapeVector)         // feature map shape
);

INSTANTIATE_TEST_SUITE_P(smoke_ROIAlignLayoutTest, ROIAlignLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        roiAlignParams,
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::ValuesIn(filterCPUInfoForDevice())),
                ROIAlignLayerCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
