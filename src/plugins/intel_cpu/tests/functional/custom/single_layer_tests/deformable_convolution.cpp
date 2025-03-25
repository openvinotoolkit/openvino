// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {
enum OffsetType { ZERO, NATURAL, REAL_POSITIVE, REAL_NEGATIVE, REAL_MISC };

typedef std::tuple<bool,       // with_bilinear_interpolation_pad
                   bool,       // with_modulation
                   OffsetType  // type of def. offsets
                   >
    DefConvSpecificParams;

typedef std::tuple<ov::op::PadType,         // pad. type
                   std::vector<ptrdiff_t>,  // pad. begin
                   std::vector<ptrdiff_t>,  // pad. end
                   std::vector<size_t>,     // strides
                   std::vector<size_t>      // dilations
                   >
    AddSpatialParamsDyn;

typedef std::tuple<AddSpatialParamsDyn,
                   std::vector<InputShape>,
                   DefConvSpecificParams,
                   ov::element::Type,  // Net precision
                   std::string         // Device name
                   >
    DefConvLayerTestParams;

typedef std::tuple<DefConvLayerTestParams, CPUSpecificParams> DefConvLayerCPUTestParamsSet;

class DefConvLayerCPUTest : public testing::WithParamInterface<DefConvLayerCPUTestParamsSet>,
                            virtual public SubgraphBaseTest,
                            public CPUTestsBase {
public:
    OffsetType offsetType;
    static std::string getTestCaseName(testing::TestParamInfo<DefConvLayerCPUTestParamsSet> obj) {
        DefConvSpecificParams dcSpecificParams;
        std::vector<InputShape> inputShape;
        ov::element::Type netPrecision;
        DefConvLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        AddSpatialParamsDyn addSpParams;
        std::string td;
        std::tie(addSpParams, inputShape, dcSpecificParams, netPrecision, td) = basicParamsSet;

        ov::op::PadType padType;
        std::vector<size_t> stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        std::tie(padType, padBegin, padEnd, stride, dilation) = addSpParams;

        // gr * in_ch_per_gr / in_ch_per_gr
        size_t groups = inputShape[0].second[0][1] / inputShape[2].second[0][1];
        // dg * ker_spat_shape[0] * ker_spat_shape[1] * 2 / (ker_spat_shape[0] * ker_spat_shape[1] * 2)
        size_t deformableGroups =
            inputShape[1].second[0][1] / (inputShape[2].second[0][2] * inputShape[2].second[0][3] * 2);

        bool withBilinearInterpolationPad, withModulation;
        OffsetType offType;
        std::tie(withBilinearInterpolationPad, withModulation, offType) = dcSpecificParams;
        std::ostringstream result;
        result << "DefConvTest(";
        result << std::to_string(obj.index) << ")_";
        result << "IS=" << ov::test::utils::vec2str(inputShape[0].second) << "_";
        result << "OS=" << ov::test::utils::vec2str(inputShape[1].second) << "_";
        result << "FS=" << ov::test::utils::vec2str(inputShape[2].second) << "_";
        if (withModulation) {
            result << "MS=" << ov::test::utils::vec2str(inputShape[3].second) << "_";
        }
        result << "G=" << groups << "_";
        result << "DG=" << deformableGroups << "_";
        result << "S=" << ov::test::utils::vec2str(stride) << "_";
        result << "PB=" << ov::test::utils::vec2str(padBegin) << "_";
        result << "PE=" << ov::test::utils::vec2str(padEnd) << "_";
        result << "D=" << ov::test::utils::vec2str(dilation) << "_";
        result << "AP=" << padType << "_";
        result << "netPRC=" << netPrecision.get_type_name() << "_";
        result << "withBilPad=" << withBilinearInterpolationPad << "_";
        result << "withMod=" << withModulation << "_";
        result << "offsetType=" << offType << "_";
        result << "trgDev=" << td;
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        auto inShape = targetInputStaticShapes[0];
        auto offShape = targetInputStaticShapes[1];
        auto filtShape = targetInputStaticShapes[2];

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            ov::test::utils::InputGenerateData in_data;
            if (i == 0) {  // "a_data"
                in_data.start_from = -1;
                in_data.range = 2;
                in_data.resolution = 100;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), inShape, in_data);
            } else if (i == 1) {  // "b_offset_vals"
                if (offsetType == OffsetType::NATURAL) {
                    in_data.start_from = 0;
                    in_data.range = 10;
                    in_data.resolution = 1;
                } else if (offsetType == OffsetType::ZERO) {
                    in_data.start_from = 0;
                    in_data.range = 1;
                    in_data.resolution = 1;
                } else if (offsetType == OffsetType::REAL_POSITIVE) {
                    in_data.start_from = 0;
                    in_data.range = 2;
                    in_data.resolution = 100;
                } else if (offsetType == OffsetType::REAL_NEGATIVE) {
                    in_data.start_from = -2;
                    in_data.range = 2;
                    in_data.resolution = 100;
                } else if (offsetType == OffsetType::REAL_MISC) {
                    in_data.start_from = -2;
                    in_data.range = 4;
                    in_data.resolution = 100;
                } else {
                    OPENVINO_THROW("Unexpected offset type");
                }
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), offShape, in_data);
            } else if (i == 2) {  // "c_filter_vals"
                in_data.start_from = -1;
                in_data.range = 2;
                in_data.resolution = 100;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), filtShape, in_data);
            } else if (i == 3) {  // "c_modulation_scalars"
                auto modShape = targetInputStaticShapes[3];
                in_data.start_from = -1;
                in_data.range = 2;
                in_data.resolution = 100;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), modShape, in_data);
            } else {
                OPENVINO_THROW("Unknown input of DeformableConvolution");
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
    void SetUp() override {
        DefConvSpecificParams dcSpecificParams;
        std::vector<InputShape> inputShape;
        ov::element::Type netPrecision;
        DefConvLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        AddSpatialParamsDyn addSpParams;
        std::tie(addSpParams, inputShape, dcSpecificParams, netPrecision, targetDevice) = basicParamsSet;
        init_input_shapes(inputShape);

        ov::op::PadType padType;
        std::vector<size_t> stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        std::tie(padType, padBegin, padEnd, stride, dilation) = addSpParams;

        // gr * in_ch_per_gr / in_ch_per_gr
        size_t groups = inputShape[0].second[0].at(1) / inputShape[2].second[0].at(1);
        // dg * ker_spat_shape[0] * ker_spat_shape[1] * 2 / (ker_spat_shape[0] * ker_spat_shape[1] * 2)
        size_t deformableGroups =
            inputShape[1].second[0].at(1) / (inputShape[2].second[0].at(2) * inputShape[2].second[0].at(3) * 2);
        bool withBilinearInterpolationPad, withModulation;
        std::tie(withBilinearInterpolationPad, withModulation, offsetType) = dcSpecificParams;
        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));
        }
        auto data = inputParams[0];
        data->set_friendly_name("a_data");
        auto offset_vals = inputParams[1];
        offset_vals->set_friendly_name("b_offset_vals");
        auto filter_vals = inputParams[2];
        filter_vals->set_friendly_name("c_filter_vals");
        ov::ParameterVector parameters{data, offset_vals, filter_vals};
        std::shared_ptr<ov::Node> deformable_conv;
        if (withModulation) {
            auto modulation_scalars = inputParams[3];
            modulation_scalars->set_friendly_name("c_modulation_scalars");

            deformable_conv = std::make_shared<ov::op::v8::DeformableConvolution>(data,
                                                                                  offset_vals,
                                                                                  filter_vals,
                                                                                  modulation_scalars,
                                                                                  stride,
                                                                                  padBegin,
                                                                                  padEnd,
                                                                                  dilation,
                                                                                  padType,
                                                                                  groups,
                                                                                  deformableGroups,
                                                                                  withBilinearInterpolationPad);
            parameters.push_back(modulation_scalars);
        } else {
            deformable_conv = std::make_shared<ov::op::v8::DeformableConvolution>(data,
                                                                                  offset_vals,
                                                                                  filter_vals,
                                                                                  stride,
                                                                                  padBegin,
                                                                                  padEnd,
                                                                                  dilation,
                                                                                  padType,
                                                                                  groups,
                                                                                  deformableGroups,
                                                                                  withBilinearInterpolationPad);
        }

        function = makeNgraphFunction(netPrecision, parameters, deformable_conv, "deformable_convolution");

        if (netPrecision == ov::element::f32) {
            abs_threshold = 5e-6;
        }
    }
};

TEST_P(DefConvLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "DeformableConvolution");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice(bool enforceRef = false) {
    std::vector<CPUSpecificParams> resCPUParams;
    if (enforceRef) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {}, {"ref_FP32"}});
    } else if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {}, {"jit_avx512_FP32"}});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {}, {"jit_avx2_FP32"}});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {}, {"jit_sse42"}});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {}, {"ref_FP32"}});
    }
    return resCPUParams;
}

const std::vector<ov::element::Type> netPrecisions = {ov::element::f32};

const auto defConvSpecificParams_Smoke =
    ::testing::Combine(::testing::ValuesIn(std::vector<bool>{true, false}),  // with_bilinear_interpolation_pad
                       ::testing::ValuesIn(std::vector<bool>{true, false}),  // with_modulation
                       ::testing::ValuesIn(std::vector<OffsetType>{
                           OffsetType::REAL_MISC,  // offset type
                       }));

const auto defConvSpecificParams =
    ::testing::Combine(::testing::ValuesIn(std::vector<bool>{true, false}),  // with_bilinear_interpolation_pad
                       ::testing::ValuesIn(std::vector<bool>{true, false}),  // with_modulation
                       ::testing::ValuesIn(std::vector<OffsetType>{OffsetType::NATURAL,  // offset type
                                                                   OffsetType::ZERO,
                                                                   OffsetType::REAL_MISC,
                                                                   OffsetType::REAL_POSITIVE,
                                                                   OffsetType::REAL_NEGATIVE}));

std::vector<ov::op::PadType> padTypes = {ov::op::PadType::EXPLICIT, ov::op::PadType::VALID};
std::vector<std::vector<size_t>> getCartProduct(const std::vector<std::vector<size_t>>& v) {
    int outSize = 1;
    int n = v.size();
    for (int i = 0; i < n; i++) {
        outSize *= v[i].size();
    }
    std::vector<std::vector<size_t>> res(outSize);
    for (int i = 0; i < outSize; i++) {
        std::vector<size_t> cortege(n);
        int curResid = i, curInd = 0;
        for (int j = v.size() - 1; j >= 0; j--) {
            curInd = curResid % v[j].size();
            curResid = curResid / v[j].size();
            cortege[j] = v[j][curInd];
        }
        res[i] = cortege;
    }
    return res;
}
std::vector<std::vector<ov::Shape>> buildStaticParams(const std::vector<std::vector<size_t>> spatParams,
                                                      const std::vector<std::vector<size_t>> chParamsUncombined) {
    std::vector<std::vector<size_t>> chParams = getCartProduct(chParamsUncombined);
    std::vector<std::vector<ov::Shape>> shapes;
    for (std::vector<size_t>& chPar : chParams) {
        const size_t batch = spatParams[0][0];
        const size_t inSpH = spatParams[1][0];
        const size_t inSpW = spatParams[1][1];
        const size_t offSpH = spatParams[2][0];
        const size_t offSpW = spatParams[2][1];
        const size_t kerSpH = spatParams[3][0];
        const size_t kerSpW = spatParams[3][1];

        const size_t gr = chPar[0];
        const size_t defGr = chPar[1];
        const size_t inChPerGr = chPar[2];
        const size_t outChPerGr = chPar[3];

        std::vector<ov::Shape> inputShape = {{batch, gr * inChPerGr, inSpH, inSpW},
                                             {batch, defGr * kerSpH * kerSpW * 2, offSpH, offSpW},
                                             {gr * outChPerGr, inChPerGr, kerSpH, kerSpW},
                                             {batch, defGr * kerSpH * kerSpW, offSpH, offSpW}};
        shapes.push_back(inputShape);
    }
    return shapes;
}

const auto addSpParams = ::testing::Combine(::testing::ValuesIn(padTypes),                      // pad. type
                                            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. begin
                                            ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. end
                                            ::testing::Values(std::vector<size_t>{1, 1}),       // strides
                                            ::testing::Values(std::vector<size_t>{1, 1})        // dilations
);

const auto addSpParamsDilationUneven =
    ::testing::Combine(::testing::ValuesIn(padTypes),                      // pad. type
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. end
                       ::testing::Values(std::vector<size_t>{1, 1}),       // strides
                       ::testing::Values(std::vector<size_t>{2, 1}));      // dilations

const std::vector<std::vector<size_t>> spatParams1 = {
    {1},       // batch
    {34, 34},  // in. spat. shape
    {32, 32},  // off. spat. shape
    {3, 3}     // ker. spat. shape
};
const std::vector<std::vector<size_t>> spatParams2 = {
    {1},     // batch
    {3, 3},  // in. spat. shape
    {2, 2},  // off. spat. shape
    {2, 2}   // ker. spat. shape
};
const std::vector<std::vector<size_t>> spatParams3 = {
    {1},     // batch
    {5, 5},  // in. spat. shape
    {4, 4},  // off. spat. shape
    {2, 2}   // ker. spat. shape
};
const std::vector<std::vector<size_t>> spatParams4 = {
    {1},     // batch
    {3, 2},  // in. spat. shape
    {2, 1},  // off. spat. shape
    {2, 2}   // ker. spat. shape
};
const std::vector<std::vector<size_t>> spatParamsDilationUneven = {
    {1},     // batch
    {3, 2},  // in. spat. shape
    {1, 1},  // off. spat. shape
    {2, 2}   // ker. spat. shape
};
const std::vector<std::vector<size_t>> spatParams5_onnx2d = {
    {1},     // batch
    {4, 4},  // in. spat. shape
    {3, 3},  // off. spat. shape
    {2, 2}   // ker. spat. shape
};
const std::vector<std::vector<size_t>> channelParamsSingleGr = {
    {1},       // gr. 2,4
    {1, 2},    // def. gr. 1,2
    {16, 32},  // in. ch. per gr.
    {16, 32}   // out. ch. per gr.
};
const std::vector<std::vector<size_t>> channelParamsSingleGr2 = {
    {1},  // gr. 2,4
    {1},  // def. gr. 1,2
    {3},  // in. ch. per gr.
    {3}   // out. ch. per gr.
};
const std::vector<std::vector<size_t>> channelParamsMulGr = {
    {2, 4},  // gr. 2,4
    {1, 2},  // def. gr. 1,2
    {3, 7},  // in. ch. per gr.
    {3, 7}   // out. ch. per gr.
};
const std::vector<std::vector<size_t>> channelParams_onnx2d = {
    {1},  // gr. 2,4
    {1},  // def. gr. 1,2
    {1},  // in. ch. per gr.
    {1}   // out. ch. per gr.
};
const std::vector<std::vector<InputShape>> dynShapeChainRef = {
    {
        // gr == 2, dg == 1, in_ch_per_gr == 3, out_ch_per_gr == 3
        // {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
        {{-1, -1, -1, -1}, {{1, 6, 3, 2}, {1, 6, 4, 3}, {1, 6, 5, 4}, {1, 6, 3, 2}}},  // input 0
        {{-1, -1, -1, -1}, {{1, 8, 2, 1}, {1, 8, 3, 2}, {1, 8, 4, 3}, {1, 8, 2, 1}}},  // input 1
        {{6, 3, 2, 2}, {{6, 3, 2, 2}, {6, 3, 2, 2}, {6, 3, 2, 2}, {6, 3, 2, 2}}},      // input 2
        {{-1, -1, -1, -1}, {{1, 4, 2, 1}, {1, 4, 3, 2}, {1, 4, 4, 3}, {1, 4, 2, 1}}}   // input 3
    },
    {{{{1, 5}, 6, {1, 10}, {1, 8}}, {{2, 6, 3, 2}, {1, 6, 4, 3}, {3, 6, 5, 4}, {2, 6, 3, 2}}},
     {{{1, 3}, 8, {1, 10}, {1, 8}}, {{2, 8, 2, 1}, {1, 8, 3, 2}, {3, 8, 4, 3}, {2, 8, 2, 1}}},
     {{6, 3, 2, 2}, {{6, 3, 2, 2}, {6, 3, 2, 2}, {6, 3, 2, 2}, {6, 3, 2, 2}}},
     {{{1, 3}, 4, {1, 10}, {1, 8}}, {{2, 4, 2, 1}, {1, 4, 3, 2}, {3, 4, 4, 3}, {2, 4, 2, 1}}}},
    {{{{1, 5}, {1, 6}, {1, 10}, {1, 8}}, {{2, 6, 3, 2}, {1, 6, 4, 3}, {3, 6, 5, 4}, {2, 6, 3, 2}}},
     {{{1, 3}, {1, 8}, {1, 10}, {1, 8}}, {{2, 8, 2, 1}, {1, 8, 3, 2}, {3, 8, 4, 3}, {2, 8, 2, 1}}},
     {{6, 3, 2, 2}, {{6, 3, 2, 2}, {6, 3, 2, 2}, {6, 3, 2, 2}, {6, 3, 2, 2}}},
     {{{1, 3}, {1, 5}, {1, 10}, {1, 8}}, {{2, 4, 2, 1}, {1, 4, 3, 2}, {3, 4, 4, 3}, {2, 4, 2, 1}}}},
};
const std::vector<std::vector<InputShape>> dynShapeChainJIT = {
    {
        // gr == 1, dg == 1, in_ch_per_gr == 16, out_ch_per_gr == 16
        // {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
        {{-1, -1, -1, -1}, {{1, 16, 3, 2}, {1, 16, 4, 3}, {1, 16, 5, 4}, {1, 16, 3, 2}}},    // input 0
        {{-1, 8, -1, -1}, {{1, 8, 2, 1}, {1, 8, 3, 2}, {1, 8, 4, 3}, {1, 8, 2, 1}}},         // input 1
        {{16, 16, 2, 2}, {{16, 16, 2, 2}, {16, 16, 2, 2}, {16, 16, 2, 2}, {16, 16, 2, 2}}},  // input 2
        {{-1, 4, -1, -1}, {{1, 4, 2, 1}, {1, 4, 3, 2}, {1, 4, 4, 3}, {1, 4, 2, 1}}}          // input 3
    },
    {
        {{{1, 5}, 16, {1, 10}, {1, 8}}, {{1, 16, 3, 2}, {1, 16, 4, 3}, {1, 16, 5, 4}, {1, 16, 3, 2}}},  // input 0
        {{{1, 5}, 8, {1, 10}, {1, 8}}, {{1, 8, 2, 1}, {1, 8, 3, 2}, {1, 8, 4, 3}, {1, 8, 2, 1}}},       // input 1
        {{16, 16, 2, 2}, {{16, 16, 2, 2}, {16, 16, 2, 2}, {16, 16, 2, 2}, {16, 16, 2, 2}}},             // input 2
        {{{1, 5}, 4, {1, 10}, {1, 8}}, {{1, 4, 2, 1}, {1, 4, 3, 2}, {1, 4, 4, 3}, {1, 4, 2, 1}}}        // input 3
    },
    {
        {{{1, 5}, {1, 16}, {1, 10}, {1, 8}}, {{1, 16, 3, 2}, {1, 16, 4, 3}, {1, 16, 5, 4}, {1, 16, 3, 2}}},  // input 0
        {{{1, 5}, {1, 8}, {1, 10}, {1, 8}}, {{1, 8, 2, 1}, {1, 8, 3, 2}, {1, 8, 4, 3}, {1, 8, 2, 1}}},       // input 1
        {{16, 16, 2, 2}, {{16, 16, 2, 2}, {16, 16, 2, 2}, {16, 16, 2, 2}, {16, 16, 2, 2}}},                  // input 2
        {{{1, 5}, {1, 5}, {1, 10}, {1, 8}}, {{1, 4, 2, 1}, {1, 4, 3, 2}, {1, 4, 4, 3}, {1, 4, 2, 1}}}        // input 3
    },
};

// autopad params
const std::vector<std::vector<InputShape>> dynShapeChainJITAutoPad = {{
    {{{1, 5}, {1, 16}, {1, 10}, {1, 10}}, {{1, 16, 3, 2}, {1, 16, 10, 10}, {1, 16, 3, 2}}},  // input 0
    {{{1, 5}, 8, {1, 10}, {1, 10}}, {{1, 8, 3, 2}, {1, 8, 10, 10}, {1, 8, 3, 2}}},           // input 1
    {{16, 16, 2, 2}, {{16, 16, 2, 2}, {16, 16, 2, 2}, {16, 16, 2, 2}}},                      // input 2
    {{{1, 5}, 4, {1, 10}, {1, 10}}, {{1, 4, 3, 2}, {1, 4, 10, 10}, {1, 4, 3, 2}}}            // input 3
}};
const std::vector<std::vector<size_t>> autoPadSpatParams = {
    {1},     // batch
    {3, 2},  // in. spat. shape
    {3, 2},  // off. spat. shape
    {2, 2}   // ker. spat. shape
};

std::vector<ov::op::PadType> padTypesAutoPad = {ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER};

const auto autoPadAddSpParams =
    ::testing::Combine(::testing::ValuesIn(padTypesAutoPad),               // pad. type
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. begin - ignored
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. end - ignored
                       ::testing::Values(std::vector<size_t>{1, 1}),       // strides
                       ::testing::Values(std::vector<size_t>{1, 1}));      // dilations

const auto params1_Smoke = ::testing::Combine(
    ::testing::Combine(addSpParams,
                       ::testing::ValuesIn(
                           static_shapes_to_test_representation(buildStaticParams(spatParams1, channelParamsSingleGr))),
                       defConvSpecificParams_Smoke,
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params2_Smoke = ::testing::Combine(
    ::testing::Combine(addSpParams,
                       ::testing::ValuesIn(
                           static_shapes_to_test_representation(buildStaticParams(spatParams2, channelParamsSingleGr))),
                       defConvSpecificParams_Smoke,
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params3_Smoke = ::testing::Combine(
    ::testing::Combine(addSpParams,
                       ::testing::ValuesIn(
                           static_shapes_to_test_representation(buildStaticParams(spatParams3, channelParamsSingleGr))),
                       defConvSpecificParams_Smoke,
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params4_Smoke = ::testing::Combine(
    ::testing::Combine(addSpParams,
                       ::testing::ValuesIn(
                           static_shapes_to_test_representation(buildStaticParams(spatParams4, channelParamsSingleGr))),
                       defConvSpecificParams_Smoke,
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params5_Smoke = ::testing::Combine(
    ::testing::Combine(
        addSpParams,
        ::testing::ValuesIn(static_shapes_to_test_representation(buildStaticParams(spatParams4, channelParamsMulGr))),
        defConvSpecificParams_Smoke,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    ::testing::ValuesIn(filterCPUInfoForDevice(true)));
const auto params6_Smoke = ::testing::Combine(::testing::Combine(addSpParams,
                                                                 ::testing::ValuesIn(dynShapeChainRef),
                                                                 defConvSpecificParams_Smoke,
                                                                 ::testing::ValuesIn(netPrecisions),
                                                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                              ::testing::ValuesIn(filterCPUInfoForDevice(true)));
const auto params7_Smoke = ::testing::Combine(::testing::Combine(addSpParams,
                                                                 ::testing::ValuesIn(dynShapeChainJIT),
                                                                 defConvSpecificParams_Smoke,
                                                                 ::testing::ValuesIn(netPrecisions),
                                                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                              ::testing::ValuesIn(filterCPUInfoForDevice(false)));
const auto params8_Smoke =
    ::testing::Combine(::testing::Combine(autoPadAddSpParams,
                                          ::testing::ValuesIn(static_shapes_to_test_representation(
                                              buildStaticParams(autoPadSpatParams, channelParamsSingleGr))),
                                          defConvSpecificParams_Smoke,
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params9_Smoke = ::testing::Combine(::testing::Combine(autoPadAddSpParams,
                                                                 ::testing::ValuesIn(dynShapeChainJITAutoPad),
                                                                 defConvSpecificParams_Smoke,
                                                                 ::testing::ValuesIn(netPrecisions),
                                                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                              ::testing::ValuesIn(filterCPUInfoForDevice(false)));

INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest1,
                         DefConvLayerCPUTest,
                         params1_Smoke,
                         DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest2,
                         DefConvLayerCPUTest,
                         params2_Smoke,
                         DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest3,
                         DefConvLayerCPUTest,
                         params3_Smoke,
                         DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest4,
                         DefConvLayerCPUTest,
                         params4_Smoke,
                         DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest5,
                         DefConvLayerCPUTest,
                         params5_Smoke,
                         DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest6,
                         DefConvLayerCPUTest,
                         params6_Smoke,
                         DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest7,
                         DefConvLayerCPUTest,
                         params7_Smoke,
                         DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest8,
                         DefConvLayerCPUTest,
                         params8_Smoke,
                         DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest9,
                         DefConvLayerCPUTest,
                         params9_Smoke,
                         DefConvLayerCPUTest::getTestCaseName);

const auto params1 = ::testing::Combine(::testing::Combine(addSpParams,
                                                           ::testing::ValuesIn(static_shapes_to_test_representation(
                                                               buildStaticParams(spatParams1, channelParamsSingleGr2))),
                                                           defConvSpecificParams,
                                                           ::testing::ValuesIn(netPrecisions),
                                                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                        ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params2 = ::testing::Combine(::testing::Combine(addSpParams,
                                                           ::testing::ValuesIn(static_shapes_to_test_representation(
                                                               buildStaticParams(spatParams2, channelParamsSingleGr))),
                                                           defConvSpecificParams,
                                                           ::testing::ValuesIn(netPrecisions),
                                                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                        ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params3 = ::testing::Combine(::testing::Combine(addSpParams,
                                                           ::testing::ValuesIn(static_shapes_to_test_representation(
                                                               buildStaticParams(spatParams3, channelParamsSingleGr))),
                                                           defConvSpecificParams,
                                                           ::testing::ValuesIn(netPrecisions),
                                                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                        ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params4 = ::testing::Combine(::testing::Combine(addSpParams,
                                                           ::testing::ValuesIn(static_shapes_to_test_representation(
                                                               buildStaticParams(spatParams4, channelParamsSingleGr))),
                                                           defConvSpecificParams,
                                                           ::testing::ValuesIn(netPrecisions),
                                                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                        ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params5 = ::testing::Combine(
    ::testing::Combine(
        addSpParams,
        ::testing::ValuesIn(static_shapes_to_test_representation(buildStaticParams(spatParams4, channelParamsMulGr))),
        defConvSpecificParams,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    ::testing::ValuesIn(filterCPUInfoForDevice(true)));
const auto params6 = ::testing::Combine(::testing::Combine(addSpParams,
                                                           ::testing::ValuesIn(dynShapeChainRef),
                                                           defConvSpecificParams,
                                                           ::testing::ValuesIn(netPrecisions),
                                                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                        ::testing::ValuesIn(filterCPUInfoForDevice(true)));
const auto params7 = ::testing::Combine(::testing::Combine(addSpParams,
                                                           ::testing::ValuesIn(dynShapeChainJIT),
                                                           defConvSpecificParams,
                                                           ::testing::ValuesIn(netPrecisions),
                                                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                        ::testing::ValuesIn(filterCPUInfoForDevice(false)));
// autopad cases
const auto params8 =
    ::testing::Combine(::testing::Combine(autoPadAddSpParams,
                                          ::testing::ValuesIn(static_shapes_to_test_representation(
                                              buildStaticParams(autoPadSpatParams, channelParamsSingleGr))),
                                          defConvSpecificParams,
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params9 = ::testing::Combine(::testing::Combine(autoPadAddSpParams,
                                                           ::testing::ValuesIn(dynShapeChainJITAutoPad),
                                                           defConvSpecificParams,
                                                           ::testing::ValuesIn(netPrecisions),
                                                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                        ::testing::ValuesIn(filterCPUInfoForDevice(false)));
const auto params10 =
    ::testing::Combine(::testing::Combine(addSpParamsDilationUneven,
                                          ::testing::ValuesIn(static_shapes_to_test_representation(
                                              buildStaticParams(spatParamsDilationUneven, channelParamsSingleGr))),
                                          defConvSpecificParams,
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::ValuesIn(filterCPUInfoForDevice(false)));
const auto params11 =
    ::testing::Combine(::testing::Combine(addSpParams,
                                          ::testing::ValuesIn(static_shapes_to_test_representation(
                                              buildStaticParams(spatParams5_onnx2d, channelParams_onnx2d))),
                                          defConvSpecificParams,
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::ValuesIn(filterCPUInfoForDevice()));

INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest1, DefConvLayerCPUTest, params1, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest2, DefConvLayerCPUTest, params2, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest3, DefConvLayerCPUTest, params3, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest4, DefConvLayerCPUTest, params4, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest5, DefConvLayerCPUTest, params5, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest6, DefConvLayerCPUTest, params6, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest7, DefConvLayerCPUTest, params7, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest8, DefConvLayerCPUTest, params8, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest9, DefConvLayerCPUTest, params9, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest10, DefConvLayerCPUTest, params10, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest11, DefConvLayerCPUTest, params11, DefConvLayerCPUTest::getTestCaseName);

const std::vector<std::vector<size_t>> blockMultigroupChParam = {
    {2},   // gr.
    {1},   // def. gr.
    {16},  // in. ch. per gr.
    {16}   // out. ch. per gr.
};
const std::vector<std::vector<size_t>> blockMultigroupSpatParam = {
    {1},     // batch
    {2, 2},  // in. spat. shape
    {2, 2},  // off. spat. shape
    {1, 1}   // ker. spat. shape
};
const auto blockMultigroupAddParam = ::testing::Combine(::testing::Values(true),   // with_bilinear_interpolation_pad
                                                        ::testing::Values(false),  // with_modulation
                                                        ::testing::Values(OffsetType::ZERO)  // offset type
);
const auto blockMultigroupKernelParam =
    ::testing::Combine(::testing::Values(ov::op::PadType::EXPLICIT),       // pad. type
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. begin
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. end
                       ::testing::Values(std::vector<size_t>{1, 1}),       // strides
                       ::testing::Values(std::vector<size_t>{1, 1}));      // dilations
const auto blockMultigroupParam =
    ::testing::Combine(::testing::Combine(blockMultigroupKernelParam,
                                          ::testing::ValuesIn(static_shapes_to_test_representation(
                                              buildStaticParams(blockMultigroupSpatParam, blockMultigroupChParam))),
                                          blockMultigroupAddParam,
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::ValuesIn(filterCPUInfoForDevice(true)));
INSTANTIATE_TEST_SUITE_P(blockMultigroupDefConvTest,
                         DefConvLayerCPUTest,
                         blockMultigroupParam,
                         DefConvLayerCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
