// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {
enum OffsetType {ZERO, NATURAL, REAL_POSITIVE, REAL_NEGATIVE, REAL_MISC};

typedef std::tuple<
    bool,       // with_bilinear_interpolation_pad
    bool,       // with_modulation
    OffsetType  // type of def. offsets
    > DefConvSpecificParams;

typedef std::tuple<
    size_t,                  // batches
    std::vector<size_t>,     // input spatial shape
    std::vector<size_t>,     // offsets spatial shape
    std::vector<size_t>,     // kernel spatial shape
    ngraph::op::PadType,     // pad. type
    std::vector<ptrdiff_t>,  // pad. begin
    std::vector<ptrdiff_t>,  // pad. end
    std::vector<size_t>,     // strides
    std::vector<size_t>      // dilations
    > SpatialParams;

typedef std::tuple<
    size_t,     // groups
    size_t,     // deformable groups
    size_t,     // input channels per group
    size_t      // output channels per group
    > ChannelParams;

typedef std::tuple<
    SpatialParams,
    ChannelParams,
    DefConvSpecificParams,
    InferenceEngine::Precision,     // Net precision
    LayerTestsUtils::TargetDevice   // Device name
    > DefConvLayerTestParams;

typedef std::tuple<
    CPULayerTestsDefinitions::DefConvLayerTestParams,
    CPUSpecificParams> DefConvLayerCPUTestParamsSet;

class DefConvLayerCPUTest : public testing::WithParamInterface<DefConvLayerCPUTestParamsSet>,
        virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    OffsetType offsetType;
    static std::string getTestCaseName(testing::TestParamInfo<DefConvLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::DefConvLayerTestParams basicParamsSet;
        std::string td;
        Precision netPr;
        InferenceEngine::Precision inPrc, outPrc;
        ChannelParams chParams;
        SpatialParams spParams;
        CPULayerTestsDefinitions::DefConvSpecificParams dcSpecificParams;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::tie(spParams, chParams, dcSpecificParams, netPr, td) = basicParamsSet;
        inPrc = outPrc = netPr;
        ngraph::op::PadType padType;
        size_t batch;
        InferenceEngine::SizeVector offsets, filter, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        InferenceEngine::SizeVector inSpatShape, offSpatShape, kerSpatShape;
        std::tie(batch, inSpatShape, offSpatShape, kerSpatShape,
                 padType, padBegin, padEnd, stride, dilation) = spParams;
        size_t groups, deformableGroups, inGrCh, outGrCh;
        std::tie(groups, deformableGroups, inGrCh, outGrCh) = chParams;
        bool withBilinearInterpolationPad, withModulation;
        OffsetType offType;
        std::tie(withBilinearInterpolationPad, withModulation, offType) = dcSpecificParams;
        std::ostringstream result;
        result << "DefConvTest(";
        result << std::to_string(obj.index) << ")_";
        result << "IS=(" << batch << "_" << groups * inGrCh << "_" << inSpatShape[0] << "_" << inSpatShape[1] << ")_";
        result << "OS=(" << batch << "_" << groups * outGrCh << "_" << offSpatShape[0] << "_" << offSpatShape[1] << ")_";
        result << "K" << CommonTestUtils::vec2str(kerSpatShape) << "_";
        result << "S" << CommonTestUtils::vec2str(stride) << "_";
        result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
        result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
        result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
        result << "O=" << groups * outGrCh << "_";
        result << "AP=" << padType << "_";
        result << "netPRC=" << netPr.name() << "_";
        result << "inPRC=" << inPrc.name() << "_";
        result << "outPRC=" << outPrc.name() << "_";
        result << "withBilPad=" << withBilinearInterpolationPad << "_";
        result << "withMod=" << withModulation << "_";
        result << "offsetType=" << offType << "_";
        result << "trgDev=" << td;
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }
protected:
    void GenerateInputs() override {
        for (const auto &input : cnnNetwork.getInputsInfo()) {
            const auto info = input.second.get();
            const auto &name = info->name();
            InferenceEngine::Blob::Ptr blob;
            if (name == "a_data") {
                blob = GenerateInput(*info);
            } else if (name == "b_offset_vals") {
                if (offsetType == OffsetType::NATURAL) {
                    blob = FuncTestUtils::createAndFillBlobFloat(info->getTensorDesc(), 10, 0, 1);
                } else if (offsetType == OffsetType::ZERO) {
                    blob = FuncTestUtils::createAndFillBlobFloat(info->getTensorDesc(), 0, 1, 1);
                } else if (offsetType == OffsetType::REAL_POSITIVE) {
                    blob = FuncTestUtils::createAndFillBlobFloat(info->getTensorDesc(), 2, 0, 100);
                } else if (offsetType == OffsetType::REAL_NEGATIVE) {
                    blob = FuncTestUtils::createAndFillBlobFloat(info->getTensorDesc(), 2, -2, 100);
                } else if (offsetType == OffsetType::REAL_MISC) {
                    blob = FuncTestUtils::createAndFillBlobFloat(info->getTensorDesc(), 4, -2, 100);
                } else {
                    IE_THROW() << "Unexpected offset type";
                }
            } else if (name == "c_filter_vals") {
                blob = GenerateInput(*info);
            } else if (name == "c_modulation_scalars") {
                blob = FuncTestUtils::createAndFillBlobFloat(info->getTensorDesc(), 1, 0, 100);
            } else {
                IE_THROW() << "Unknown input of DeformableConvolution";
            }
            inputs.push_back(blob);
        }
    }
    void SetUp() override {
        ChannelParams chParams;
        SpatialParams spParams;
        CPULayerTestsDefinitions::DefConvSpecificParams dcSpecificParams;

        std::vector<size_t> inShape;
        InferenceEngine::Precision netPrecision;
        CPULayerTestsDefinitions::DefConvLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::tie(spParams, chParams, dcSpecificParams, netPrecision, targetDevice) = basicParamsSet;

        inPrc = outPrc = netPrecision;
        inLayout = outLayout = InferenceEngine::Layout::ANY;
        ngraph::op::PadType padType;
        size_t batch;
        InferenceEngine::SizeVector offsets, filter, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        InferenceEngine::SizeVector inSpatShape, offSpatShape, kerSpatShape;
        std::tie(batch, inSpatShape, offSpatShape, kerSpatShape,
                 padType, padBegin, padEnd, stride, dilation) = spParams;

        size_t groups, deformableGroups, inGrCh, outGrCh;
        std::tie(groups, deformableGroups, inGrCh, outGrCh) = chParams;
        bool withBilinearInterpolationPad, withModulation;
        std::tie(withBilinearInterpolationPad, withModulation, offsetType) = dcSpecificParams;

        inShape = std::vector<size_t>({batch, groups * inGrCh, inSpatShape[0], inSpatShape[1]});
        offsets = std::vector<size_t> {batch, deformableGroups * kerSpatShape[0] * kerSpatShape[1] * 2,
                                       offSpatShape[0], offSpatShape[1]};
        filter = std::vector<size_t> {groups * outGrCh, inGrCh, kerSpatShape[0], kerSpatShape[1]};

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inShape, offsets, filter});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto data = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape(inShape));
        data->set_friendly_name("a_data");
        auto offset_vals = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape(offsets));
        offset_vals->set_friendly_name("b_offset_vals");
        auto filter_vals = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape(filter));
        filter_vals->set_friendly_name("c_filter_vals");
        ngraph::ParameterVector parameters{data, offset_vals, filter_vals};
        std::shared_ptr<ngraph::Node> deformable_conv;
        if (withModulation) {
            auto modulation_shape = ngraph::Shape(offsets);
            modulation_shape[1] = offsets[1] / 2;
            auto modulation_scalars = std::make_shared<ngraph::op::Parameter>(ngPrc, modulation_shape);
            modulation_scalars->set_friendly_name("c_modulation_scalars");

            deformable_conv = std::make_shared<ngraph::op::v8::DeformableConvolution>(data, offset_vals, filter_vals, modulation_scalars, stride, padBegin,
                                                                                      padEnd, dilation, padType, groups, deformableGroups,
                                                                                      withBilinearInterpolationPad);
            parameters.push_back(modulation_scalars);
        } else {
            deformable_conv = std::make_shared<ngraph::op::v8::DeformableConvolution>(data, offset_vals, filter_vals, stride, padBegin, padEnd, dilation,
                                                                                      padType, groups, deformableGroups, withBilinearInterpolationPad);
        }

        function = makeNgraphFunction(ngPrc, parameters, deformable_conv, "deformable_convolution");
    }
};

TEST_P(DefConvLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
    CheckPluginRelatedResults(executableNetwork, "DeformableConvolution");
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

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

const auto defConvSpecificParams_Smoke = ::testing::Combine(
    ::testing::ValuesIn(std::vector<bool> {
        true,
        false
    }),  // with_bilinear_interpolation_pad
    ::testing::ValuesIn(std::vector<bool> {
        true,
        false
    }),  // with_modulation
    ::testing::ValuesIn(std::vector<OffsetType> {
        OffsetType::REAL_MISC,
    })  // offset type
);

const auto defConvSpecificParams = ::testing::Combine(
    ::testing::ValuesIn(std::vector<bool> {
        true,
        false
    }),  // with_bilinear_interpolation_pad
    ::testing::ValuesIn(std::vector<bool> {
        true,
        false
    }),  // with_modulation
    ::testing::ValuesIn(std::vector<OffsetType> {
        OffsetType::NATURAL,
        OffsetType::ZERO,
        OffsetType::REAL_MISC,
        OffsetType::REAL_POSITIVE,
        OffsetType::REAL_NEGATIVE
    })  // offset type
);

std::vector<ngraph::op::PadType> padTypes = {
    ngraph::op::PadType::EXPLICIT,
    ngraph::op::PadType::VALID
};

const auto spParams1 = ::testing::Combine(
    ::testing::Values(1),  // batch
    ::testing::Values(std::vector<size_t>({34, 34})),  // in. spat. shape
    ::testing::Values(std::vector<size_t>({32, 32})),  // off. spat. shape
    ::testing::Values(std::vector<size_t>({3, 3})),  // ker. spat. shape
    ::testing::ValuesIn(padTypes),  // pad. type
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. begin
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. end
    ::testing::Values(std::vector<size_t> {1, 1}),  // strides
    ::testing::Values(std::vector<size_t> {1, 1})  // dilations
);

const auto spParams2 = ::testing::Combine(
        ::testing::Values(1),  // batch
        ::testing::Values(std::vector<size_t>({3, 3})),  // in. spat. shape
        ::testing::Values(std::vector<size_t>({2, 2})),  // off. spat. shape
        ::testing::Values(std::vector<size_t>({2, 2})),  // ker. spat. shape
        ::testing::ValuesIn(padTypes),  // pad. type
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. begin
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. end
        ::testing::Values(std::vector<size_t> {1, 1}),  // strides
        ::testing::Values(std::vector<size_t> {1, 1})  // dilations
);

const auto spParams3 = ::testing::Combine(
        ::testing::Values(1),  // batch
        ::testing::Values(std::vector<size_t>({5, 5})),  // in. spat. shape
        ::testing::Values(std::vector<size_t>({4, 4})),  // off. spat. shape
        ::testing::Values(std::vector<size_t>({2, 2})),  // ker. spat. shape
        ::testing::ValuesIn(padTypes),  // pad. type
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. begin
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. end
        ::testing::Values(std::vector<size_t> {1, 1}),  // strides
        ::testing::Values(std::vector<size_t> {1, 1})  // dilations
);
const auto spParams4 = ::testing::Combine(
        ::testing::Values(1),  // batch
        ::testing::Values(std::vector<size_t>({3, 2})),  // in. spat. shape
        ::testing::Values(std::vector<size_t>({2, 1})),  // off. spat. shape
        ::testing::Values(std::vector<size_t>({2, 2})),  // ker. spat. shape
        ::testing::ValuesIn(padTypes),  // pad. type
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. begin
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  // pad. end
        ::testing::Values(std::vector<size_t> {1, 1}),  // strides
        ::testing::Values(std::vector<size_t> {1, 1})  // dilations
);

const auto chParamsSingleGr = ::testing::Combine(
        ::testing::ValuesIn(std::vector<size_t> {1}),  // gr. 1
        ::testing::ValuesIn(std::vector<size_t> {1, 2}),  // def. gr. 1,2
        ::testing::ValuesIn(std::vector<size_t> {16, 32}),  // in. ch. per gr.
        ::testing::ValuesIn(std::vector<size_t> {16, 32}));  // out. ch. per gr.

const auto chParamsMulGr = ::testing::Combine(
        ::testing::ValuesIn(std::vector<size_t> {2, 4}),  // gr. 2,4
        ::testing::ValuesIn(std::vector<size_t> {1, 2}),  // def. gr. 1,2
        ::testing::ValuesIn(std::vector<size_t> {3, 7}),  // in. ch. per gr.
        ::testing::ValuesIn(std::vector<size_t> {3, 7}));  // out. ch. per gr.

const auto params1_Smoke = ::testing::Combine(
                             ::testing::Combine(
                                 spParams1,
                                 chParamsSingleGr,
                                 defConvSpecificParams_Smoke,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                             ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params2_Smoke = ::testing::Combine(
                             ::testing::Combine(
                                 spParams2,
                                 chParamsSingleGr,
                                 defConvSpecificParams_Smoke,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                             ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params3_Smoke = ::testing::Combine(
                             ::testing::Combine(
                                 spParams3,
                                 chParamsSingleGr,
                                 defConvSpecificParams_Smoke,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                             ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params4_Smoke = ::testing::Combine(
                             ::testing::Combine(
                                 spParams4,
                                 chParamsSingleGr,
                                 defConvSpecificParams_Smoke,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                             ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params5_Smoke = ::testing::Combine(
                             ::testing::Combine(
                                 spParams4,
                                 chParamsMulGr,
                                 defConvSpecificParams_Smoke,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                             ::testing::ValuesIn(filterCPUInfoForDevice(true)));
INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest1, DefConvLayerCPUTest, params1_Smoke, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest2, DefConvLayerCPUTest, params2_Smoke, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest3, DefConvLayerCPUTest, params3_Smoke, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest4, DefConvLayerCPUTest, params4_Smoke, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DefConvLayoutTest5, DefConvLayerCPUTest, params5_Smoke, DefConvLayerCPUTest::getTestCaseName);

const auto params1 = ::testing::Combine(
                         ::testing::Combine(
                             spParams1,
                             chParamsSingleGr,
                             defConvSpecificParams,
                             ::testing::ValuesIn(netPrecisions),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params2 = ::testing::Combine(
                         ::testing::Combine(
                             spParams2,
                             chParamsSingleGr,
                             defConvSpecificParams,
                             ::testing::ValuesIn(netPrecisions),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params3 = ::testing::Combine(
                         ::testing::Combine(
                             spParams3,
                             chParamsSingleGr,
                             defConvSpecificParams,
                             ::testing::ValuesIn(netPrecisions),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params4 = ::testing::Combine(
                         ::testing::Combine(
                             spParams4,
                             chParamsSingleGr,
                             defConvSpecificParams,
                             ::testing::ValuesIn(netPrecisions),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         ::testing::ValuesIn(filterCPUInfoForDevice()));
const auto params5 = ::testing::Combine(
                         ::testing::Combine(
                             spParams4,
                             chParamsMulGr,
                             defConvSpecificParams,
                             ::testing::ValuesIn(netPrecisions),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         ::testing::ValuesIn(filterCPUInfoForDevice(true)));
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest1, DefConvLayerCPUTest, params1, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest2, DefConvLayerCPUTest, params2, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest3, DefConvLayerCPUTest, params3, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest4, DefConvLayerCPUTest, params4, DefConvLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DefConvLayoutTest5, DefConvLayerCPUTest, params5, DefConvLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
