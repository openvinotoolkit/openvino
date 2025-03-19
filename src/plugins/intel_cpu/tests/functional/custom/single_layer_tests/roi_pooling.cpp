// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/test_enums.hpp"

#include "utils/cpu_test_utils.hpp"
#include "utils/bfloat16.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
enum ProposalGenerationMode { RANDOM, ULTIMATE_RIGHT_BORDER };

using roiPoolingShapes = std::vector<InputShape>;

using roiPoolingParams = std::tuple<roiPoolingShapes,                // Input shapes
                                    std::vector<size_t>,             // Pooled shape {pooled_h, pooled_w}
                                    float,                           // Spatial scale
                                    utils::ROIPoolingTypes,          // ROIPooling method
                                    ov::element::Type,               // Net precision
                                    ov::test::TargetDevice>;  // Device name

using ROIPoolingCPUTestParamsSet = std::tuple<roiPoolingParams,
                                              CPUSpecificParams,
                                              ProposalGenerationMode,
                                              ov::AnyMap>;

class ROIPoolingCPULayerTest : public testing::WithParamInterface<ROIPoolingCPUTestParamsSet>,
                               public ov::test::SubgraphBaseTest,
                               public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ROIPoolingCPUTestParamsSet> obj) {
        roiPoolingParams basicParamsSet;
        CPUSpecificParams cpuParams;
        ProposalGenerationMode propMode;
        ov::AnyMap additionalConfig;

        std::tie(basicParamsSet, cpuParams, propMode, additionalConfig) = obj.param;

        roiPoolingShapes inputShapes;
        std::vector<size_t> poolShape;
        float spatial_scale;
       utils::ROIPoolingTypes pool_method;
        ov::element::Type netPrecision;
        std::string targetDevice;
        std::tie(inputShapes, poolShape, spatial_scale, pool_method, netPrecision, targetDevice) = basicParamsSet;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.to_string() << "_";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({ shape.first }) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }

        result << "PS=" << ov::test::utils::vec2str(poolShape) << "_";
        result << "Scale=" << spatial_scale << "_";
        switch (pool_method) {
        case utils::ROIPoolingTypes::ROI_MAX:
            result << "Max_";
            break;
        case utils::ROIPoolingTypes::ROI_BILINEAR:
            result << "Bilinear_";
            break;
        }
        result << "trgDev=" << targetDevice;
        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second.as<std::string>();
            }
        }
        switch (propMode) {
            case ProposalGenerationMode::ULTIMATE_RIGHT_BORDER:
                result << "_UltimateRightBorderProposal";
                break;
            case ProposalGenerationMode::RANDOM:
            default:
                result << "_RandomProposal";
                break;
        }

        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        const ProposalGenerationMode propMode = std::get<2>(this->GetParam());
        const float spatial_scale = std::get<2>(std::get<0>(this->GetParam()));
        const utils::ROIPoolingTypes pool_method = std::get<3>(std::get<0>(this->GetParam()));

        inputs.clear();
        const auto& funcInputs = function->inputs();

        auto feat_map_shape = targetInputStaticShapes[0];
        const auto is_roi_max_mode = (pool_method ==utils::ROIPoolingTypes::ROI_MAX);
        const int height = is_roi_max_mode ? feat_map_shape[2] / spatial_scale : 1;
        const int width = is_roi_max_mode ? feat_map_shape[3] / spatial_scale : 1;

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 1) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                if (propMode == ULTIMATE_RIGHT_BORDER) {
                    // because of nonalgebraic character of floating point operation, the following values causes inequity:
                    // ((end_h - start_h) * (input_h - 1) / (pooled_h - 1)) * (pooled_h - 1) > (end_h - start_h) * (input_h - 1)
                    // and as result excess of right limit for proposal value if the border case (current_h == pooled_h - 1)
                    // will not be handled explicitly
                    switch (funcInput.get_element_type()) {
                    case ov::element::f32: {
                        auto* dataPtr = tensor.data<float>();
                        for (size_t i = 0; i < tensor.get_size(); i += 5) {
                            dataPtr[i] = 0;
                            dataPtr[i + 1] = 0.f;
                            dataPtr[i + 2] = 0.248046786f;
                            dataPtr[i + 3] = 0.471333951f;
                            dataPtr[i + 4] = 1.f;
                        }
                        break;
                    }
                    case ov::element::bf16: {
                        auto* dataPtr = tensor.data<std::int16_t>();
                        for (size_t i = 0; i < tensor.get_size(); i += 5) {
                            dataPtr[i] = static_cast<std::int16_t>(ov::float16(0.f).to_bits());
                            dataPtr[i + 1] = static_cast<std::int16_t>(ov::float16(0.f).to_bits());
                            dataPtr[i + 2] = static_cast<std::int16_t>(ov::float16(0.248046786f).to_bits());
                            dataPtr[i + 3] = static_cast<std::int16_t>(ov::float16(0.471333951f).to_bits());
                            dataPtr[i + 4] = static_cast<std::int16_t>(ov::float16(1.f).to_bits());
                        }
                        break;
                    }
                    default:
                        OPENVINO_THROW("roi_pooling. Unsupported precision");
                    }
                } else {
                    switch (funcInput.get_element_type()) {
                    case ov::element::f32:
                    case ov::element::bf16: {
                        ov::test::utils::fill_data_roi(tensor, feat_map_shape[0] - 1, height, width, 1.f, is_roi_max_mode);
                        break;
                    }
                    default:
                        OPENVINO_THROW("roi_pooling. Unsupported precision");
                    }
                }
            } else {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = 10;
                in_data.resolution = 1000;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            }

            inputs.insert({ funcInput.get_node_shared_ptr(), tensor });
        }
    }

    void SetUp() override {
        roiPoolingParams basicParamsSet;
        CPUSpecificParams cpuParams;
        ProposalGenerationMode propMode;
        ov::AnyMap additionalConfig;

        std::tie(basicParamsSet, cpuParams, propMode, additionalConfig) = this->GetParam();
        roiPoolingShapes inputShapes;
        std::vector<size_t> poolShape;
        float spatial_scale;
       utils::ROIPoolingTypes pool_method;
        ov::element::Type netPrecision;
        std::tie(inputShapes, poolShape, spatial_scale, pool_method, netPrecision, targetDevice) = basicParamsSet;

        auto it = additionalConfig.find(ov::hint::inference_precision.name());
        if (it != additionalConfig.end() && it->second.as<ov::element::Type>() == ov::element::bf16)
            netPrecision = ov::element::bf16;
        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }
        selectedType.push_back('_');

        if (!with_cpu_x86_avx512_core() && netPrecision == ElementType::bf16) {
            selectedType += ov::element::f32.to_string();
        } else {
            selectedType += netPrecision.to_string();
        }

        if (netPrecision == ov::element::bf16) {
            rel_threshold = 1e-2;
        }

        init_input_shapes(inputShapes);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));

        std::shared_ptr<ov::Node> roi_pooling;
        if (ov::test::utils::ROIPoolingTypes::ROI_MAX == pool_method) {
            roi_pooling = std::make_shared<ov::op::v0::ROIPooling>(params[0], params[1], poolShape, spatial_scale, "max");
        } else {
            roi_pooling = std::make_shared<ov::op::v0::ROIPooling>(params[0], params[1], poolShape, spatial_scale, "bilinear");
        }
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(roi_pooling)};

        function = makeNgraphFunction(netPrecision, params, roi_pooling, "ROIPooling");
        functionRefs = function->clone();
    }
};

TEST_P(ROIPoolingCPULayerTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "ROIPooling");
}

namespace {

std::vector<ov::AnyMap> additionalConfig = {{ov::hint::inference_precision(ov::element::f32)},
                                            {ov::hint::inference_precision(ov::element::bf16)}};
/* have to select particular implementation type, since currently
 * nodes always choose the best one */
std::vector<CPUSpecificParams> selectCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, nc}, {nChw16c}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, nc}, {nChw8c}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, nc}, {nChw8c}, {"jit_sse42"}, "jit_sse42"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, nc}, {nChw8c}, {"ref"}, "ref"});
    }

    return resCPUParams;
}

const std::vector<roiPoolingShapes> inShapes = {
    roiPoolingShapes{{{}, {{1, 3, 8, 8}}}, {{}, {{1, 5}}}},
    roiPoolingShapes{{{}, {{1, 3, 8, 8}}}, {{}, {{3, 5}}}},
    roiPoolingShapes{{{}, {{3, 4, 50, 50}}}, {{}, {{3, 5}}}},
    roiPoolingShapes{{{}, {{3, 4, 50, 50}}}, {{}, {{5, 5}}}},
    roiPoolingShapes{
        // input 0
        {
            // dynamic
            {-1, -1, -1, -1},
            // static
            {
                {3, 4, 50, 50}, {3, 4, 50, 50}, {3, 4, 50, 50}, {1, 3, 8, 8}, {1, 3, 8, 8}, {3, 4, 50, 50}
            }
        },
        // input 1
        {
            // dynamic
            {-1, 5},
            // static
            {
                {1, 5}, {3, 5}, {5, 5}, {1, 5}, {3, 5}, {5, 5}
            }
        },
    },
    roiPoolingShapes{
        // input 0
        {
            // dynamic
            {-1, {3, 5}, {7, 60}, -1},
            // static
            {
                {3, 4, 50, 50}, {1, 3, 7, 8}, {3, 4, 50, 50}, {1, 3, 7, 8},
            }
        },
        // input 1
        {
            // dynamic
            {{1, 5}, 5},
            // static
            {
                {1, 5}, {2, 5}, {1, 5}, {2, 5}
            }
        },
    },
    roiPoolingShapes{
        // input 0
        {
            // dynamic
            {{1, 8}, {3, 5}, {7, 60}, {5, 50}},
            // static
            {
                {3, 4, 50, 50}, {1, 3, 7, 8}, {8, 5, 59, 5}, {1, 3, 7, 8},
            }
        },
        // input 1
        {
            // dynamic
            {{1, 5}, 5},
            // static
            {
                {1, 5}, {2, 5}, {1, 5}, {2, 5}
            }
        },
    },
};

const std::vector<std::vector<size_t>> pooledShapes_max = {
    {1, 1},
    {2, 2},
    {3, 3},
    {6, 6}
};

const std::vector<std::vector<size_t>> pooledShapes_bilinear = {
    {1, 1},
    {2, 2},
    {3, 3},
    {6, 6}
};

const std::vector<ov::element::Type> netPRCs = {ov::element::f32, ov::element::bf16};

const std::vector<float> spatial_scales = {0.625f, 1.f};

const auto test_ROIPooling_max = ::testing::Combine(::testing::ValuesIn(inShapes),
                                                    ::testing::ValuesIn(pooledShapes_max),
                                                    ::testing::ValuesIn(spatial_scales),
                                                    ::testing::Values(utils::ROIPoolingTypes::ROI_MAX),
                                                    ::testing::ValuesIn(netPRCs),
                                                    ::testing::Values(ov::test::utils::DEVICE_CPU));

const auto test_ROIPooling_bilinear = ::testing::Combine(::testing::ValuesIn(inShapes),
                                                         ::testing::ValuesIn(pooledShapes_bilinear),
                                                         ::testing::Values(spatial_scales[1]),
                                                         ::testing::Values(utils::ROIPoolingTypes::ROI_BILINEAR),
                                                         ::testing::ValuesIn(netPRCs),
                                                         ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_ROIPoolingCPU_max,
                        ROIPoolingCPULayerTest,
                        ::testing::Combine(test_ROIPooling_max,
                                           ::testing::ValuesIn(selectCPUInfoForDevice()),
                                           ::testing::Values(ProposalGenerationMode::RANDOM),
                                           ::testing::ValuesIn(additionalConfig)),
                        ROIPoolingCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ROIPoolingCPU_bilinear,
                        ROIPoolingCPULayerTest,
                        ::testing::Combine(test_ROIPooling_bilinear,
                                           ::testing::ValuesIn(selectCPUInfoForDevice()),
                                           ::testing::Values(ProposalGenerationMode::RANDOM),
                                           ::testing::ValuesIn(additionalConfig)),
                        ROIPoolingCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_ROIPoolingCPU_bilinear_ultimateRightBorderProposal,
    ROIPoolingCPULayerTest,
    ::testing::Combine(::testing::Combine(::testing::Values(roiPoolingShapes{{{}, {{1, 1, 50, 50}}}, {{}, {{1, 5}}}}),
                                          ::testing::Values(std::vector<size_t>{4, 4}),
                                          ::testing::Values(spatial_scales[1]),
                                          ::testing::Values(utils::ROIPoolingTypes::ROI_BILINEAR),
                                          ::testing::Values(ov::element::f32),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::ValuesIn(selectCPUInfoForDevice()),
                       ::testing::Values(ProposalGenerationMode::ULTIMATE_RIGHT_BORDER),
                       ::testing::Values(ov::AnyMap{{ov::hint::inference_precision(ov::element::f32)}})),
    ROIPoolingCPULayerTest::getTestCaseName);
} // namespace
}  // namespace test
}  // namespace ov
