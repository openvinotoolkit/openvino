// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/roi_pooling.hpp"

namespace {
enum ProposalGenerationMode { RANDOM, ULTIMATE_RIGHT_BORDER };

using ROIPoolingShapes = std::vector<ov::test::InputShape>;

typedef std::tuple<
    ROIPoolingShapes,                   // Input shapes
    std::vector<size_t>,                // Pooled shape {pooled_h, pooled_w}
    float,                              // Spatial scale
    ov::test::utils::ROIPoolingTypes,   // ROIPooling method
    ov::element::Type                   // Model type
> ROIPoolingParams;

typedef std::tuple<
    ROIPoolingParams,
    ProposalGenerationMode
> ROIPoolingGPUTestParams;

class ROIPoolingLayerGPUTest : public testing::WithParamInterface<ROIPoolingGPUTestParams>,
                               virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ROIPoolingGPUTestParams> obj) {
        const auto& [basic_params_set, prop_mode] = obj.param;

        const auto& [shapes, pool_shape, spatial_scale, pool_method, model_type] = basic_params_set;

        std::ostringstream result;
        result << "netPRC=" << model_type << "_";
        for (const auto& shape : shapes) {
            result << ov::test::utils::partialShape2str({ shape.first }) << "_";
        }
        result << "TS=";
        for (const auto& shape : shapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }

        result << "PS=" << ov::test::utils::vec2str(pool_shape) << "_";
        result << "Scale=" << spatial_scale << "_";
        switch (pool_method) {
        case ov::test::utils::ROIPoolingTypes::ROI_MAX:
            result << "Max_";
            break;
        case ov::test::utils::ROIPoolingTypes::ROI_BILINEAR:
            result << "Bilinear_";
            break;
        }
        switch (prop_mode) {
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
        const ProposalGenerationMode prop_mode = std::get<1>(this->GetParam());
        const float spatial_scale = std::get<2>(std::get<0>(this->GetParam()));
        const ov::test::utils::ROIPoolingTypes pool_method = std::get<3>(std::get<0>(this->GetParam()));

        inputs.clear();
        const auto& funcInputs = function->inputs();

        auto feat_map_shape = targetInputStaticShapes[0];
        const auto is_roi_max_mode = (pool_method == ov::test::utils::ROIPoolingTypes::ROI_MAX);
        const int height = is_roi_max_mode ? feat_map_shape[2] / spatial_scale : 1;
        const int width = is_roi_max_mode ? feat_map_shape[3] / spatial_scale : 1;

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 1) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                if (prop_mode == ULTIMATE_RIGHT_BORDER) {
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
        const auto& [basic_params_set, prop_mode] = this->GetParam();

        const auto& [shapes, pool_shape, spatial_scale, pool_method, model_type] = basic_params_set;

        targetDevice = ov::test::utils::DEVICE_GPU;
        init_input_shapes(shapes);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));

        std::shared_ptr<ov::Node> roi_pooling;
        if (ov::test::utils::ROIPoolingTypes::ROI_MAX == pool_method) {
            roi_pooling = std::make_shared<ov::op::v0::ROIPooling>(params[0], params[1], pool_shape, spatial_scale, "max");
        } else {
            roi_pooling = std::make_shared<ov::op::v0::ROIPooling>(params[0], params[1], pool_shape, spatial_scale, "bilinear");
        }
        ov::ResultVector results;
        for (size_t i = 0; i < roi_pooling->get_output_size(); i++)
            results.push_back(std::make_shared<ov::op::v0::Result>(roi_pooling->output(i)));
        function = std::make_shared<ov::Model>(results, params, "ROIPooling");
        functionRefs = function->clone();
    }
};

TEST_P(ROIPoolingLayerGPUTest, Inference) {
    run();
}

const std::vector<ROIPoolingShapes> inShapes = {
    ROIPoolingShapes{{{}, {{1, 3, 8, 8}}}, {{}, {{1, 5}}}},
    ROIPoolingShapes{{{}, {{1, 3, 8, 8}}}, {{}, {{3, 5}}}},
    ROIPoolingShapes{{{}, {{3, 4, 50, 50}}}, {{}, {{3, 5}}}},
    ROIPoolingShapes{{{}, {{3, 4, 50, 50}}}, {{}, {{5, 5}}}},
    ROIPoolingShapes{
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
    ROIPoolingShapes{
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
    ROIPoolingShapes{
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

const std::vector<ov::element::Type> model_types = {ov::element::f32};

const std::vector<float> spatial_scales = {0.625f, 1.f};

const auto test_ROIPooling_max = ::testing::Combine(::testing::ValuesIn(inShapes),
                                                    ::testing::ValuesIn(pooledShapes_max),
                                                    ::testing::ValuesIn(spatial_scales),
                                                    ::testing::Values(ov::test::utils::ROIPoolingTypes::ROI_MAX),
                                                    ::testing::ValuesIn(model_types));

const auto test_ROIPooling_bilinear = ::testing::Combine(::testing::ValuesIn(inShapes),
                                                         ::testing::ValuesIn(pooledShapes_bilinear),
                                                         ::testing::Values(spatial_scales[1]),
                                                         ::testing::Values(ov::test::utils::ROIPoolingTypes::ROI_BILINEAR),
                                                         ::testing::ValuesIn(model_types));

INSTANTIATE_TEST_SUITE_P(smoke_ROIPoolingGPU_max, ROIPoolingLayerGPUTest,
                        ::testing::Combine(test_ROIPooling_max,
                                           ::testing::Values(ProposalGenerationMode::RANDOM)),
                         ROIPoolingLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ROIPoolingGPU_bilinear, ROIPoolingLayerGPUTest,
                        ::testing::Combine(test_ROIPooling_bilinear,
                                           ::testing::Values(ProposalGenerationMode::RANDOM)),
                        ROIPoolingLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ROIPoolingGPU_bilinear_ultimateRightBorderProposal, ROIPoolingLayerGPUTest,
                        ::testing::Combine(::testing::Combine(::testing::Values(ROIPoolingShapes{{{}, {{1, 1, 50, 50}}}, {{}, {{1, 5}}}}),
                                                              ::testing::Values(std::vector<size_t> { 4, 4 }),
                                                              ::testing::Values(spatial_scales[1]),
                                                              ::testing::Values(ov::test::utils::ROIPoolingTypes::ROI_BILINEAR),
                                                              ::testing::Values(ov::element::f32)),
                                           ::testing::Values(ProposalGenerationMode::ULTIMATE_RIGHT_BORDER)),
                        ROIPoolingLayerGPUTest::getTestCaseName);

} // namespace
