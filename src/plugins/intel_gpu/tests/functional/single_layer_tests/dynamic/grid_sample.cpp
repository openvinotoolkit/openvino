// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/select.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ov_models/builders.hpp"
#include <string>
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;
using ov::op::v9::GridSample;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>,                 // Input shapes
        GridSample::InterpolationMode,           // Interpolation mode
        GridSample::PaddingMode,                 // Padding mode
        bool,                                    // Align corners
        ElementType,                             // Data precision
        ElementType                              // Grid precision
> GridSampleLayerTestGPUParams;

class GridSampleLayerTestGPU : public testing::WithParamInterface<GridSampleLayerTestGPUParams>,
                          virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GridSampleLayerTestGPUParams> obj) {
        std::vector<InputShape> inputShapes;
        GridSample::InterpolationMode interpolateMode;
        GridSample::PaddingMode paddingMode;
        bool alignCorners;
        ElementType dataPrecision, gridPrecision;

        std::tie(inputShapes, interpolateMode, paddingMode, alignCorners, dataPrecision, gridPrecision) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (size_t i = 0lu; i < inputShapes.size(); i++) {
            result << ov::test::utils::partialShape2str({inputShapes[i].first}) << (i < inputShapes.size() - 1lu ? "_" : "");
        }
        result << ")_TS=";
        for (size_t i = 0lu; i < inputShapes.front().second.size(); i++) {
            result << "{";
            for (size_t j = 0lu; j < inputShapes.size(); j++) {
                result << ov::test::utils::vec2str(inputShapes[j].second[i]) << (j < inputShapes.size() - 1lu ? "_" : "");
            }
            result << "}_";
        }

        result << "interpMode=" << (interpolateMode == GridSample::InterpolationMode::BILINEAR ? "BILINEAR" :
                                    interpolateMode == GridSample::InterpolationMode::BICUBIC ? "BICUBIC" : "NEAREST") << "_";
        result << "padMode=" << (paddingMode == GridSample::PaddingMode::ZEROS ? "ZEROS" :
                                 paddingMode == GridSample::PaddingMode::BORDER ? "BORDER" : "REFLECTION") << "_";
        result << "alignCorners=" << (alignCorners ? "True" : "False") << "_";
        result << "dataPrc=" << dataPrecision << "_";
        result << "gridPrc=" << gridPrecision;

        return result.str();
    }

protected:
    void SetUp() override {
        abs_threshold = 0.0005;
        std::vector<InputShape> inputShapes;
        GridSample::InterpolationMode interpolateMode;
        GridSample::PaddingMode paddingMode;
        bool alignCorners;
        ElementType dataPrecision, gridPrecision;

        std::tie(inputShapes, interpolateMode, paddingMode, alignCorners, dataPrecision, gridPrecision) = this->GetParam();
        targetDevice = ov::test::utils::DEVICE_GPU;
        init_input_shapes(inputShapes);

        if (gridPrecision == ov::element::bf16) {
            rel_threshold = 0.01f;
        }

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(dataPrecision, inputDynamicShapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(gridPrecision, inputDynamicShapes[1])};
        params[0]->set_friendly_name("data");
        params[1]->set_friendly_name("grid");
        GridSample::Attributes attributes = {alignCorners, interpolateMode, paddingMode};
        auto gridSampleNode = std::make_shared<GridSample>(params[0], params[1], attributes);

        ngraph::ResultVector results;
        for (size_t i = 0; i < gridSampleNode->get_output_size(); i++) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(gridSampleNode->output(i)));
        }

        function = std::make_shared<ngraph::Function>(results, params, "GridSampleGPU");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::runtime::Tensor tensor;

            if (funcInput.get_node()->get_friendly_name() == "data") {
                int32_t range = std::accumulate(targetInputStaticShapes[0].begin(), targetInputStaticShapes[0].end(), 1u, std::multiplies<uint32_t>());
                tensor = utils::create_and_fill_tensor(
                        funcInput.get_element_type(), targetInputStaticShapes[0], range, -range / 2, 1);
            } else if (funcInput.get_node()->get_friendly_name() == "grid") {
                int32_t range = std::max(targetInputStaticShapes[0][2], targetInputStaticShapes[0][3]) + 2;
                int32_t resolution = range / 2;
                tensor = utils::create_and_fill_tensor(
                        funcInput.get_element_type(), targetInputStaticShapes[1], range, -1, resolution == 0 ? 1 : resolution);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(GridSampleLayerTestGPU, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

std::vector<GridSample::InterpolationMode> interpolateMode {
        GridSample::InterpolationMode::BILINEAR,
        GridSample::InterpolationMode::BICUBIC,
        GridSample::InterpolationMode::NEAREST };

std::vector<GridSample::PaddingMode> paddingMode {
        GridSample::PaddingMode::ZEROS,
        GridSample::PaddingMode::BORDER,
        GridSample::PaddingMode::REFLECTION };

std::vector<bool> alignCorners { true, false };


const std::vector<std::vector<InputShape>> dynamicInSapes = {
    {
        {
            { 3, -1, -1, -1 }, { {3, 2, 1, 23}, {3, 4, 3, 8}, {3, 6, 5, 5} }
        },
        {
            { -1, -1, -1, 2 }, { {3, 31, 1, 2}, {3, 6, 4, 2}, {3, 23, 1, 2} }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic, GridSampleLayerTestGPU,
        ::testing::Combine(
                ::testing::ValuesIn(dynamicInSapes),
                ::testing::ValuesIn(interpolateMode),
                ::testing::ValuesIn(paddingMode),
                ::testing::ValuesIn(alignCorners),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32)),
        GridSampleLayerTestGPU::getTestCaseName);

} // namespace GPULayerTestsDefinitions
