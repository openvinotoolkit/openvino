// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/interpolate.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "openvino/core/preprocess/pre_post_process.hpp"

using namespace ov::test;
using ngraph::helpers::operator<<;

namespace GPULayerTestsDefinitions {

using InterpolateSpecificParams = std::tuple<ngraph::op::v4::Interpolate::InterpolateMode,          // InterpolateMode
                                             ngraph::op::v4::Interpolate::CoordinateTransformMode,  // CoordinateTransformMode
                                             ngraph::op::v4::Interpolate::NearestMode,              // NearestMode
                                             bool,                                                  // AntiAlias
                                             std::vector<size_t>,                                   // PadBegin
                                             std::vector<size_t>,                                   // PadEnd
                                             double>;                                               // Cube coef

using ShapeParams = std::tuple<ngraph::op::v4::Interpolate::ShapeCalcMode, // ShapeCalculationMode
                               InputShape,                                 // Input shapes
                               // params describing input, choice of which depends on ShapeCalcMode
                               ngraph::helpers::InputLayerType,            // sizes input type
                               ngraph::helpers::InputLayerType,            // scales input type
                               std::vector<std::vector<float>>,            // scales or sizes values
                               std::vector<int64_t>>;                      // axes

using InterpolateLayerGPUTestParamsSet = std::tuple<InterpolateSpecificParams,
                                                    ShapeParams,
                                                    ElementType,
                                                    bool>;                 // use Interpolate_v11

class InterpolateLayerGPUTest : public testing::WithParamInterface<InterpolateLayerGPUTestParamsSet>,
                                virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateLayerGPUTestParamsSet> obj) {
        InterpolateSpecificParams specificParams;
        ShapeParams shapeParams;
        ElementType prec;
        bool useInterpolateV11;
        std::map<std::string, std::string> additionalConfig;
        std::tie(specificParams, shapeParams, prec, useInterpolateV11) = obj.param;

        ngraph::op::v4::Interpolate::InterpolateMode mode;
        ngraph::op::v4::Interpolate::CoordinateTransformMode transfMode;
        ngraph::op::v4::Interpolate::NearestMode nearMode;
        bool antiAlias;
        std::vector<size_t> padBegin;
        std::vector<size_t> padEnd;
        double cubeCoef;
        std::tie(mode, transfMode, nearMode, antiAlias, padBegin, padEnd, cubeCoef) = specificParams;

        ngraph::op::v4::Interpolate::ShapeCalcMode shapeCalcMode;
        InputShape inputShapes;
        ngraph::helpers::InputLayerType sizesInputType;
        ngraph::helpers::InputLayerType scalesInputType;
        std::vector<std::vector<float>> shapeDataForInput;
        std::vector<int64_t> axes;
        std::tie(shapeCalcMode, inputShapes, sizesInputType, scalesInputType, shapeDataForInput, axes) = shapeParams;

        std::ostringstream result;
        result << "ShapeCalcMode=" << shapeCalcMode << "_";
        result << "IS=";
        result << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
        result << "TS=";
        for (const auto& shape : inputShapes.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        if (shapeCalcMode == ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES) {
            result << "Scales=";
        } else {
            result << "Sizes=";
        }
        for (const auto &data : shapeDataForInput) {
            result << ov::test::utils::vec2str(data) << "_";
        }
        result << "sizesInputType=" << sizesInputType << "_";
        result << "scalesInputType=" << scalesInputType << "_";
        result << "InterpolateMode=" << mode << "_";
        result << "CoordinateTransformMode=" << transfMode << "_";
        result << "NearestMode=" << nearMode << "_";
        result << "CubeCoef=" << cubeCoef << "_";
        result << "Antialias=" << antiAlias << "_";
        result << "PB=" << ov::test::utils::vec2str(padBegin) << "_";
        result << "PE=" << ov::test::utils::vec2str(padEnd) << "_";
        result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
        result << "PRC=" << prec << "_";
        result << "v11=" << useInterpolateV11 << "_";

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second;
            }
        }

        return result.str();
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 0) {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 2560, 0, 256);
            } else if (i == 1) {
                if (shapeCalcMode == ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES || funcInputs.size() == 3) {
                    tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], sizes[inferRequestNum].data());
                } else {
                    tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], scales[inferRequestNum].data());
                }
            } else {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], scales[inferRequestNum].data());
            }

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
        inferRequestNum++;
    }

    void configure_model() override {
        ov::preprocess::PrePostProcessor p(function);
        {
            auto& params = function->get_parameters();
            for (size_t i = 0; i < params.size(); i++) {
                if (i > 0) {
                    continue;
                }
                if (inType != ov::element::Type_t::undefined) {
                    p.input(i).tensor().set_element_type(inType);
                }
            }
        }
        {
            auto results = function->get_results();
            for (size_t i = 0; i < results.size(); i++) {
                if (outType != ov::element::Type_t::undefined) {
                    p.output(i).tensor().set_element_type(outType);
                }
            }
        }
        function = p.build();
    }

protected:
    std::vector<std::vector<float>> scales;
    std::vector<std::vector<int32_t>> sizes;
    ngraph::op::v4::Interpolate::ShapeCalcMode shapeCalcMode;
    size_t inferRequestNum = 0;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        InterpolateSpecificParams specificParams;
        ShapeParams shapeParams;
        ElementType ngPrc;
        bool useInterpolateV11;
        std::tie(specificParams, shapeParams, ngPrc, useInterpolateV11) = this->GetParam();

        ngraph::op::v4::Interpolate::InterpolateMode mode;
        ngraph::op::v4::Interpolate::CoordinateTransformMode transfMode;
        ngraph::op::v4::Interpolate::NearestMode nearMode;
        bool antiAlias;
        std::vector<size_t> padBegin;
        std::vector<size_t> padEnd;
        double cubeCoef;
        std::tie(mode, transfMode, nearMode, antiAlias, padBegin, padEnd, cubeCoef) = specificParams;

        InputShape dataShape;
        ngraph::helpers::InputLayerType sizesInputType;
        ngraph::helpers::InputLayerType scalesInputType;
        std::vector<std::vector<float>> shapeDataForInput;
        std::vector<int64_t> axes;
        std::tie(shapeCalcMode, dataShape, sizesInputType, scalesInputType, shapeDataForInput, axes) = shapeParams;

        if (shapeCalcMode == ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES) {
            scales = shapeDataForInput;
            sizes.resize(scales.size(), std::vector<int32_t>(scales.front().size(), 0));
        } else {
            sizes.resize(shapeDataForInput.size());
            for (size_t i = 0; i < shapeDataForInput.size(); i++) {
                for (size_t j = 0; j < shapeDataForInput[i].size(); j++) {
                    sizes[i].push_back(shapeDataForInput[i][j]);
                }
            }
            scales.resize(sizes.size(), std::vector<float>(sizes.front().size(), 0));
        }

        std::vector<InputShape> inputShapes;
        inputShapes.push_back(dataShape);
        if (sizesInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            inputShapes.push_back(InputShape({static_cast<int64_t>(axes.size())}, std::vector<ov::Shape>(dataShape.second.size(), {axes.size()})));
        }
        if (scalesInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            inputShapes.push_back(InputShape({static_cast<int64_t>(axes.size())}, std::vector<ov::Shape>(dataShape.second.size(), {axes.size()})));
        }

        init_input_shapes(inputShapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, inputDynamicShapes.front())};

        std::shared_ptr<ov::Node> sizesInput, scalesInput;
        if (shapeCalcMode == ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES) {
            if (scalesInputType == ngraph::helpers::InputLayerType::PARAMETER) {
                auto paramNode = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::Type_t::f32, ov::Shape{scales.front().size()});
                params.push_back(paramNode);
                scalesInput = paramNode;
            } else {
                scalesInput = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::f32, ov::Shape{scales.front().size()}, scales.front());
            }
            if (sizesInputType == ngraph::helpers::InputLayerType::PARAMETER) {
                auto paramNode = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::Type_t::i32, ov::Shape{sizes.front().size()});
                params.push_back(paramNode);
                sizesInput = paramNode;
            } else {
                sizesInput = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i32, ov::Shape{sizes.front().size()}, sizes.front());
            }
        } else {
            if (sizesInputType == ngraph::helpers::InputLayerType::PARAMETER) {
                auto paramNode = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::Type_t::i32, ov::Shape{sizes.front().size()});
                params.push_back(paramNode);
                sizesInput = paramNode;
            } else {
                sizesInput = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i32, ov::Shape{sizes.front().size()}, sizes.front());
            }
            if (scalesInputType == ngraph::helpers::InputLayerType::PARAMETER) {
                auto paramNode = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::Type_t::f32, ov::Shape{scales.front().size()});
                params.push_back(paramNode);
                scalesInput = paramNode;
            } else {
                scalesInput = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::f32, ov::Shape{scales.front().size()}, scales.front());
            }
        }

        auto axesInput = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ov::Shape{axes.size()}, axes);

        for (size_t i = 0; i < params.size(); i++) {
            params[i]->set_friendly_name(std::string("param_") + std::to_string(i));
        }

        ngraph::op::v4::Interpolate::InterpolateAttrs interpAttr{mode, shapeCalcMode, padBegin, padEnd, transfMode, nearMode,
                                                                            antiAlias, cubeCoef};
        std::shared_ptr<ngraph::op::Op> interpolate;
        bool scalesMode = shapeCalcMode == ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES;
        if (useInterpolateV11) {
            if (axes.size() != dataShape.first.size()) {
                interpolate = std::make_shared<ngraph::op::v11::Interpolate>(params[0],
                                                                             scalesMode ? scalesInput : sizesInput,
                                                                             axesInput,
                                                                             interpAttr);
            } else {
                interpolate = std::make_shared<ngraph::op::v11::Interpolate>(params[0],
                                                                             scalesMode ? scalesInput : sizesInput,
                                                                             interpAttr);
            }
        } else {
            interpolate = std::make_shared<ngraph::op::v4::Interpolate>(params[0],
                                                                        sizesInput,
                                                                        scalesInput,
                                                                        axesInput,
                                                                        interpAttr);
        }

        ngraph::ResultVector results;
        for (size_t i = 0; i < interpolate->get_output_size(); ++i) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(interpolate->output(i)));
        }
        function = std::make_shared<ngraph::Function>(results, params, "InterpolateGPU");
    }
};

TEST_P(InterpolateLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes_Smoke = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes_Full = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes_Smoke = {
        ngraph::op::v4::Interpolate::NearestMode::SIMPLE,
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ngraph::op::v4::Interpolate::NearestMode::FLOOR,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes_Full = {
        ngraph::op::v4::Interpolate::NearestMode::SIMPLE,
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ngraph::op::v4::Interpolate::NearestMode::FLOOR,
        ngraph::op::v4::Interpolate::NearestMode::CEIL,
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_CEIL,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defNearestModes = {
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
};

const std::vector<bool> antialias = {
        false,
};

const std::vector<double> cubeCoefs = {
        -0.75f,
};

const std::vector<std::vector<size_t>> pads4D = {
        {0, 0, 0, 0},
        {0, 0, 1, 1},
};

const std::vector<std::vector<int64_t>> defaultAxes4D = {
    {0, 1, 2, 3}
};

const std::vector<std::vector<int64_t>> reducedAxes4D = {
    {2, 3}, {3}
};

const std::vector<ShapeParams> shapeParams4D_Smoke = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f}, {1.f, 1.f, 1.25f, 1.25f}, {1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {1, 10}, -1, -1}, {{1, 2, 12, 20}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        ngraph::helpers::InputLayerType::PARAMETER,
         {{1.f, 1.f, 0.5f, 2.0f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {1, 10}, -1, -1}, {{1, 2, 12, 20}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        ngraph::helpers::InputLayerType::PARAMETER,
        {{0.5f, 2.0f}},
        reducedAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 11, 5, 6}, {2, 7, 8, 7}, {1, 11, 5, 6}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {1, 10}, -1, -1}, {{1, 2, 12, 20}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1, 2, 24, 10}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {1, 10}, -1, -1}, {{1, 2, 12, 20}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        ngraph::helpers::InputLayerType::PARAMETER,
        {{24, 10}},
        reducedAxes4D.front()
    }
};

const std::vector<ShapeParams> shapeParams4D_Full = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 11, 5, 6}},
        defaultAxes4D.front()
    }
};

const std::vector<ShapeParams> shapeParams4DReducedAxis_Full = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 11, 5, 6}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.5f}},
        reducedAxes4D.back()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::CONSTANT,
        {{6}},
        reducedAxes4D.back()
    }
};

const auto interpolateCasesNN_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesNN_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(nearestModes_Full),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
             interpolateCasesNN_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::Values(true, false)),
    InterpolateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_Test, InterpolateLayerGPUTest,
         ::testing::Combine(
            interpolateCasesNN_Full,
            ::testing::ValuesIn(shapeParams4DReducedAxis_Full),
            ::testing::Values(ElementType::f32),
            ::testing::Values(true, false)),
     InterpolateLayerGPUTest::getTestCaseName);

const auto interpolateCasesLinearOnnx_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesLinearOnnx_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinearOnnx_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::Values(false)),
    InterpolateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::Values(true, false)),
    InterpolateLayerGPUTest::getTestCaseName);

const auto interpolateCasesLinear_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::LINEAR),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesLinear_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::LINEAR),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinear_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesLinear_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::Values(false)),
    InterpolateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinear_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesLinear_Full,
            ::testing::ValuesIn(shapeParams4DReducedAxis_Full),
            ::testing::Values(ElementType::f32),
            ::testing::Values(true, false)),
    InterpolateLayerGPUTest::getTestCaseName);

const auto interpolateCasesCubic_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::CUBIC),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesCubic_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::CUBIC),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateCubic_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesCubic_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::Values(false)),
    InterpolateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateCubic_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesCubic_Full,
            ::testing::ValuesIn(shapeParams4DReducedAxis_Full),
            ::testing::Values(ElementType::f32),
            ::testing::Values(true, false)),
    InterpolateLayerGPUTest::getTestCaseName);

////////////////////////5D/////////////////////////////

const std::vector<std::vector<size_t>> pads5D = {
        {0, 0, 0, 0, 0}
};

const std::vector<std::vector<int64_t>> defaultAxes5D = {
    {0, 1, 2, 3, 4}
};

const std::vector<std::vector<int64_t>> reducedAxes5D = {
    {2, 3, 4}
};

const std::vector<ShapeParams> shapeParams5D_Smoke = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f, 0.5f}, {1.f, 1.f, 1.25f, 1.25f, 1.25f}, {1.f, 1.f, 1.25f, 1.5f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 10}, -1, -1, -1}, {{1, 4, 2, 3, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.5f, 2.f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 11, 5, 6, 2}, {2, 7, 8, 7, 4}, {1, 11, 5, 6, 2}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 10}, -1, -1, -1}, {{1, 4, 2, 3, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1, 4, 4, 1, 6}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 10}, -1, -1, -1}, {{1, 4, 2, 3, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        ngraph::helpers::InputLayerType::PARAMETER,
        {{4, 1, 6}},
        reducedAxes5D.front()
    },
};

const std::vector<ShapeParams> shapeParams5D_Full = {
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {1, 11, 5, 5, 8}, {1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 11, 5, 6, 4}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {1, 11, 5, 5, 8}, {1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 6, 4}},
        reducedAxes5D.front()
    }
};

const auto interpolateCasesLinearOnnx5D_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesLinearOnnx5D_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinearOnnx5D_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx5D_Smoke,
            ::testing::ValuesIn(shapeParams5D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::Values(false)),
    InterpolateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx5D_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx5D_Full,
            ::testing::ValuesIn(shapeParams5D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::Values(true, false)),
    InterpolateLayerGPUTest::getTestCaseName);

const auto interpolateCasesNN5D_Smoke = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesNN5D_Full = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(nearestModes_Full),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN5D_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesNN5D_Smoke,
            ::testing::ValuesIn(shapeParams5D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::Values(true, false)),
    InterpolateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN5D_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesNN5D_Full,
            ::testing::ValuesIn(shapeParams5D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::Values(true, false)),
    InterpolateLayerGPUTest::getTestCaseName);

} // namespace

} // namespace GPULayerTestsDefinitions
