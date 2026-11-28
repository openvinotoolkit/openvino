// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/interpolate.hpp"

#include <cstring>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

using InterpolateSpecificParams = std::tuple<ov::op::v4::Interpolate::InterpolateMode,          // InterpolateMode
                                             ov::op::v4::Interpolate::CoordinateTransformMode,  // CoordinateTransformMode
                                             ov::op::v4::Interpolate::NearestMode,              // NearestMode
                                             bool,                                                  // AntiAlias
                                             std::vector<size_t>,                                   // PadBegin
                                             std::vector<size_t>,                                   // PadEnd
                                             double>;                                               // Cube coef

using ShapeParams = std::tuple<ov::op::v4::Interpolate::ShapeCalcMode, // ShapeCalculationMode
                               InputShape,                                 // Input shapes
                               // params describing input, choice of which depends on ShapeCalcMode
                               ov::test::utils::InputLayerType,            // sizes input type
                               ov::test::utils::InputLayerType,            // scales input type
                               std::vector<std::vector<float>>,            // scales or sizes values
                               std::vector<int64_t>>;                      // axes

using InterpolateLayerGPUTestParamsSet = std::tuple<InterpolateSpecificParams,
                                                    ShapeParams,
                                                    ov::element::Type,
                                                    bool>;                 // use Interpolate_v11

class InterpolateLayerGPUTest : public testing::WithParamInterface<InterpolateLayerGPUTestParamsSet>,
                                virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateLayerGPUTestParamsSet>& obj) {
        std::map<std::string, std::string> additionalConfig;
        const auto& [specificParams, shapeParams, prec, useInterpolateV11] = obj.param;

        const auto& [mode, transfMode, nearMode, antiAlias, padBegin, padEnd, cubeCoef] = specificParams;

        const auto& [shapeCalcMode, inputShapes, sizesInputType, scalesInputType, shapeDataForInput, axes] = shapeParams;

        std::ostringstream result;
        using ov::operator<<;
        result << "ShapeCalcMode=" << shapeCalcMode << "_";
        result << "IS=";
        result << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
        result << "TS=";
        for (const auto& shape : inputShapes.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        if (shapeCalcMode == ov::op::v4::Interpolate::ShapeCalcMode::SCALES) {
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

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 0) {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = 2560;
                in_data.resolution = 256;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            } else if (i == 1) {
                if (shapeCalcMode == ov::op::v4::Interpolate::ShapeCalcMode::SIZES || funcInputs.size() == 3) {
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
                if (inType != ov::element::Type_t::dynamic) {
                    p.input(i).tensor().set_element_type(inType);
                }
            }
        }
        {
            auto results = function->get_results();
            for (size_t i = 0; i < results.size(); i++) {
                if (outType != ov::element::Type_t::dynamic) {
                    p.output(i).tensor().set_element_type(outType);
                }
            }
        }
        function = p.build();
    }

protected:
    std::vector<std::vector<float>> scales;
    std::vector<std::vector<int32_t>> sizes;
    ov::op::v4::Interpolate::ShapeCalcMode shapeCalcMode;
    size_t inferRequestNum = 0;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [specificParams, shapeParams, ngPrc, useInterpolateV11] = this->GetParam();

        const auto& [mode, transfMode, nearMode, antiAlias, padBegin, padEnd, cubeCoef] = specificParams;

        if (mode == ov::op::v4::Interpolate::InterpolateMode::CUBIC && ngPrc == ov::element::f32) {
            abs_threshold = 1e-3;
        }

        const auto& [_shapeCalcMode, dataShape, sizesInputType, scalesInputType, shapeDataForInput, axes] = shapeParams;
        shapeCalcMode = _shapeCalcMode;

        if (shapeCalcMode == ov::op::v4::Interpolate::ShapeCalcMode::SCALES) {
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
        if (sizesInputType == ov::test::utils::InputLayerType::PARAMETER) {
            inputShapes.push_back(InputShape({static_cast<int64_t>(axes.size())}, std::vector<ov::Shape>(dataShape.second.size(), {axes.size()})));
        }
        if (scalesInputType == ov::test::utils::InputLayerType::PARAMETER) {
            inputShapes.push_back(InputShape({static_cast<int64_t>(axes.size())}, std::vector<ov::Shape>(dataShape.second.size(), {axes.size()})));
        }

        init_input_shapes(inputShapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, inputDynamicShapes.front())};

        std::shared_ptr<ov::Node> sizesInput, scalesInput;
        if (shapeCalcMode == ov::op::v4::Interpolate::ShapeCalcMode::SCALES) {
            if (scalesInputType == ov::test::utils::InputLayerType::PARAMETER) {
                auto paramNode = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{scales.front().size()});
                params.push_back(paramNode);
                scalesInput = paramNode;
            } else {
                scalesInput = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{scales.front().size()}, scales.front());
            }
            if (sizesInputType == ov::test::utils::InputLayerType::PARAMETER) {
                auto paramNode = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{sizes.front().size()});
                params.push_back(paramNode);
                sizesInput = paramNode;
            } else {
                sizesInput = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{sizes.front().size()}, sizes.front());
            }
        } else {
            if (sizesInputType == ov::test::utils::InputLayerType::PARAMETER) {
                auto paramNode = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{sizes.front().size()});
                params.push_back(paramNode);
                sizesInput = paramNode;
            } else {
                sizesInput = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{sizes.front().size()}, sizes.front());
            }
            if (scalesInputType == ov::test::utils::InputLayerType::PARAMETER) {
                auto paramNode = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{scales.front().size()});
                params.push_back(paramNode);
                scalesInput = paramNode;
            } else {
                scalesInput = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{scales.front().size()}, scales.front());
            }
        }

        auto axesInput = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);

        for (size_t i = 0; i < params.size(); i++) {
            params[i]->set_friendly_name(std::string("param_") + std::to_string(i));
        }

        ov::op::v4::Interpolate::InterpolateAttrs interpAttr{mode, shapeCalcMode, padBegin, padEnd, transfMode, nearMode,
                                                                            antiAlias, cubeCoef};
        std::shared_ptr<ov::op::Op> interpolate;
        bool scalesMode = shapeCalcMode == ov::op::v4::Interpolate::ShapeCalcMode::SCALES;
        if (useInterpolateV11) {
            if (axes.size() != dataShape.first.size()) {
                interpolate = std::make_shared<ov::op::v11::Interpolate>(params[0],
                                                                             scalesMode ? scalesInput : sizesInput,
                                                                             axesInput,
                                                                             interpAttr);
            } else {
                interpolate = std::make_shared<ov::op::v11::Interpolate>(params[0],
                                                                             scalesMode ? scalesInput : sizesInput,
                                                                             interpAttr);
            }
        } else {
            interpolate = std::make_shared<ov::op::v4::Interpolate>(params[0],
                                                                        sizesInput,
                                                                        scalesInput,
                                                                        axesInput,
                                                                        interpAttr);
        }

        ov::ResultVector results;
        for (size_t i = 0; i < interpolate->get_output_size(); ++i) {
            results.push_back(std::make_shared<ov::op::v0::Result>(interpolate->output(i)));
        }
        function = std::make_shared<ov::Model>(results, params, "InterpolateGPU");
    }
};

TEST_P(InterpolateLayerGPUTest, Inference) {
    run();
}

const std::vector<ov::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes_Smoke = {
        ov::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ov::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
};

const std::vector<ov::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes_Full = {
        ov::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
        ov::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ov::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ov::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
        ov::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

const std::vector<ShapeParams> rank3EquivalenceShapes = {
    {ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
     InputShape{{2, 18, 101}, {{2, 18, 101}}},
     ov::test::utils::InputLayerType::CONSTANT,
     ov::test::utils::InputLayerType::CONSTANT,
     {{1.f, 1.f, 160.f}},
     {0, 1, 2}},
    {ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
     InputShape{{-1, 18, -1}, {{1, 18, 7}, {2, 18, 11}}},
     ov::test::utils::InputLayerType::CONSTANT,
     ov::test::utils::InputLayerType::CONSTANT,
     {{31.f}, {31.f}},
     {2}},
};

class InterpolateRank3EquivalenceGPUTest : public InterpolateLayerGPUTest {
protected:
    void SetUp() override {
        InterpolateLayerGPUTest::SetUp();
        configuration.insert(ov::hint::inference_precision(std::get<2>(GetParam())));
    }

    void compile_model() override {
        InterpolateLayerGPUTest::compile_model();
        const auto original = ov::as_type_ptr<ov::op::util::InterpolateBase>(function->get_result()->input_value(0).get_node_shared_ptr());
        OPENVINO_ASSERT(original, "Rank-3 equivalence expects an Interpolate result");
        const auto data =
            std::make_shared<ov::op::v0::Parameter>(function->get_parameters()[0]->get_element_type(), function->get_parameters()[0]->get_partial_shape());
        const auto singleton_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        const auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(data, singleton_axis);
        const auto axes = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});
        const bool scales_mode = shapeCalcMode == ov::op::v4::Interpolate::ShapeCalcMode::SCALES;
        const auto values = scales_mode ? ov::op::v0::Constant::create(ov::element::f32, {2}, std::vector<float>{1.f, scales.front().back()})
                                        : ov::op::v0::Constant::create(ov::element::i32, {2}, std::vector<int32_t>{1, sizes.front().back()});
        auto attrs = original->get_attrs();
        attrs.pads_begin = attrs.pads_end = {0, 0, 0, 0};
        std::shared_ptr<ov::Node> interpolate;
        if (std::get<3>(GetParam())) {
            interpolate = std::make_shared<ov::op::v11::Interpolate>(unsqueeze, values, axes, attrs);
        } else {
            const auto unused = ov::op::v0::Constant::create(scales_mode ? ov::element::i32 : ov::element::f32, {2}, {1, 1});
            interpolate = std::make_shared<ov::op::v4::Interpolate>(unsqueeze, scales_mode ? unused : values, scales_mode ? values : unused, axes, attrs);
        }
        const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(interpolate, singleton_axis);
        const auto explicit_rank4 = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{data});
        rank4_request = core->compile_model(explicit_rank4, targetDevice, configuration).create_infer_request();
    }

    std::vector<ov::Tensor> calculate_refs() override {
        // Compare two GPU paths, not GPU arithmetic against the reference evaluator.
        rank4_request.set_input_tensor(inputs.at(function->get_parameters()[0]));
        rank4_request.infer();
        return {rank4_request.get_output_tensor()};
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), 1);
        ASSERT_EQ(actual.size(), 1);
        const auto& input_shape = inputs.at(function->get_parameters()[0]).get_shape();
        const auto length = shapeCalcMode == ov::op::v4::Interpolate::ShapeCalcMode::SCALES ? static_cast<size_t>(input_shape[2] * scales.front().back())
                                                                                            : static_cast<size_t>(sizes.front().back());
        const ov::Shape output_shape{input_shape[0], input_shape[1], length};
        ASSERT_EQ(actual[0].get_shape(), output_shape);
        ASSERT_EQ(expected[0].get_shape(), output_shape);
        ASSERT_EQ(actual[0].get_element_type(), expected[0].get_element_type());
        ASSERT_EQ(actual[0].get_byte_size(), expected[0].get_byte_size());
        EXPECT_EQ(std::memcmp(actual[0].data(), expected[0].data(), actual[0].get_byte_size()), 0)
            << "Normalized rank-3 and explicit rank-4 GPU outputs must be bit-identical";
    }

private:
    ov::InferRequest rank4_request;
};

TEST_P(InterpolateRank3EquivalenceGPUTest, MatchesExplicitRank4) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateRank3Equivalence,
                         InterpolateRank3EquivalenceGPUTest,
                         testing::Combine(testing::Combine(testing::Values(ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX),
                                                           testing::Values(ov::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL),
                                                           testing::Values(ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR),
                                                           testing::Values(false),
                                                           testing::Values(std::vector<size_t>{0, 0, 0}),
                                                           testing::Values(std::vector<size_t>{0, 0, 0}),
                                                           testing::Values(-0.75)),
                                          testing::ValuesIn(rank3EquivalenceShapes),
                                          testing::Values(ov::element::f32, ov::element::f16),
                                          testing::Bool()),
                         InterpolateLayerGPUTest::getTestCaseName);

const std::vector<ov::op::v4::Interpolate::NearestMode> nearestModes_Smoke = {
        ov::op::v4::Interpolate::NearestMode::SIMPLE,
        ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ov::op::v4::Interpolate::NearestMode::FLOOR,
};

const std::vector<ov::op::v4::Interpolate::NearestMode> nearestModes_Full = {
        ov::op::v4::Interpolate::NearestMode::SIMPLE,
        ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ov::op::v4::Interpolate::NearestMode::FLOOR,
        ov::op::v4::Interpolate::NearestMode::CEIL,
        ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_CEIL,
};

const std::vector<ov::op::v4::Interpolate::NearestMode> defNearestModes = {
        ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
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
        ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f}, {1.f, 1.f, 1.25f, 1.25f}, {1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {1, 10}, -1, -1}, {{1, 2, 12, 20}}},
        ov::test::utils::InputLayerType::PARAMETER,
        ov::test::utils::InputLayerType::PARAMETER,
         {{1.f, 1.f, 0.5f, 2.0f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {1, 10}, -1, -1}, {{1, 2, 12, 20}}},
        ov::test::utils::InputLayerType::PARAMETER,
        ov::test::utils::InputLayerType::PARAMETER,
        {{0.5f, 2.0f}},
        reducedAxes4D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        ov::test::utils::InputLayerType::CONSTANT,
        {{1, 11, 5, 6}, {2, 7, 8, 7}, {1, 11, 5, 6}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {1, 10}, -1, -1}, {{1, 2, 12, 20}}},
        ov::test::utils::InputLayerType::PARAMETER,
        ov::test::utils::InputLayerType::PARAMETER,
        {{1, 2, 24, 10}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {1, 10}, -1, -1}, {{1, 2, 12, 20}}},
        ov::test::utils::InputLayerType::PARAMETER,
        ov::test::utils::InputLayerType::PARAMETER,
        {{24, 10}},
        reducedAxes4D.front()
    }
};

const std::vector<ShapeParams> shapeParams4D_Full = {
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::CONSTANT,
        {{1, 11, 5, 6}},
        defaultAxes4D.front()
    }
};

const std::vector<ShapeParams> shapeParams4DReducedAxis_Full = {
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::CONSTANT,
        {{1, 11, 5, 6}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::CONSTANT,
        {{1.5f}},
        reducedAxes4D.back()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::CONSTANT,
        {{6}},
        reducedAxes4D.back()
    }
};

const auto interpolateCasesNN_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesNN_Full = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::NEAREST),
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
            ::testing::Values(ov::element::f32),
            ::testing::Values(true, false)),
    InterpolateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_Test, InterpolateLayerGPUTest,
         ::testing::Combine(
            interpolateCasesNN_Full,
            ::testing::ValuesIn(shapeParams4DReducedAxis_Full),
            ::testing::Values(ov::element::f32),
            ::testing::Values(true, false)),
     InterpolateLayerGPUTest::getTestCaseName);

const auto interpolateCasesLinearOnnx_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesLinearOnnx_Full = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX),
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
            ::testing::Values(ov::element::f32),
            ::testing::Values(false)),
    InterpolateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ov::element::f32),
            ::testing::Values(true, false)),
    InterpolateLayerGPUTest::getTestCaseName);

const auto interpolateCasesLinear_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::LINEAR),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesLinear_Full = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::LINEAR),
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
            ::testing::Values(ov::element::f32),
            ::testing::Values(false)),
    InterpolateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinear_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesLinear_Full,
            ::testing::ValuesIn(shapeParams4DReducedAxis_Full),
            ::testing::Values(ov::element::f32),
            ::testing::Values(true, false)),
    InterpolateLayerGPUTest::getTestCaseName);

const auto interpolateCasesCubic_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::CUBIC),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesCubic_Full = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::CUBIC),
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
            ::testing::Values(ov::element::f32),
            ::testing::Values(false)),
    InterpolateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateCubic_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesCubic_Full,
            ::testing::ValuesIn(shapeParams4DReducedAxis_Full),
            ::testing::Values(ov::element::f32),
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
        ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f, 0.5f}, {1.f, 1.f, 1.25f, 1.25f, 1.25f}, {1.f, 1.f, 1.25f, 1.5f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 10}, -1, -1, -1}, {{1, 4, 2, 3, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        ov::test::utils::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.5f, 2.f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        ov::test::utils::InputLayerType::CONSTANT,
        {{1, 11, 5, 6, 2}, {2, 7, 8, 7, 4}, {1, 11, 5, 6, 2}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 10}, -1, -1, -1}, {{1, 4, 2, 3, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        ov::test::utils::InputLayerType::PARAMETER,
        {{1, 4, 4, 1, 6}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 10}, -1, -1, -1}, {{1, 4, 2, 3, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        ov::test::utils::InputLayerType::PARAMETER,
        {{4, 1, 6}},
        reducedAxes5D.front()
    },
};

const std::vector<ShapeParams> shapeParams5D_Full = {
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {1, 11, 5, 5, 8}, {1, 11, 4, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::CONSTANT,
        {{1, 11, 5, 6, 4}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {1, 11, 5, 5, 8}, {1, 11, 4, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::CONSTANT,
        {{1, 6, 4}},
        reducedAxes5D.front()
    }
};

const auto interpolateCasesLinearOnnx5D_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesLinearOnnx5D_Full = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX),
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
            ::testing::Values(ov::element::f32),
            ::testing::Values(false)),
    InterpolateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx5D_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx5D_Full,
            ::testing::ValuesIn(shapeParams5D_Full),
            ::testing::Values(ov::element::f32),
            ::testing::Values(true, false)),
    InterpolateLayerGPUTest::getTestCaseName);

const auto interpolateCasesNN5D_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesNN5D_Full = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::NEAREST),
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
            ::testing::Values(ov::element::f32),
            ::testing::Values(true, false)),
    InterpolateLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN5D_Layout_Test, InterpolateLayerGPUTest,
        ::testing::Combine(
            interpolateCasesNN5D_Full,
            ::testing::ValuesIn(shapeParams5D_Full),
            ::testing::Values(ov::element::f32),
            ::testing::Values(true, false)),
    InterpolateLayerGPUTest::getTestCaseName);


//
// Test cache save/load for Interpolate V4/V11
//
class InterpolateLayerCacheGPUTest : public InterpolateLayerGPUTest {
};

const std::vector<ShapeParams> shapeParams4D_Smoke_single = {
    ShapeParams{
        ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    }
};

const std::vector<std::vector<size_t>> pads4D_cache = {
        {0, 0, 0, 0},
};

const auto interpolateCasesNNCache_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::NEAREST),
        ::testing::Values(ov::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL),
        ::testing::Values(ov::op::v4::Interpolate::NearestMode::SIMPLE),
        ::testing::Values(false),
        ::testing::ValuesIn(pads4D_cache),
        ::testing::ValuesIn(pads4D_cache),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_Test, InterpolateLayerCacheGPUTest,
        ::testing::Combine(
             interpolateCasesNNCache_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke_single),
            ::testing::Values(ov::element::f32),
            ::testing::Values(true, false)),
    InterpolateLayerCacheGPUTest::getTestCaseName);

TEST_P(InterpolateLayerCacheGPUTest, InferenceCache) {
    std::string cacheDirName = ov::test::utils::generateTestFilePrefix() + "_gpu_model_cache";

    ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
    ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
    ov::test::utils::removeDir(cacheDirName);
    core->set_property(ov::cache_dir(cacheDirName));
    compile_model();
    run();

    ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
    ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
    ov::test::utils::removeDir(cacheDirName);
}

} // namespace
