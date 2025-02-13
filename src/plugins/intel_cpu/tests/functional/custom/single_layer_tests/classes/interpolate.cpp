// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate.hpp"
#include "gtest/gtest.h"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"
#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/convert_interpolate11_downgrade.hpp"

using namespace CPUTestUtils;
using namespace ov::intel_cpu;

namespace ov {
namespace test {

std::string InterpolateLayerCPUTest::getTestCaseName(testing::TestParamInfo<InterpolateLayerCPUTestParamsSet> obj) {
    InterpolateSpecificParams specificParams;
    ShapeParams shapeParams;
    ElementType prec;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    ov::AnyMap additionalConfig;
    std::tie(specificParams, shapeParams, prec, cpuParams, fusingParams, additionalConfig) = obj.param;

    ov::op::v11::Interpolate::InterpolateMode mode;
    ov::op::v11::Interpolate::CoordinateTransformMode transfMode;
    ov::op::v11::Interpolate::NearestMode nearMode;
    bool antiAlias;
    std::vector<size_t> padBegin;
    std::vector<size_t> padEnd;
    double cubeCoef;
    std::tie(mode, transfMode, nearMode, antiAlias, padBegin, padEnd, cubeCoef) = specificParams;

    ov::op::v11::Interpolate::ShapeCalcMode shapeCalcMode;
    InputShape inputShapes;
    ov::test::utils::InputLayerType shapeInputType;
    std::vector<std::vector<float>> shapeDataForInput;
    std::vector<int64_t> axes;
    std::tie(shapeCalcMode, inputShapes, shapeInputType, shapeDataForInput, axes) = shapeParams;

    using ov::test::utils::operator<<;
    std::ostringstream result;
    result << "ShapeCalcMode=" << shapeCalcMode << "_";
    result << "IS=";
    result << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
    result << "TS=";
    for (const auto& shape : inputShapes.second) {
        result << ov::test::utils::vec2str(shape) << "_";
    }
    if (shapeCalcMode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES) {
        result << "Scales=";
    } else {
        result << "Sizes=";
    }
    for (const auto &data : shapeDataForInput) {
        result << ov::test::utils::vec2str(data) << "_";
    }
    result << shapeInputType << "_";
    result << "InterpolateMode=" << mode << "_";
    result << "CoordinateTransformMode=" << transfMode << "_";
    result << "NearestMode=" << nearMode << "_";
    result << "CubeCoef=" << cubeCoef << "_";
    result << "Antialias=" << antiAlias << "_";
    result << "PB=" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE=" << ov::test::utils::vec2str(padEnd) << "_";
    result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "PRC=" << prec << "_";

    result << CPUTestsBase::getTestCaseName(cpuParams);
    result << CpuTestWithFusing::getTestCaseName(fusingParams);

    if (!additionalConfig.empty()) {
        result << "_PluginConf";
        for (auto& item : additionalConfig) {
            result << "_" << item.first << "=" << item.second.as<std::string>();
        }
    }

    return result.str();
}

void InterpolateLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor;

        if (i == 1) {
            if (shapeCalcMode == ov::op::v4::Interpolate::ShapeCalcMode::SIZES) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], sizes[inferRequestNum].data());
            } else {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], scales[inferRequestNum].data());
            }
        } else {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 2560;
            in_data.resolution = 256;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
        }

        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
    inferRequestNum++;
}

void InterpolateLayerCPUTest::configure_model() {
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

void InterpolateLayerCPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    InterpolateSpecificParams specificParams;
    ShapeParams shapeParams;
    ElementType ngPrc;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    ov::AnyMap additionalConfig;
    std::tie(specificParams, shapeParams, ngPrc, cpuParams, fusingParams, additionalConfig) = this->GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    ov::op::v11::Interpolate::InterpolateMode mode;
    ov::op::v11::Interpolate::CoordinateTransformMode transfMode;
    ov::op::v11::Interpolate::NearestMode nearMode;
    bool antiAlias;
    std::vector<size_t> padBegin;
    std::vector<size_t> padEnd;
    double cubeCoef;
    std::tie(mode, transfMode, nearMode, antiAlias, padBegin, padEnd, cubeCoef) = specificParams;

    InputShape dataShape;
    ov::test::utils::InputLayerType shapeInputType;
    std::vector<std::vector<float>> shapeDataForInput;
    std::vector<int64_t> axes;
    std::tie(shapeCalcMode, dataShape, shapeInputType, shapeDataForInput, axes) = shapeParams;

    if (shapeCalcMode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES) {
        scales = shapeDataForInput;
    } else {
        sizes.resize(shapeDataForInput.size());
        for (size_t i = 0; i < shapeDataForInput.size(); i++) {
            for (size_t j = 0; j < shapeDataForInput[i].size(); j++) {
                sizes[i].push_back(shapeDataForInput[i][j]);
            }
        }
    }

    std::vector<InputShape> inputShapes;
    inputShapes.push_back(dataShape);
    if (shapeInputType == ov::test::utils::InputLayerType::PARAMETER) {
        inputShapes.push_back(InputShape({static_cast<int64_t>(axes.size())}, std::vector<ov::Shape>(dataShape.second.size(), {axes.size()})));
    }

    auto it = additionalConfig.find(ov::hint::inference_precision.name());
    if (it != additionalConfig.end() && it->second.as<ov::element::Type>() == ov::element::bf16) {
        inType = outType = ngPrc = ElementType::bf16;
        rel_threshold = 1e-2f;
    } else if (it != additionalConfig.end() && it->second.as<ov::element::Type>() == ov::element::f16) {
        inType = outType = ngPrc = ElementType::f16;
        rel_threshold = 1e-2f;
    } else {
        inType = outType = ngPrc;
    }

    init_input_shapes(inputShapes);

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, inputDynamicShapes.front())};
    std::shared_ptr<ov::Node> sizesInput, scalesInput;
    if (shapeCalcMode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES) {
        if (shapeInputType == ov::test::utils::InputLayerType::PARAMETER) {
            auto paramNode = std::make_shared<ov::op::v0::Parameter>(ElementType::f32, ov::Shape{scales.front().size()});
            params.push_back(paramNode);
            scalesInput = paramNode;
        } else {
            scalesInput = std::make_shared<ov::op::v0::Constant>(ElementType::f32, ov::Shape{scales.front().size()}, scales.front());
        }
    } else {
        if (shapeInputType == ov::test::utils::InputLayerType::PARAMETER) {
            auto paramNode = std::make_shared<ov::op::v0::Parameter>(ElementType::i32, ov::Shape{sizes.front().size()});
            params.push_back(paramNode);
            sizesInput = paramNode;
        } else {
            sizesInput = std::make_shared<ov::op::v0::Constant>(ElementType::i32, ov::Shape{sizes.front().size()}, sizes.front());
        }
    }
    auto axesInput = std::make_shared<ov::op::v0::Constant>(ElementType::i64, ov::Shape{axes.size()}, axes);

    for (size_t i = 0; i < params.size(); i++) {
        params[i]->set_friendly_name(std::string("param_") + std::to_string(i));
    }

    ov::op::v11::Interpolate::InterpolateAttrs interpAttr{mode, shapeCalcMode, padBegin, padEnd, transfMode, nearMode,
                                                          antiAlias, cubeCoef};

    std::shared_ptr<ov::op::v11::Interpolate> interp = nullptr;
    if (shapeCalcMode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES) {
        interp = std::make_shared<ov::op::v11::Interpolate>(params[0], scalesInput, axesInput, interpAttr);
    } else {
        interp = std::make_shared<ov::op::v11::Interpolate>(params[0], sizesInput, axesInput, interpAttr);
    }

    function = makeNgraphFunction(ngPrc, params, interp, "InterpolateCPU");

    ov::pass::Manager m;
    m.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    m.run_passes(function);
    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }
    selectedType = makeSelectedTypeStr(selectedType, ngPrc);
}

TEST_P(InterpolateLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Interpolate");
}


namespace Interpolate {
const std::vector<ov::op::v11::Interpolate::NearestMode> defNearestModes() {
    static const std::vector<ov::op::v11::Interpolate::NearestMode> defNearestModes {
        ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR
    };
    return defNearestModes;
}

const std::vector<bool> antialias() {
    return { false };
}

const std::vector<double> cubeCoefs() {
    return { -0.75f };
}

}  // namespace Interpolate
}  // namespace test
}  // namespace ov
