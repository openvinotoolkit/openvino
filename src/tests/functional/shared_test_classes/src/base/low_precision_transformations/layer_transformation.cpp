// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

#include <vector>
#include <string>

#include <ie_core.hpp>
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ov_models/pass/convert_prc.hpp"

using namespace InferenceEngine;
using namespace ngraph;

namespace LayerTestsUtils {
ov::pass::low_precision::LayerTransformation::Params LayerTransformationParamsNGraphFactory::createParamsU8I8AndI8() {
    return ov::pass::low_precision::LayerTransformation::Params();
}

ov::pass::low_precision::LayerTransformation::Params LayerTransformationParamsNGraphFactory::createParamsU8I8() {
    return ov::pass::low_precision::LayerTransformation::Params();
}

ov::pass::low_precision::LayerTransformation::Params LayerTransformationParamsNGraphFactory::createParamsI8I8() {
    return ov::pass::low_precision::LayerTransformation::Params();
}

ov::pass::low_precision::LayerTransformation::Params LayerTransformationParamsNGraphFactory::createParams() {
    return ov::pass::low_precision::LayerTransformation::Params();
}

LayerTransformation::LayerTransformation() {
    rel_threshold = 1.1;
    abs_threshold = 1.0e-4;
    configuration[PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE] = PluginConfigParams::YES;
}

std::pair<float, float> LayerTransformation::getQuantizationInterval(ngraph::element::Type precision) {
    const bool unsignedInterval = precision == ngraph::element::u8;
    const float low = unsignedInterval ? 0.f : -128.f;
    const float hight = unsignedInterval ? 255.f : 127.f;
    return std::make_pair(low, hight);
}

std::string LayerTransformation::toString(const ov::pass::low_precision::LayerTransformation::Params& params) {
    using namespace ov::pass::low_precision;
    std::ostringstream result;
    result <<
        (params.updatePrecisions ? "" : "notUpdatePrecisions_") <<
        params.deqPrecision;

    return result.str();
}

std::string LayerTransformation::get_test_case_name_by_params(
    ov::element::Type precision,
    const ov::PartialShape& inputShapes,
    const std::string& targetDevice,
    const ov::pass::low_precision::LayerTransformation::Params& params) {
    std::ostringstream result;
    result << precision << "_" << inputShapes << "_" << targetDevice << "_" << toString(params);
    return result.str();
}

std::string LayerTransformation::getRuntimePrecision(const std::string& layerName) {
    const ov::CompiledModel& execNet = compiledModel;
    const std::shared_ptr<const ov::Model>& execFunction = execNet.get_runtime_model();

    for (const auto& op : execFunction->get_ops()) {
        const auto name = op->get_friendly_name();
        if (name == layerName) {
            const auto& rtInfo = op->get_rt_info();
            const auto& it = rtInfo.find("runtimePrecision");
            OPENVINO_ASSERT(it != rtInfo.end(), "Runtime precision is not found for node: ", name);
            return it->second.as<std::string>();
        }
    }

    return "";
}

std::string LayerTransformation::getRuntimePrecisionByType(const std::string& layerType) {
    const ov::CompiledModel& execNet = compiledModel;
    const std::shared_ptr<const ov::Model>& execFunction = execNet.get_runtime_model();

    for (const auto& op : execFunction->get_ops()) {
        const auto& rtInfo = op->get_rt_info();
        const auto& typeIt = rtInfo.find("layerType");

        OPENVINO_ASSERT(typeIt != rtInfo.end(), "Layer is not found for type: ", layerType);

        auto type = typeIt->second.as<std::string>();
        if (type == layerType) {
            const auto& it = rtInfo.find("runtimePrecision");
            OPENVINO_ASSERT(it != rtInfo.end(), "Runtime precision is not found for node: ", type);
            return it->second.as<std::string>();
        }
    }

    return "";
}

std::string LayerTransformation::getRuntimePrecisionByFusedName(const std::string& layerName) {
    const ov::CompiledModel& execNet = compiledModel;
    const std::shared_ptr<const ov::Model>& execFunction = execNet.get_runtime_model();

    const auto parse = [](const std::string& originalLayersNames) -> std::set<std::string> {
        std::set<std::string> names;

        std::string tmp = originalLayersNames;
        size_t beginPosition = 0ul;
        size_t endPosition;
        while ((endPosition = tmp.find(",", beginPosition)) != std::string::npos) {
            names.insert(tmp.substr(beginPosition, endPosition - beginPosition));
            beginPosition = endPosition + 1;
        }

        names.insert(tmp.substr(beginPosition, endPosition - beginPosition));
        return names;
    };

    for (const auto& op : execFunction->get_ops()) {
        const auto& rtInfo = op->get_rt_info();

        const auto& nameIt = rtInfo.find("originalLayersNames");
        OPENVINO_ASSERT(nameIt != rtInfo.end(), "originalLayersNames is not found for node: ", layerName);
        const auto fusedName = parse(nameIt->second.as<std::string>());
        if (fusedName.find(layerName) == fusedName.end()) {
            continue;
        }

        const auto& it = rtInfo.find("runtimePrecision");
        OPENVINO_ASSERT(it != rtInfo.end(), "runtimePrecision is not found for node: ", layerName);
        const auto rtPrecisionPtr = it->second.as<std::string>();
        return rtPrecisionPtr;
    }

    return "";
}

std::map<std::string, ngraph::Node::RTMap> LayerTransformation::getRuntimeInfo() {
    const ov::CompiledModel& execNet = compiledModel;
    const std::shared_ptr<const ov::Model>& function = execNet.get_runtime_model();

    std::map<std::string, ngraph::Node::RTMap> runtimeInfo;
    for (const auto& op : function->get_ops()) {
        runtimeInfo[op->get_friendly_name()] = op->get_rt_info();
    }
    return runtimeInfo;
}

void LayerTransformation::init_input_shapes(const ov::PartialShape& shape) {
    std::pair<ov::PartialShape, std::vector<ov::Shape>> input_shapes(shape, { shape.to_shape() });
    SubgraphBaseTest::init_input_shapes({ input_shapes });
}

void LayerTransformation::init_input_shapes(const std::vector<ov::PartialShape>& shapes) {
    std::vector<ov::test::InputShape> input_shapes;
    for (const auto& shape : shapes) {
        std::pair<ov::PartialShape, std::vector<ov::Shape>> tmp_shapes(shape, { shape.to_shape() });
        input_shapes.push_back(tmp_shapes);
    }
    SubgraphBaseTest::init_input_shapes(input_shapes);
}

}  // namespace LayerTestsUtils
