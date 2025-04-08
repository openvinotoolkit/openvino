// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

#include <vector>
#include <string>

#include "openvino/util/common_util.hpp"

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
    configuration["LP_TRANSFORMS_MODE"] = true;
}

std::pair<float, float> LayerTransformation::get_quantization_interval(ov::element::Type precision) {
    const bool unsignedInterval = precision == ov::element::u8;
    const float low = unsignedInterval ? 0.f : -128.f;
    const float hight = unsignedInterval ? 255.f : 127.f;
    return std::make_pair(low, hight);
}

std::string LayerTransformation::to_string(const ov::pass::low_precision::LayerTransformation::Params& params) {
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
    result << precision << "_" << inputShapes << "_" << targetDevice << "_" << to_string(params);
    return result.str();
}

namespace {
template <typename IsNodeF>
std::string find_node_by_runtime_property(const ov::CompiledModel& execNet, IsNodeF is_node_f, const std::string& propertyName = "runtimePrecision") {
    const std::shared_ptr<const ov::Model>& execFunction = execNet.get_runtime_model();

    for (const auto& op : execFunction->get_ops()) {
        if (!is_node_f(op))
            continue;
        const ov::RTMap& rtInfo = op->get_rt_info();
        const auto& it = rtInfo.find(propertyName);
        OPENVINO_ASSERT(it != rtInfo.end(), "Runtime property \"", propertyName, "\" is not found for node: ", op->get_friendly_name());
        return it->second.as<std::string>();
    }

    return "";
}
} // namespace

std::string LayerTransformation::get_runtime_precision(const std::string& layerName) {
    auto is_node_f = [layerName](const std::shared_ptr<ov::Node>& op) {
        return op->get_friendly_name() == layerName;
    };
    return find_node_by_runtime_property(compiledModel, is_node_f);
}

std::string LayerTransformation::get_runtime_precision_by_type(const std::string& layerType) {
    auto is_node_f = [layerType](const std::shared_ptr<ov::Node>& op) {
        const auto& rtInfo = op->get_rt_info();
        const auto& typeIt = rtInfo.find("layerType");

        OPENVINO_ASSERT(typeIt != rtInfo.end(), "Layer is not found for type: ", layerType);
        return typeIt->second.as<std::string>() == layerType;
    };
    return find_node_by_runtime_property(compiledModel, is_node_f);
}

std::string LayerTransformation::get_property_by_type(const std::string& layerTypeName, const std::string& propertyName) {
    auto is_node_f = [&layerTypeName](const std::shared_ptr<ov::Node>& op) {
        const auto& rtInfo = op->get_rt_info();
        const auto& typeIt = rtInfo.find("layerType");

        OPENVINO_ASSERT(typeIt != rtInfo.end(), "Layer is not found for type: ", layerTypeName);
        return typeIt->second.as<std::string>() == layerTypeName;
    };
    return ov::util::to_lower(find_node_by_runtime_property(compiledModel, is_node_f, propertyName));
}

namespace {
bool has_layer(const std::string& names, const std::string& layer_name) {
    size_t beginPosition = 0ul;
    size_t endPosition;
    while ((endPosition = names.find(',', beginPosition)) != std::string::npos) {
        if (names.substr(beginPosition, endPosition - beginPosition) == layer_name)
            return true;
        beginPosition = endPosition + 1;
    }

    return names.substr(beginPosition, endPosition - beginPosition) == layer_name;
}
} // namespace

std::string LayerTransformation::get_runtime_precision_by_fused_name(const std::string& layerName) {
    auto is_node_f = [layerName](const std::shared_ptr<ov::Node>& op) {
        const auto& rtInfo = op->get_rt_info();

        const auto& nameIt = rtInfo.find("originalLayersNames");
        OPENVINO_ASSERT(nameIt != rtInfo.end(), "originalLayersNames is not found for node: ", layerName);
        return has_layer(nameIt->second.as<std::string>(), layerName);
    };
    return find_node_by_runtime_property(compiledModel, is_node_f);
}

bool LayerTransformation::check_execution_order(const std::vector<std::string>& orderedOpsTypes) {
    if (orderedOpsTypes.empty()) {
        return true;
    }

    size_t comparisonIndex = 0;
    const std::shared_ptr<const ov::Model>& execFunction = compiledModel.get_runtime_model();
    for (const auto& op : execFunction->get_ordered_ops()) {
        const auto& rtInfo = op->get_rt_info();
        const auto& typeIt = rtInfo.find("layerType");
        OPENVINO_ASSERT(typeIt != rtInfo.end(), "layerType is not found");

        const auto layerType = typeIt->second.as<std::string>();
        if (orderedOpsTypes[comparisonIndex] == layerType) {
            // if comparisonIndex == 0 then start comparision
            // if comparisonIndex != 0 then comparision has been started, check next operation type in sequence
            comparisonIndex++;

            if (comparisonIndex >= orderedOpsTypes.size()) {
                // all operation types in sequence were checked, comparision is ended
                return true;
            }
        } else if (comparisonIndex != 0) {
            // if comparision has been started and operation type is not equal then exit
            return false;
        }
    }

    // actually we can be here only if operation sequence too long
    // (execution graph doesn't have some operations from operations sequence)
    return comparisonIndex == orderedOpsTypes.size();
}

std::map<std::string, ov::Node::RTMap> LayerTransformation::get_runtime_info() {
    const ov::CompiledModel& execNet = compiledModel;
    const std::shared_ptr<const ov::Model>& function = execNet.get_runtime_model();

    std::map<std::string, ov::Node::RTMap> runtimeInfo;
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
    auto input_shapes = ov::test::static_partial_shapes_to_test_representation(shapes);
    SubgraphBaseTest::init_input_shapes(input_shapes);
}

}  // namespace LayerTestsUtils
