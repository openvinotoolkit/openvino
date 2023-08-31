// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

#include <vector>
#include <string>

#include <ie_core.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

using namespace InferenceEngine;
using namespace ngraph;

namespace LayerTestsUtils {
ngraph::pass::low_precision::LayerTransformation::Params LayerTransformationParamsNGraphFactory::createParamsU8I8AndI8() {
    return ngraph::pass::low_precision::LayerTransformation::Params();
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformationParamsNGraphFactory::createParamsU8I8() {
    return ngraph::pass::low_precision::LayerTransformation::Params();
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformationParamsNGraphFactory::createParamsI8I8() {
    return ngraph::pass::low_precision::LayerTransformation::Params();
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformationParamsNGraphFactory::createParams() {
    return ngraph::pass::low_precision::LayerTransformation::Params();
}

LayerTransformation::LayerTransformation() {
    rel_threshold = 0.05;
    configuration[PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE] = PluginConfigParams::YES;
}

namespace {
ov::runtime::Tensor generate_default(const std::shared_ptr<ngraph::Node>& node,
                                     size_t port,
                                     const ov::element::Type& elemType,
                                     const ov::Shape& targetShape) {
    return ov::test::utils::create_and_fill_tensor(elemType, targetShape, 10, 0.0);
}

template<typename ExpectedT, typename ActualT>
void compare_tensors(const ov::Tensor& expected_tensor,
                     const ov::Tensor& actual_tensor,
                     const double abs_threshold_ = std::numeric_limits<double>::max(),
                     const double rel_threshold_ = std::numeric_limits<double>::max()) {
    auto expected_shape = expected_tensor.get_shape();
    auto actual_shape = actual_tensor.get_shape();
    if (expected_shape != actual_shape) {
        std::ostringstream out_stream;
        out_stream << "Expected and actual shape are different: " << expected_shape << " " << actual_shape;
        throw  std::runtime_error(out_stream.str());
    }
    if (shape_size(actual_shape) == 0) {
        return;
    }

    auto expected = expected_tensor.data<ExpectedT>();
    auto actual = actual_tensor.data<ActualT>();
    double abs_threshold = abs_threshold_;
    double threshold = rel_threshold_;

    size_t shape_size_cnt = shape_size(expected_shape);
    for (std::size_t i = 0; i < shape_size_cnt; ++i) {
        const auto& ref = expected[i];
        const auto& res = actual[i];
        const auto absoluteDifference = ov::test::utils::ie_abs(res - ref);
        if (abs_threshold > 0.f && absoluteDifference > abs_threshold) {
            IE_THROW() << "Absolute comparison of values expected: " << std::to_string(ref) << " and actual: " << std::to_string(res)
                << " at index " << i << " with absolute threshold " << abs_threshold
                << " failed";
        }
        if (absoluteDifference <= threshold) {
            continue;
        }

        double max;
        if (ov::test::utils::ie_abs(res) > ov::test::utils::ie_abs(ref)) {
            max = ov::test::utils::ie_abs(res);
        } else {
            max = ov::test::utils::ie_abs(ref);
        }

        double diff = static_cast<float>(absoluteDifference) / max;
        if (max == 0 || (diff > static_cast<float>(threshold)) ||
            (std::isnan(static_cast<float>(res)) ^ std::isnan(static_cast<float>(ref)))) {
            IE_THROW() << "Relative comparison of values expected: " << std::to_string(ref) << " and actual: " << std::to_string(res)
                << " at index " << i << " with threshold " << threshold
                << " failed";
        }
    }
}

void compare_default(const std::shared_ptr<ov::Node>& node,
                     size_t port,
                     const ov::runtime::Tensor& expected,
                     const ov::runtime::Tensor& actual,
                     double absThreshold,
                     double relThreshold) {
    switch (expected.get_element_type()) {
        case ov::element::f16: {
            switch (actual.get_element_type()) {
                case ov::element::f16: {
                    compare_tensors<float16, float16>(expected, actual, absThreshold, relThreshold);
                    break;
                }
                case ov::element::f32: {
                    compare_tensors<float16, float>(expected, actual, absThreshold, relThreshold);
                    break;
                }
                default: {
                    IE_THROW() << "actual precision is not supported: " << actual.get_element_type();
                }
            }
            break;
        }
        case ov::element::f32: {
            switch (actual.get_element_type()) {
                case ov::element::f16: {
                    compare_tensors<float, float16>(expected, actual, absThreshold, relThreshold);
                    break;
                }
                case ov::element::f32: {
                    compare_tensors<float, float>(expected, actual, absThreshold, relThreshold);
                    break;
                }
                default: {
                    IE_THROW() << "actual precision is not supported: " << actual.get_element_type();
                }
            }
            break;
        }
        default: {
            IE_THROW() << "expected precision is not supported: " << expected.get_element_type();
        }
    }
}

} // namespace

ov::test::utils::InputsMap LayerTransformation::get_input_map() {
    static ov::test::utils::InputsMap inputs_map{
        { ov::op::Op::get_type_info_static(), generate_default }
    };
    return inputs_map;
}

ov::test::utils::CompareMap LayerTransformation::get_compare_map() {
    static ov::test::utils::CompareMap compare_map{
        { ov::op::Op::get_type_info_static(), compare_default }
    };
    return compare_map;
}

std::pair<double, double> LayerTransformation::getQuantizationInterval(const ngraph::element::Type precision) {
    const bool unsignedInterval = precision == ngraph::element::u8;
    const double low = unsignedInterval ? 0.0 : -128.0;
    const double hight = unsignedInterval ? 255.0 : 127.0;
    return std::make_pair(low, hight);
}

std::string LayerTransformation::toString(const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    using namespace ngraph::pass::low_precision;
    std::ostringstream result;
    result <<
        (params.updatePrecisions ? "" : "notUpdatePrecisions_") <<
        params.deqPrecision;

    return result.str();
}

std::string LayerTransformation::getTestCaseNameByParams(
    const InferenceEngine::Precision precision,
    const InferenceEngine::SizeVector& inputShapes,
    const std::string& targetDevice,
    const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    std::ostringstream result;
    result << precision.name() << "_" << ngraph::Shape(inputShapes) << "_" << targetDevice << "_" << toString(params);
    return result.str();
}

std::string LayerTransformation::getTestCaseNameByParams(
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShapes,
    const std::string& targetDevice,
    const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    std::ostringstream result;
    result << precision << "_" << inputShapes << "_" << targetDevice << "_" << toString(params);
    return result.str();
}

std::string LayerTransformation::getRuntimePrecision(const std::string& layerName) {
    const ov::CompiledModel& execNet = compiledModel;
    const std::shared_ptr<const ov::Model>& function = execNet.get_runtime_model();
    const auto& execFunction = function;

    for (const auto& op : execFunction->get_ops()) {
        const auto name = op->get_friendly_name();
        if (name == layerName) {
            const auto& rtInfo = op->get_rt_info();
            const auto& it = rtInfo.find("runtimePrecision");
            IE_ASSERT(it != rtInfo.end()) << "Runtime precision is not found for node: " << name;
            return it->second.as<std::string>();
        }
    }

    return "";
}

std::string LayerTransformation::getRuntimePrecisionByType(const std::string& layerType) {
    const ov::CompiledModel& execNet = compiledModel;
    const std::shared_ptr<const ov::Model>& function = execNet.get_runtime_model();
    const auto& execFunction = function;

    for (const auto& op : execFunction->get_ops()) {
        const auto& rtInfo = op->get_rt_info();
        const auto& typeIt = rtInfo.find("layerType");

        IE_ASSERT(typeIt != rtInfo.end()) << "Layer is not found for type: " << layerType;

        auto type = typeIt->second.as<std::string>();
        if (type == layerType) {
            const auto& it = rtInfo.find("runtimePrecision");
            IE_ASSERT(it != rtInfo.end()) << "Runtime precision is not found for node: " << type;
            return it->second.as<std::string>();
        }
    }

    return "";
}

std::string LayerTransformation::getRuntimePrecisionByFusedName(const std::string& layerName) {
    const ov::CompiledModel& execNet = compiledModel;
    const std::shared_ptr<const ov::Model>& function = execNet.get_runtime_model();
    const auto& execFunction = function;

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
        IE_ASSERT(nameIt != rtInfo.end()) << "originalLayersNames is not found for node: " << layerName;
        const auto fusedName = parse(nameIt->second.as<std::string>());
        if (fusedName.find(layerName) == fusedName.end()) {
            continue;
        }

        const auto& it = rtInfo.find("runtimePrecision");
        IE_ASSERT(it != rtInfo.end()) << "runtimePrecision is not found for node: " << layerName;
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
