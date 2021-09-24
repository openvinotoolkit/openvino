// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <condition_variable>
#include <exec_graph_info.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "openvino/runtime/runtime.hpp"
#include "transformations/serialize.hpp"

namespace ov {
namespace test {
using OVExecGraphTests = ov::test::BehaviorTestsBasic;

inline std::vector<std::string> separateStrToVec(const std::string& str, const char sep) {
    std::vector<std::string> result;

    std::istringstream stream(str);
    std::string strVal;

    while (getline(stream, strVal, sep)) {
        result.push_back(strVal);
    }
    return result;
}

// Load correct network to Plugin to get executable network
TEST_P(OVExecGraphTests, canLoadCorrectNetworkToGetExecutable) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::ExecutableNetwork execNet;
    ASSERT_NO_THROW(execNet = ie->compile_model(function, targetDevice, configuration));
}

TEST_P(OVExecGraphTests, CanCreateTwoExeNetworks) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    ov::runtime::ExecutableNetwork execNet;
    for (auto i = 0; i < 2; i++) {
        ASSERT_NO_THROW(execNet = ie->compile_model(function, targetDevice, configuration));
    }
}

TEST_P(OVExecGraphTests, CheckExecGraphInfoBeforeExecution) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<const ov::Function> execGraph;
    // Load CNNNetwork to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    ASSERT_NO_THROW(execGraph = execNet.get_runtime_function());
    // Create InferRequest
    ov::runtime::InferRequest req;
    ASSERT_NO_THROW(req = execNet.create_infer_request());
    // Store all the original layers from the network
    const auto originalLayers = function->get_ops();
    std::map<std::string, int> originalLayersMap;
    for (const auto& layer : originalLayers) {
        originalLayersMap[layer->get_friendly_name()] = 0;
    }
    int IteratorForLayersConstant = 0;

    auto function = execGraph;
    ASSERT_NE(function, nullptr);

    for (const auto& op : function->get_ops()) {
        const auto& rtInfo = op->get_rt_info();

        auto getExecValue = [&rtInfo](const std::string& paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it) << " paramName: " << paramName;
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);

            return value->get();
        };

        // Each layer from the execGraphInfo network must have PM data option set
        ASSERT_EQ("not_executed", getExecValue(ExecGraphInfoSerialization::PERF_COUNTER));
        // Parse origin layer names (fused/merged layers) from the executable graph
        // and compare with layers from the original model
        auto origFromExecLayer = getExecValue(ExecGraphInfoSerialization::ORIGINAL_NAMES);
        if (origFromExecLayer == "")
            IteratorForLayersConstant++;
        std::vector<std::string> origFromExecLayerSep = separateStrToVec(origFromExecLayer, ',');
        std::for_each(origFromExecLayerSep.begin(), origFromExecLayerSep.end(), [&](const std::string& layer) {
            auto origLayer = originalLayersMap.find(layer);
            ASSERT_NE(originalLayersMap.end(), origLayer) << layer;
            origLayer->second++;
        });
    }

    // All layers from the original IR must be present with in ExecGraphInfo
    for (auto& layer : originalLayersMap) {
        if ((layer.second == 0) && (IteratorForLayersConstant > 0)) {
            IteratorForLayersConstant--;
            continue;
        }
        ASSERT_GE(layer.second, 0);
    }
}

TEST_P(OVExecGraphTests, CheckExecGraphInfoAfterExecution) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<const ov::Function> execGraph;
    // Load CNNNetwork to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    ASSERT_NO_THROW(execGraph = execNet.get_runtime_function());
    // Create InferRequest
    ov::runtime::InferRequest req;
    ASSERT_NO_THROW(req = execNet.create_infer_request());
    // Store all the original layers from the network
    const auto originalLayers = function->get_ops();
    std::map<std::string, int> originalLayersMap;
    for (const auto& layer : originalLayers) {
        originalLayersMap[layer->get_friendly_name()] = 0;
    }
    int IteratorForLayersConstant = 0;
    // Store all the layers from the executable graph information represented as CNNNetwork
    bool has_layer_with_valid_time = false;
    auto function = execGraph;
    ASSERT_NE(nullptr, function);

    for (const auto& op : function->get_ops()) {
        const auto& rtInfo = op->get_rt_info();

        auto getExecValue = [&rtInfo](const std::string& paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);

            return value->get();
        };

        // At least one layer in the topology should be executed and have valid perf counter value
        try {
            float x = static_cast<float>(std::atof(getExecValue(ExecGraphInfoSerialization::PERF_COUNTER).c_str()));
            ASSERT_GE(x, 0.0f);
            has_layer_with_valid_time = true;
        } catch (std::exception&) {
        }

        // Parse origin layer names (fused/merged layers) from the executable graph
        // and compare with layers from the original model
        auto origFromExecLayer = getExecValue(ExecGraphInfoSerialization::ORIGINAL_NAMES);
        std::vector<std::string> origFromExecLayerSep = separateStrToVec(origFromExecLayer, ',');
        if (origFromExecLayer == "")
            IteratorForLayersConstant++;
        std::for_each(origFromExecLayerSep.begin(), origFromExecLayerSep.end(), [&](const std::string& layer) {
            auto origLayer = originalLayersMap.find(layer);
            ASSERT_NE(originalLayersMap.end(), origLayer) << layer;
            origLayer->second++;
        });
    }

    ASSERT_TRUE(has_layer_with_valid_time);

    // All layers from the original IR must be present within ExecGraphInfo
    for (auto& layer : originalLayersMap) {
        if ((layer.second == 0) && (IteratorForLayersConstant > 0)) {
            IteratorForLayersConstant--;
            continue;
        }
        ASSERT_GE(layer.second, 0);
    }
}

TEST_P(OVExecGraphTests, CheckExecGraphInfoSerialization) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto ts = CommonTestUtils::GetTimestamp();
    std::string out_xml_path = GetTestName().substr(0, CommonTestUtils::maxFileNameLength) + "_" + ts + ".xml";
    std::string out_bin_path = GetTestName().substr(0, CommonTestUtils::maxFileNameLength) + "_" + ts + ".bin";

    std::shared_ptr<const ov::Function> execGraph;
    // Load CNNNetwork to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    ASSERT_NO_THROW(execGraph = execNet.get_runtime_function());
    // Create InferRequest
    ov::runtime::InferRequest req;
    ASSERT_NO_THROW(req = execNet.create_infer_request());
    // FIXME: ?? const cast to serialize
    ngraph::pass::Serialize(out_xml_path, out_bin_path)
        .run_on_function(std::const_pointer_cast<ov::Function>(execGraph));
    ASSERT_EQ(0, std::remove(out_xml_path.c_str()));
    ASSERT_EQ(0, std::remove(out_bin_path.c_str()));
}
}  // namespace test
}  // namespace ov
