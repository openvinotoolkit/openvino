// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/core_config.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_utils.hpp"
#include "config.h"

void CoreConfiguration(LayerTestsUtils::LayerTestsCommon* test) {
    // Within the test scope we don't need any implicit bf16 optimisations, so let's run the network as is.
    auto& configuration = test->GetConfiguration();
    if (!configuration.count(InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16)) {
        configuration.insert({InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO});
    }
    //fp16 precision is used as default precision on ARM for non-convolution networks
    #if defined(OV_CPU_ARM_ENABLE_FP16)
        if (!configuration.count(ov::hint::inference_precision.name())) {
            auto function = test->GetFunction();
            ov::intel_cpu::Config::ModelType modelType = ov::intel_cpu::getModelType(function);
            if (modelType != ov::intel_cpu::Config::ModelType::CNN) {
                configuration.insert({ov::hint::inference_precision.name(), ov::element::f16.to_string()});
            }
        }
    #endif
}

namespace ov {
namespace test {

void core_configuration(ov::test::SubgraphBaseTest* test) {
    #if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
        if (!test->configuration.count(InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16)) {
            test->configuration.insert({InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO});
        }
    #endif
    #if defined(OV_CPU_ARM_ENABLE_FP16)
        if (!test->configuration.count(ov::hint::inference_precision.name())) {
            auto function = test->compiledModel.get_runtime_model();
            ov::intel_cpu::Config::ModelType modelType = ov::intel_cpu::getModelType(function);
            if (modelType != ov::intel_cpu::Config::ModelType::CNN) {
                test->configuration.insert({ov::hint::inference_precision.name(), ov::element::f16.to_string()});
            }
        }
    #endif
}

} // namespace test
} // namespace ov
