// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/core_config.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

void CoreConfiguration(LayerTestsUtils::LayerTestsCommon* test) {
    // Within the test scope we don't need any implicit bf16 optimisations, so let's run the network as is.
    auto& configuration = test->GetConfiguration();
    if (!configuration.count(InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16)) {
        configuration.insert({InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO});
    }
    #if defined(OV_CPU_ARM_ENABLE_FP16)
        //force fp32 inference precision if it is not configured specially
        if (!configuration.count(ov::hint::inference_precision.name())) {
            configuration.insert({ov::hint::inference_precision.name(), ov::element::f32.to_string()});
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
        //force fp32 inference precision if it is not configured specially
        if (!test->configuration.count(ov::hint::inference_precision.name())) {
            test->configuration.insert({ov::hint::inference_precision.name(), ov::element::f32.to_string()});
        }
    #endif
    // todo: issue: 123320
    test->convert_precisions = {{ ov::element::bf16, ov::element::f32 }, { ov::element::f16, ov::element::f32 }};
}

} // namespace test
} // namespace ov
