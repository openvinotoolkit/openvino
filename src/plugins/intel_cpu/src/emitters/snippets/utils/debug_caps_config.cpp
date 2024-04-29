// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#include "debug_caps_config.hpp"

namespace ov {
namespace intel_cpu {

void SnippetsDebugCapsConfig::readProperties() {
    auto readEnv = [](const char* envVar) {
        const char* env = std::getenv(envVar);
        if (env && *env)
            return env;

        return (const char*)nullptr;
    };

    enable_segfault_detector = readEnv("OV_CPU_SNIPPETS_SEGFAULT_DETECTOR") ? true : false;
    const char* envVarValue = nullptr;
    if ((envVarValue = readEnv("OV_CPU_SNIPPETS_LIR_PATH")))
        LIRSerializePath = envVarValue;
}

}   // namespace intel_cpu
}   // namespace ov

#endif // SNIPPETS_DEBUG_CAPS
