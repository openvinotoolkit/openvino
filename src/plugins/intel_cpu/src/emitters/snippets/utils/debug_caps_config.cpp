// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#    include "debug_caps_config.hpp"

namespace ov::intel_cpu {

void SnippetsDebugCapsConfig::readProperties() {
    auto readEnv = [](const char* envVar) {
        const char* env = std::getenv(envVar);
        if ((env != nullptr) && (*env != 0)) {
            return env;
        }

        return static_cast<const char*>(nullptr);
    };

    enable_segfault_detector = (readEnv("OV_CPU_SNIPPETS_SEGFAULT_DETECTOR") != nullptr) ? true : false;
}

}  // namespace ov::intel_cpu

#endif  // SNIPPETS_DEBUG_CAPS
