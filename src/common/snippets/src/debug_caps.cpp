// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#include "snippets/debug_caps.hpp"

namespace ov {
namespace snippets {

void DebugCapsConfig::readProperties() {
    auto readEnv = [](const char* envVar) {
        const char* env = std::getenv(envVar);
        if (env && *env)
            return env;

        return (const char*)nullptr;
    };

    const char* envVarValue = nullptr;
    if ((envVarValue = readEnv("OV_SNIPPETS_DUMP_LIR")))
        dumpLIR.parseAndSet(envVarValue);
}

}   // namespace snippets
}   // namespace ov

#endif // SNIPPETS_DEBUG_CAPS
