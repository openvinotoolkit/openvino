// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#pragma once

#include <string>
#include <cstdlib>

namespace ov {
namespace intel_cpu {

class SnippetsDebugCapsConfig {
public:
    SnippetsDebugCapsConfig() {
        readProperties();
    }

    bool enable_segfault_detector;

private:
    void readProperties();
};

}   // namespace intel_cpu
}   // namespace ov

#endif // SNIPPETS_DEBUG_CAPS
