// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#    pragma once

#    include <cstdlib>
#    include <string>

namespace ov::intel_cpu {

class SnippetsDebugCapsConfig {
public:
    SnippetsDebugCapsConfig() {
        readProperties();
    }

    bool enable_segfault_detector;

private:
    void readProperties();
};

}  // namespace ov::intel_cpu

#endif  // SNIPPETS_DEBUG_CAPS
