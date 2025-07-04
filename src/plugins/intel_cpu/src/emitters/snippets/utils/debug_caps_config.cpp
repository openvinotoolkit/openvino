// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstdlib>
#ifdef SNIPPETS_DEBUG_CAPS

#    include "debug_caps_config.hpp"
#    include "openvino/util/env_util.hpp"

namespace ov::intel_cpu {

void SnippetsDebugCapsConfig::readProperties() {
    enable_segfault_detector = ov::util::getenv_bool("OV_CPU_SNIPPETS_SEGFAULT_DETECTOR");
}

}  // namespace ov::intel_cpu

#endif  // SNIPPETS_DEBUG_CAPS
