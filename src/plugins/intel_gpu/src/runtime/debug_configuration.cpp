// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include <iostream>
#include <fstream>

namespace ov::intel_gpu {
std::ostream& get_verbose_stream() {
#ifdef GPU_DEBUG_CONFIG
    if (ExecutionConfig::get_log_to_file().length() > 0) {
        static std::ofstream fout;
        if (!fout.is_open())
            fout.open(ExecutionConfig::get_log_to_file());
        return fout;
    } else {
        return std::cout;
    }
#else
    return std::cout;
#endif
}
}  // namespace ov::intel_gpu
