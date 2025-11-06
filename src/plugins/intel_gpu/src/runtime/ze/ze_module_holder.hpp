// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_common.hpp"
#include <vector>
#include <string>

namespace cldnn {
namespace ze {

// RAII wrapper for Level Zero module
class ze_module_holder {
public:
    // Take ownership of existing module and build log handles
    explicit ze_module_holder(ze_module_handle_t module, ze_module_build_log_handle_t build_log) : m_module(module), m_build_log(build_log) {}

    ze_module_holder(const ze_module_holder& other) = delete;
    ze_module_holder& operator=(const ze_module_holder& other) = delete;
    ~ze_module_holder() {
        OV_ZE_WARN(zeModuleBuildLogDestroy(m_build_log));
        OV_ZE_WARN(zeModuleDestroy(m_module));
    }
    ze_module_handle_t get_module_handle() const { return m_module; }
    ze_module_build_log_handle_t get_build_log_handle() const { return m_build_log; }

private:
    ze_module_handle_t m_module;
    ze_module_build_log_handle_t m_build_log;
};
}  // namespace ze
}  // namespace cldnn
