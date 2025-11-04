// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_common.hpp"

namespace cldnn {
namespace ze {

// RAII wrapper for Level Zero module
class ze_module_holder {
public:
    // Take ownership of existing module handle
    explicit ze_module_holder(ze_module_handle_t handle) : m_handle(handle) {}

    ze_module_holder(const ze_module_holder& other) = delete;
    ze_module_holder& operator=(const ze_module_holder& other) = delete;
    ~ze_module_holder() {
        OV_ZE_WARN(zeModuleDestroy(m_handle));
    }
    ze_module_handle_t get_module() { return m_handle; }
private:
    ze_module_handle_t m_handle;

};
}  // namespace ze
}  // namespace cldnn
