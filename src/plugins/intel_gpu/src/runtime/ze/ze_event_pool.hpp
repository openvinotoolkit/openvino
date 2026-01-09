// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_engine.hpp"

namespace cldnn {
namespace ze {

// RAII wrapper for Level Zero event pool
struct ze_event_pool {
    ze_event_pool(const ze_engine& engine, uint32_t capacity, ze_event_pool_flags_t flags);
    ~ze_event_pool();
    ze_event_pool(const ze_event_pool&) = delete;
    ze_event_pool& operator=(const ze_event_pool&) = delete;

    using ptr = std::shared_ptr<ze_event_pool>;

    ze_event_pool_handle_t m_handle;
    const ze_engine& m_engine;
};

}  // namespace ze
}  // namespace cldnn

