// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "intel_gpu/runtime/engine_configuration.hpp"

namespace cldnn {

struct program_node;

enum class fusion_kind : uint8_t {
    activation,
    quantize,
    eltwise,
    terminal_reorder,
    reorder_elimination,
};

enum class fusion_decision : uint8_t {
    defer_to_common,
    reject,
    accept,
};

struct fusion_query {
    fusion_kind kind;
    const program_node& producer;
    const program_node& consumer;
    const program_node* peer = nullptr;
};

/// Backend-specific compile-time fusion policy. Queries are resolved while the
/// graph is built and never enter the inference dispatch path.
class backend_fusion_policy {
public:
    virtual ~backend_fusion_policy() = default;

    virtual bool limits_program_to_simple_fusions() const noexcept {
        return false;
    }

    virtual bool controls(fusion_kind kind) const noexcept {
        return false;
    }

    virtual fusion_decision evaluate(const fusion_query& query) const {
        return fusion_decision::defer_to_common;
    }
};

void register_backend_fusion_policy(runtime_types runtime, const backend_fusion_policy& policy);
const backend_fusion_policy& get_backend_fusion_policy(runtime_types runtime) noexcept;

class backend_fusion_policy_registration {
public:
    backend_fusion_policy_registration(runtime_types runtime, const backend_fusion_policy& policy) {
        register_backend_fusion_policy(runtime, policy);
    }
};

}  // namespace cldnn
