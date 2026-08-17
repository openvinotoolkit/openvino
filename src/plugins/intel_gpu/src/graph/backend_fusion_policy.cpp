// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend_fusion_policy.hpp"

#include <map>

#include "openvino/core/except.hpp"

namespace cldnn {
namespace {

class common_fusion_policy final : public backend_fusion_policy {};

auto& registered_policies() {
    static std::map<runtime_types, const backend_fusion_policy*> policies;
    return policies;
}

}  // namespace

void register_backend_fusion_policy(runtime_types runtime, const backend_fusion_policy& policy) {
    auto& registered = registered_policies()[runtime];
    OPENVINO_ASSERT(registered == nullptr || registered == &policy, "[GPU] A fusion policy is already registered for runtime ", runtime);
    registered = &policy;
}

const backend_fusion_policy& get_backend_fusion_policy(runtime_types runtime) noexcept {
    static const common_fusion_policy common_policy;
    const auto& policies = registered_policies();
    const auto it = policies.find(runtime);
    return it == policies.end() ? common_policy : *it->second;
}

}  // namespace cldnn
