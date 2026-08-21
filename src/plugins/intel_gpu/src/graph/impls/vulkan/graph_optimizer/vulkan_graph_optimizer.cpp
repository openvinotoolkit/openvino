// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_graph_optimizer.hpp"

#include "../eltwise_fusion_policy.hpp"
#include "backend_graph_optimizer.hpp"
#include "fuse_terminal_eltwise.hpp"

namespace cldnn::vulkan {
namespace {

class vulkan_graph_optimizer final : public backend_graph_optimizer {
public:
    explicit vulkan_graph_optimizer(const backend_fusion_policy& fusion_policy)
        : _terminal_eltwise_fusion(fusion_policy) {}

    void optimize(program& program) const override {
        _terminal_eltwise_fusion.run(program);
    }

private:
    fuse_terminal_eltwise _terminal_eltwise_fusion;
};

}  // namespace

void register_graph_optimizer() {
    static const eltwise_fusion_policy fusion_policy;
    static const vulkan_graph_optimizer graph_optimizer{fusion_policy};
    static const backend_fusion_policy_registration fusion_policy_registration{runtime_types::vulkan, fusion_policy};
    static const backend_graph_optimizer_registration graph_optimizer_registration{runtime_types::vulkan,
                                                                                    graph_optimizer};
    (void)fusion_policy_registration;
    (void)graph_optimizer_registration;
}

}  // namespace cldnn::vulkan
