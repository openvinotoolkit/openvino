// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

namespace cldnn {

struct program;
struct program_node;

namespace vulkan {

class eltwise_fusion_policy;

class fuse_eltwise {
public:
    explicit fuse_eltwise(const eltwise_fusion_policy& policy) : _policy(policy) {}

    void run(program& program) const;

private:
    struct fusion_candidate {
        program_node* producer;
        program_node* consumer;
        program_node* peer;
    };

    std::optional<fusion_candidate> select_candidate(program& program, program_node& consumer) const;
    std::optional<fusion_candidate> select_eltwise_candidate(program& program, program_node& consumer) const;

    const eltwise_fusion_policy& _policy;
};

}  // namespace vulkan
}  // namespace cldnn
