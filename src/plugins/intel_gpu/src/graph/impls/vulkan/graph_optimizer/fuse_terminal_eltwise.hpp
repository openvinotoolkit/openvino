// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

namespace cldnn {

struct program;
struct program_node;
class backend_fusion_policy;

namespace vulkan {

class fuse_terminal_eltwise {
public:
    explicit fuse_terminal_eltwise(const backend_fusion_policy& fusion_policy) : _fusion_policy(fusion_policy) {}

    void run(program& program) const;

private:
    struct fusion_candidate {
        program_node* producer;
        program_node* output;
        program_node* peer;
    };

    std::optional<fusion_candidate> select_candidate(program& program, program_node& output) const;
    std::optional<fusion_candidate> select_eltwise_candidate(program& program, program_node& output) const;

    const backend_fusion_policy& _fusion_policy;
};

}  // namespace vulkan
}  // namespace cldnn
