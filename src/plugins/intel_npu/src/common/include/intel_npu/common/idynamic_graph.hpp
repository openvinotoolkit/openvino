// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/runtime/npu_vm_runtime.hpp"

namespace intel_npu {

class IDynamicGraph : public IGraph {
public:
    IDynamicGraph() = default;
    ~IDynamicGraph() override = default;

    virtual npu_vm_runtime_handle_t get_vm_runtime_handle() const = 0;
    virtual uint64_t get_num_subgraphs() const = 0;
};

}  // namespace intel_npu
