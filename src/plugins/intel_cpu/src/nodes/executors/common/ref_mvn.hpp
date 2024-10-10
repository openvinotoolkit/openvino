// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include "cpu_memory.h"
#include "nodes/executors/mvn_config.hpp"

namespace ov {
namespace intel_cpu {

class CommonMVNExecutor : public Executor {
public:
    CommonMVNExecutor(const MVNAttrs& attrs,
                      const PostOps& postOps,
                      const MemoryArgs& memory,
                      const ExecutorContext::CPtr context) : refMVNAttrs(attrs) {}

    void execute(const MemoryArgs& memory) override;

    impl_desc_type implType() const override {
        return impl_desc_type::ref;
    }

    // offloads execution data preparation from the exec call
    bool update(const MemoryArgs& memory) override;

    static bool supports(const MVNConfig& config);

private:
    MVNAttrs refMVNAttrs;
    VectorDims shape5D;
    void mvn_ref(const uint8_t *in_ptr_, uint8_t *out_ptr_, const VectorDims& shape5d);
};

}  // namespace intel_cpu
}  // namespace ov