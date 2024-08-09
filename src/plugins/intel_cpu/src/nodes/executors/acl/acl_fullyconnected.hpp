// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "cpu_memory.h"
#include "nodes/executors/fullyconnected_config.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov {
namespace intel_cpu {

class ACLFCExecutor : public Executor {
public:
    ACLFCExecutor(const FCAttrs& attrs,
                  const PostOps& postOps,
                  const MemoryArgs& memory,
                  const ExecutorContext::CPtr context);

    void execute(const MemoryArgs& memory) override;

    impl_desc_type implType() const override {
        return impl_desc_type::acl;
    }

    // offloads execution data preparation from the exec call
    bool update(const MemoryArgs& memory) override;

    static bool supports(const FCConfig& config);

    void moveMemToNumaNode(int numaNodeID) override;

private:
    const FCAttrs& m_attrs;
    const MemoryArgs& m_memoryArgs;
    const MemoryCPtr packedWeights;
    int64_t M, N, K;
    int curNumaNode = -1;
};

using ACLFCExecutorPtr = std::shared_ptr<ACLFCExecutor>;

}  // namespace intel_cpu
}  // namespace ov
