// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class PagedSelectiveSSM : public Node {
public:
    PagedSelectiveSSM(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::PagedSelectiveSSM;
    }
    bool isExecutable() const override {
        return !isInputTensorAtPortEmpty(3);
    }
    bool needPrepareParams() const override {
        return true;
    }

    void createPrimitive() override;
    void prepareParams() override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    void update_scratchpad();

    MemoryPtr m_state_scratch;
    MemoryPtr m_block_owners;
    size_t m_scratch_head_dim = 0;
    size_t m_scratch_state_size = 0;
    size_t m_cached_physical_blocks = 0;
};

}  // namespace ov::intel_cpu::node
