// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "graph_context.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

// Executes ov::op::internal::NGramHashFunction.
class NgramHash : public Node {
public:
    NgramHash(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    bool needPrepareParams() const override {
        return false;
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    struct JitKernel;

    template <typename T>
    void executeImpl();

    std::shared_ptr<JitKernel> m_jit_kernel;
    size_t m_ngram_size = 0;
    std::vector<int64_t> m_heads_per_order;
    std::vector<int64_t> m_multipliers;
    std::vector<int64_t> m_head_vocab_sizes;
    std::vector<int64_t> m_head_offsets;
    ov::element::Type m_precision;
};

}  // namespace ov::intel_cpu::node
