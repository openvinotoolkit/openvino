// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory_state.h"
#include "node.h"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class ScaledDotProductAttention : public Node {
public:
    ScaledDotProductAttention(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::ScaledDotProductAttention;
    }
    // pastkv may have zero dimension
    bool isExecutable() const override {
        return !isInputTensorAtPortEmpty(0) && !isInputTensorAtPortEmpty(1) && !isInputTensorAtPortEmpty(2);
    }
    bool needPrepareParams() const override {
        return false;
    }
    void executeDynamicImpl(dnnl::stream strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    void createPrimitive() override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    enum KernelTypes { KT_REF, KT_ONEDNN, KT_MLAS, KT_ACL};

    void assignState(const std::shared_ptr<VariableStateKVcache>& state, int idx);

    const std::vector<size_t> getKVCacheOrder() const {
        const auto& permute_axes = m_config.config.permute_axes;
        std::vector<size_t> real_order = m_kvstate_layout;
        if (!permute_axes.empty())
            real_order = {permute_axes[2], permute_axes[0], permute_axes[1], permute_axes[3]};
        return real_order;
    }

    ov::element::Type getKVCachePrecision();

private:
    void gatherConcatPastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v, const MemoryPtr& mem_beam_idx);
    void updateBeamTable(const MemoryPtr& mem_beam_idx, size_t new_q_len);
    void updatePastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v);
    ov::element::Type getRuntimePrecision() const override;
    void resetBeamTablePastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v, const MemoryPtr& mem_beam_idx);

    struct Config {
        ScaledDotProductAttentionWithKVCache::Config config;
    };

    struct Executor {
        virtual void execute(dnnl::stream strm, const Config& config, const std::vector<MemoryPtr>& inputs, const MemoryPtr output,
                             const MemoryPtr presentk_input, const MemoryPtr presentv_input, const MemoryPtr beam_input,
                             const PlainTensor& k_scale_zp, const PlainTensor& v_scale_zp) = 0;
        virtual ~Executor() = default;
    };

    Config m_config;
    std::shared_ptr<Executor> m_executor;
    template <KernelTypes KType, typename T> struct AttentionExecutor;
    friend struct ScaledDotProductAttentionKey;

    std::shared_ptr<VariableStateKVcache> m_k_state;
    std::shared_ptr<VariableStateKVcache> m_v_state;
    // KV cache layout
    // (0, 1, 2, 3) for BHLS
    // (2, 0, 1, 3) for LBHS
    std::vector<size_t> m_kvstate_layout = {2, 0, 1, 3};
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
