// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <ie_common.h>
#include <node.h>
#include <memory_state.h>

#include <memory>
#include <string>
#include <vector>

#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class ScaledDotProductAttention : public Node {
public:
    ScaledDotProductAttention(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

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
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    enum KernelTypes { KT_REF, KT_ONEDNN, KT_MLAS};

    void assignState(const std::shared_ptr<VariableStateKVcache>& state, int idx);

    const std::vector<size_t>& getKVCacheOrder() const {
        return m_config.config.permute_axes;
    }

    ov::element::Type getKVCachePrecision();

private:
    void gatherConcatPastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v, const MemoryPtr& mem_beam_idx);
    void updateBeamTable(const MemoryPtr& mem_beam_idx, size_t new_q_len);
    void updatePastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v);
    ov::element::Type getRuntimePrecision() const override;

    struct Config {
        ScaledDotProductAttentionWithKVCache::Config config;
    };

    struct Executor {
        virtual void execute(dnnl::stream strm, const Config& config, const std::vector<MemoryPtr>& inputs, const MemoryPtr output,
                             const MemoryPtr presentk_input, const MemoryPtr presentv_input, const MemoryPtr beam_input) = 0;
    };

    Config m_config;
    std::shared_ptr<Executor> m_executor;
    template <KernelTypes KType, typename T> struct AttentionExecutor;
    friend struct ScaledDotProductAttentionKey;

    std::shared_ptr<VariableStateKVcache> m_k_state;
    std::shared_ptr<VariableStateKVcache> m_v_state;

    ov::element::Type m_kvcache_precision = ov::element::undefined;
    PlainTensor m_tmp_reorder;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
