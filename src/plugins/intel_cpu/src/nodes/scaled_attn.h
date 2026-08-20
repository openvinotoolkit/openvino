// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "kernels/scaled_attn/cache_spec.hpp"
#include "kernels/scaled_attn/codecs/codec_kernels.hpp"
#include "kernels/scaled_attn/mha_kv_cache_codec.hpp"
#include "memory_state.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node {

class ScaledDotProductAttention : public Node {
public:
    ScaledDotProductAttention(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::ScaledDotProductAttention;
    }

    bool neverExecute() const override {
        return getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(0) ||
               getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(1) ||
               getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(2);
    }
    // pastkv may have zero dimension
    bool isExecutable() const override {
        return !isInputTensorAtPortEmpty(0) && !isInputTensorAtPortEmpty(1) && !isInputTensorAtPortEmpty(2);
    }
    bool needPrepareParams() const override {
        return false;
    }
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    void createPrimitive() override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    enum KernelTypes : uint8_t { KT_REF, KT_ONEDNN, KT_MLAS, KT_ACL };

    void assignState(const std::shared_ptr<VariableStateKVcache>& state, int idx);

    std::vector<size_t> getKVCacheOrder() const {
        const auto& permute_axes = m_config.config.permute_axes;
        std::vector<size_t> real_order = m_kvstate_layout;
        if (!permute_axes.empty()) {
            real_order = {permute_axes[2], permute_axes[0], permute_axes[1], permute_axes[3]};
        }
        return real_order;
    }
    ov::element::Type getKeyCachePrecision();
    ov::element::Type getValueCachePrecision();
    // Legacy: returns key-side precision. Kept for graph_dumper serialization.
    ov::element::Type getKVCachePrecision() {
        return getKeyCachePrecision();
    }
    const ov::Extensions::Cpu::CacheSpec& getKeySpec() const {
        return m_key_spec;
    }
    const ov::Extensions::Cpu::CacheSpec& getValueSpec() const {
        return m_value_spec;
    }

private:
    void gatherConcatPastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v, const MemoryPtr& mem_beam_idx);
    void updateBeamTable(const MemoryPtr& mem_beam_idx, size_t L1);
    void updatePastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v);
    ov::element::Type getRuntimePrecision() const override;
    void resetBeamTablePastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v, const MemoryPtr& mem_beam_idx);
    // Derive per-thread scratch {base, stride} (f32 slots) from m_per_thread_head_scratch.
    // Indexed as ws[tid] to get per-thread buffer start. {nullptr, 0} when non-codec.
    ov::Extensions::Cpu::StridedData<float> get_per_thread_scratch() const;

    struct Config {
        ScaledDotProductAttentionWithKVCache::Config config;
    };

    struct Executor {
        virtual void execute(const dnnl::stream& strm,
                             const Config& config,
                             const std::vector<MemoryPtr>& inputs,
                             MemoryPtr output,
                             MemoryPtr presentk_input,
                             MemoryPtr presentv_input,
                             MemoryPtr beam_input,
                             const PlainTensor& k_scale_zp,
                             const PlainTensor& v_scale_zp,
                             const ov::Extensions::Cpu::CacheSpec& k_spec,
                             const ov::Extensions::Cpu::CacheSpec& v_spec,
                             float* per_thread_head_scratch,
                             size_t per_thread_head_stride,
                             const PlainTensor& k_quant_meta_data,
                             const PlainTensor& v_quant_meta_data,
                             const PlainTensor& wht_signs) = 0;
        [[nodiscard]] virtual impl_desc_type implType() const = 0;
        virtual ~Executor() = default;
    };

    Config m_config;
    std::shared_ptr<Executor> m_executor;
    template <KernelTypes KType, typename T>
    struct AttentionExecutor;
    friend struct ScaledDotProductAttentionKey;

    std::shared_ptr<VariableStateKVcache> m_k_state;
    std::shared_ptr<VariableStateKVcache> m_v_state;
    // KV cache layout
    // (0, 1, 2, 3) for BHLS
    // (2, 0, 1, 3) for LBHS
    std::vector<size_t> m_kvstate_layout = {2, 0, 1, 3};
    ov::Extensions::Cpu::CacheSpec m_key_spec;
    ov::Extensions::Cpu::CacheSpec m_value_spec;
    MemoryPtr m_per_thread_head_scratch;
    // Per-token TBQ norm. Populated only when a side has alg=TURBO; empty otherwise.
    PlainTensor m_k_quant_meta_data;
    PlainTensor m_v_quant_meta_data;
    // Random ±1 sign vector for WHT rotation.
    PlainTensor m_wht_signs;
};

}  // namespace ov::intel_cpu::node
