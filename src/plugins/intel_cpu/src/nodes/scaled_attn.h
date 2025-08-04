// Copyright (C) 2018-2025 Intel Corporation
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
#include "memory_state.h"
#include "node.h"
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
    struct SDPAQuantParam {
        ov::element::Type precision = ov::element::dynamic;
        size_t groupSize = 0;
        bool isByChannel = false;
    };
    ov::element::Type getKVCachePrecision();
    const SDPAQuantParam& getKeyQuantParam();
    const SDPAQuantParam& getValueQuantParam();

private:
    void gatherConcatPastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v, const MemoryPtr& mem_beam_idx);
    void updateBeamTable(const MemoryPtr& mem_beam_idx, size_t L1);
    void updatePastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v);
    ov::element::Type getRuntimePrecision() const override;
    void resetBeamTablePastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v, const MemoryPtr& mem_beam_idx);

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
                             const PlainTensor& v_scale_zp) = 0;
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
    SDPAQuantParam m_key_quant_param;
    SDPAQuantParam m_value_quant_param;
};

}  // namespace ov::intel_cpu::node
