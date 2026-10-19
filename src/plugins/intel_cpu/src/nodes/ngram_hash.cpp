// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngram_hash.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/ngram_hash_function.hpp"
#include "openvino/reference/ngram_hash_function.hpp"
#include "shape_inference/custom/ngram_hash.hpp"
#include "utils/general_utils.h"

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/jit_generator.hpp"
#endif

namespace ov::intel_cpu::node {

#if defined(OPENVINO_ARCH_X86_64)
struct NgramHash::JitKernel : public dnnl::impl::cpu::x64::jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(ngram_hash_barrett)

    struct Args {
        const void* src;
        int32_t* dst;
        size_t count;
    };

    explicit JitKernel(const NgramHash& node) : jit_generator_t(jit_name()), m_node(node) {}

    void operator()(const Args& args) const {
        using Function = void (*)(const Args*);
        reinterpret_cast<Function>(const_cast<uint8_t*>(jit_ker()))(&args);
    }

private:
    void generate() override {
        using dnnl::impl::cpu::x64::abi_param_regs;
        const auto input_i64 = m_node.m_precision == ov::element::i64;
        const auto input_size = input_i64 ? sizeof(int64_t) : sizeof(int32_t);
        const auto order = m_node.m_ngram_size;
        const auto num_heads = m_node.m_head_vocab_sizes.size();
        Xbyak::Label loop;
        Xbyak::Label done;
        preamble();
        mov(r8, ptr[abi_param1 + offsetof(Args, src)]);
        mov(r9, ptr[abi_param1 + offsetof(Args, dst)]);
        mov(r10, ptr[abi_param1 + offsetof(Args, count)]);
        test(r10, r10);
        jz(done, T_NEAR);
        L(loop);
        const auto load_token = [&](const Xbyak::Reg64& target, size_t back) {
            const auto offset = (order - 1 - back) * input_size;
            if (input_i64) {
                mov(target, qword[r8 + offset]);
            } else {
                movsxd(target, dword[r8 + offset]);
            }
            mov(rcx, static_cast<uint64_t>(m_node.m_multipliers[back]));
            imul(target, rcx);
        };
        load_token(r11, 0);
        size_t head = 0;
        for (size_t back = 1; back < order; ++back) {
            load_token(rax, back);
            xor_(r11, rax);
            mov(r12, r11);
            sar(r12, 63);
            for (int64_t index = 0; index < m_node.m_heads_per_order[back - 1]; ++index, ++head) {
                const auto divisor = static_cast<uint64_t>(m_node.m_head_vocab_sizes[head]);
                const auto offset = static_cast<uint32_t>(m_node.m_head_offsets[head]);
                if (divisor == 1) {
                    mov(dword[r9 + head * sizeof(int32_t)], offset);
                    continue;
                }
                const auto max_value = std::numeric_limits<uint64_t>::max();
                const auto magic = max_value / divisor;
                const auto two64_mod = (max_value % divisor + 1) % divisor;
                mov(rax, r11);
                mov(rdx, magic);
                mul(rdx);
                mov(r13, divisor);
                imul(rdx, r13);
                mov(r14, r11);
                sub(r14, rdx);
                mov(rax, r14);
                sub(rax, r13);
                cmovae(r14, rax);
                if (two64_mod != 0) {
                    mov(rax, two64_mod);
                    and_(rax, r12);
                    sub(r14, rax);
                    mov(rax, r14);
                    add(rax, r13);
                    test(r14, r14);
                    cmovs(r14, rax);
                }
                add(r14, offset);
                mov(dword[r9 + head * sizeof(int32_t)], r14d);
            }
        }
        add(r8, order * input_size);
        add(r9, num_heads * sizeof(int32_t));
        dec(r10);
        jnz(loop, T_NEAR);
        L(done);
        postamble();
    }

    const NgramHash& m_node;
};
#endif

bool NgramHash::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::as_type_ptr<const ov::op::internal::NGramHashFunction>(op)) {
            errorMessage = "Only internal NGramHashFunction operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

NgramHash::NgramHash(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgramHashShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto ngram_hash = ov::as_type_ptr<const ov::op::internal::NGramHashFunction>(op);
    m_ngram_size = static_cast<size_t>(ngram_hash->get_ngram_size());
    m_heads_per_order = ngram_hash->get_heads_per_order();
    m_multipliers = ngram_hash->get_multipliers();
    m_head_vocab_sizes = ngram_hash->get_head_vocab_sizes();
    m_head_offsets = ngram_hash->get_head_offsets();
}

void NgramHash::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    m_precision = getOriginalInputPrecisionAtPort(0);
    if (none_of(m_precision, ov::element::i32, ov::element::i64)) {
        m_precision = ov::element::i32;
    }
    auto implementation = impl_desc_type::ref_any;
#if defined(OPENVINO_ARCH_X86_64)
    if (m_ngram_size <= 32 && m_head_vocab_sizes.size() <= 128) {
        m_jit_kernel = std::make_shared<JitKernel>(*this);
        OPENVINO_ASSERT(m_jit_kernel->create_kernel() == dnnl::impl::status::success,
                        "Failed to create NgramHash Barrett kernel");
        implementation = impl_desc_type::jit_uni;
    }
#endif
    addSupportedPrimDesc({{LayoutType::ncsp, m_precision}},
                         {{LayoutType::ncsp, ov::element::i32}},
                         implementation);
}

template <typename T>
void NgramHash::executeImpl() {
    const auto* ngrams = getSrcDataAtPortAs<const T>(0);
    auto* row_ids = getDstDataAtPortAs<int32_t>(0);
    const size_t num_tokens = ov::shape_size(getSrcMemoryAtPort(0)->getStaticDims()) / m_ngram_size;
    const size_t num_heads = m_head_vocab_sizes.size();

    auto hash_range = [&](size_t begin, size_t end) {
#if defined(OPENVINO_ARCH_X86_64)
        if (m_jit_kernel) {
            const JitKernel::Args args{ngrams + begin * m_ngram_size, row_ids + begin * num_heads, end - begin};
            (*m_jit_kernel)(args);
            return;
        }
#endif
        ov::reference::ngram_hash_function(ngrams,
                                          row_ids,
                                          begin,
                                          end,
                                          m_ngram_size,
                                          m_heads_per_order.data(),
                                          m_multipliers.data(),
                                          m_head_vocab_sizes.data(),
                                          m_head_offsets.data(),
                                          num_heads);
    };

    constexpr size_t serial_window_limit = 256;
    if (num_tokens <= serial_window_limit) {
        hash_range(0, num_tokens);
        return;
    }
    const auto& cpu_parallel = context->getCpuParallel();
    const auto num_partitions = std::min(num_tokens, static_cast<size_t>(cpu_parallel->get_num_threads()));
    cpu_parallel->parallel_for(num_partitions, [&](size_t partition_index) {
        size_t begin = 0;
        size_t end = 0;
        splitter(num_tokens, num_partitions, partition_index, begin, end);
        hash_range(begin, end);
    });
}

void NgramHash::execute([[maybe_unused]] const dnnl::stream& strm) {
    if (m_precision == ov::element::i64) {
        executeImpl<int64_t>();
    } else {
        executeImpl<int32_t>();
    }
}

void NgramHash::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool NgramHash::created() const {
    return getType() == Type::NgramHash;
}

}  // namespace ov::intel_cpu::node
