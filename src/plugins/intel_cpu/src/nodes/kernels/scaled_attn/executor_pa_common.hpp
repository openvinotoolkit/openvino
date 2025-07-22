// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <xbyak/xbyak.h>

#include <cstddef>
#include <cstdint>
#include <openvino/core/type/element_type.hpp>
#include <utility>
#include <vector>

#include "cpu/x64/jit_generator.hpp"
#include "cpu_memory.h"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu {

// this file will contain features that do not require multiple instantiation

struct PagedAttentionExecutor {
    // PagedAttention input index
    static const size_t ID_Q = 0;                          // [B_token, H * S], float
    static const size_t ID_K = 1;                          // [B_token, Hk * S], float
    static const size_t ID_V = 2;                          // [B_token, Hk * S], float
    static const size_t ID_KCACHE = 3;                     // [block_number, H, block_size, S], float
    static const size_t ID_VCACHE = 4;                     // [block_number, H, block_size, S], float
    static const size_t ID_PAST_LENS = 5;                  // [B_seq]
    static const size_t ID_SUBSEQUENCE_BEGINS = 6;         // [B_seq+1]
    static const size_t ID_BLOCK_INDICES = 7;              // [num_blocks]
    static const size_t ID_BLOCK_INDICES_BEGINS = 8;       // [B_seq+1]
    static const size_t ID_SCALE = 9;                      // [], float
    static const size_t ID_SLIDING_WINDOW = 10;            // []
    static const size_t ID_ALIBI_SLOPES = 11;              // [H|0], float
    static const size_t ID_MAX_CONTEXT_LEN = 12;           // []
    static const size_t ID_SCORE_AGGREGATION_WINDOW = 13;  // [B_seq || 0], i32
    static const size_t ID_ROTATED_BLOCK_INDICES = 14;     // [num_rotated_blocks || 0], int32
    static const size_t ID_ROTATION_DELTAS = 15;           // [num_rotated_blocks * block_size || 0], int32
    static const size_t ID_ROTATION_TRIG_LUT = 16;         // [max_context_length * S || 0], f32
    virtual void execute(const std::vector<ov::intel_cpu::MemoryPtr>& inputs,
                         std::vector<ov::intel_cpu::MemoryPtr> outputs) = 0;
    virtual ~PagedAttentionExecutor() = default;
};

struct PagedAttnQuantParams {
    size_t key_group_size = 0UL;
    size_t value_group_size = 0UL;
    bool quant_key_bychannel = false;
    bool quant_value_bychannel = false;
    bool is_sage_attn = false;
};

struct AttnWorkItem {
    int32_t batch_in_reorder;  // which batch in reorder buffer will be used
    int32_t batch_in_seq;      // batch idx in sequence
    int32_t q_len;             // current sequence length, 1 for second token, 2+ for first token
    int32_t q_block_id;        // block id in this seq, valid at first token
};
struct ReorderWorkItem {
    int32_t batch_in_seq;      // batch idx in sequence
    int32_t batch_in_reorder;  // which batch in reorder buffer will be used
    int32_t kv_block_id;       // block id in this kv cache seq
    int32_t block_number;      // block_number in global cache
    int32_t valid_block_len;
};
struct WorkItems {
private:
    std::vector<AttnWorkItem> attn_items;
    std::vector<ReorderWorkItem> reorder_items;
    int32_t max_kv_len_in_reorder = 0;  // max kv len between first tokens
    int32_t max_batch_in_reorder = 0;
    int32_t total_kv_len = 0;

public:
    void reset([[maybe_unused]] const ov::intel_cpu::PlainTensor& query,
               const ov::intel_cpu::PlainTensor& past_lens,
               const ov::intel_cpu::PlainTensor& subsequence_begins,
               const ov::intel_cpu::PlainTensor& block_indices,
               const ov::intel_cpu::PlainTensor& block_indices_begins,
               size_t block_size) {
        attn_items.clear();
        reorder_items.clear();
        max_kv_len_in_reorder = 0;
        max_batch_in_reorder = 0;
        total_kv_len = 0;
        auto seq_cout = static_cast<int32_t>(past_lens.m_dims[0]);
        for (int32_t i = 0; i < seq_cout; i++) {
            auto q_len = subsequence_begins.ptr<int32_t>()[i + 1] - subsequence_begins.ptr<int32_t>()[i];
            auto kv_len = past_lens.ptr<int32_t>()[i] + q_len;
            auto kv_len_in_block = static_cast<int32_t>(ov::intel_cpu::div_up(kv_len, block_size));
            if (q_len == 1) {
                attn_items.emplace_back(AttnWorkItem{0,     // batch_in_reorder
                                                     i,     // batch_in_seq
                                                     1ull,  // q_len
                                                     // kv_len in blocks, used in the sort function
                                                     kv_len_in_block - 1});
            } else {
                auto reorder_sub_work_count = kv_len_in_block;
                max_kv_len_in_reorder = std::max(max_kv_len_in_reorder, kv_len);
                for (int32_t block_id = 0; block_id < reorder_sub_work_count; block_id++) {
                    int32_t valid_block_size =
                        block_id == (reorder_sub_work_count - 1) ? kv_len - block_id * block_size : block_size;
                    auto block_number = block_indices.ptr<int32_t>()[block_indices_begins.ptr<int32_t>()[i] + block_id];
                    reorder_items.emplace_back(ReorderWorkItem{i,                     // batch_in_seq
                                                               max_batch_in_reorder,  // batch_in_reorder
                                                               block_id,              // kv_block_id
                                                               block_number,          // block_number
                                                               valid_block_size});    // valid_block_len
                }

                // workitems for attention
                auto attn_sub_work_count = static_cast<int32_t>(ov::intel_cpu::div_up(q_len, block_size));
                for (int32_t block_id = 0; block_id < attn_sub_work_count; block_id++) {
                    attn_items.emplace_back(AttnWorkItem{
                        max_batch_in_reorder,  // batch_in_reorder
                        i,                     // batch_in_seq
                        q_len,                 // q_len
                        block_id               // q_block_id
                    });
                }
                max_batch_in_reorder++;
            }
            total_kv_len += kv_len;
        }
    }
    [[nodiscard]] const AttnWorkItem& get_attn_work_item(size_t idx) const {
        return attn_items[idx];
    }
    [[nodiscard]] size_t attn_work_size() const {
        return attn_items.size();
    }
    [[nodiscard]] const ReorderWorkItem& get_reorder_work_item(size_t idx) const {
        return reorder_items[idx];
    }
    [[nodiscard]] size_t reorder_work_size() const {
        return reorder_items.size();
    }
    [[nodiscard]] size_t get_reorder_max_batch_size() const {
        return static_cast<size_t>(max_batch_in_reorder);
    }
    [[nodiscard]] size_t get_reorder_max_kv_len() const {
        return static_cast<size_t>(max_kv_len_in_reorder);
    }
    [[nodiscard]] size_t get_total_kv_len() const {
        return static_cast<size_t>(total_kv_len);
    }
};

#ifdef OPENVINO_ARCH_X86_64

// w = query * Key
//
// query: [1,      S]
// Key  : [block_size, S]
// w    : [1, block_size]
//
// S is known at compile time
struct TileConfig {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
    void reset(int palette, int _startRow, const std::vector<std::pair<int, int>>& _rows_columnsBytes);
};

class TileConfiger : public dnnl::impl::cpu::x64::jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(TileConfiger)
    TileConfiger();
    void generate() override;
};

class JitMatMulVecAMX : public dnnl::impl::cpu::x64::jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(JitMatMulVecAMX)
    void operator=(const JitMatMulVecAMX&) = delete;
    int m_head_size;
    int m_block_size;
    ov::element::Type m_amx_prec;
    TileConfiger m_tile_configer;
    TileConfig m_tile_cfg{};
    JitMatMulVecAMX(int head_size, int block_size, ov::element::Type amx_prec);

    void tile_config() {
        m_tile_configer(&m_tile_cfg);
    }
    void tile_release() {
        m_tile_configer(nullptr);
    }

    // to save push/pop: do not use `abi_save_gpr_regs`
    static constexpr auto abi_param_regs = dnnl::impl::cpu::x64::abi_param_regs;
    Xbyak::Reg64 reg_q_addr = abi_param1;
    Xbyak::Reg64 reg_k_addr = abi_param2;
    Xbyak::Reg64 reg_dst_addr = abi_param3;
    Xbyak::Reg64 reg_stride_A = rax;
    Xbyak::Reg64 reg_stride_BC = r9;

    Xbyak::Tmm tmmC = tmm0;
    Xbyak::Tmm tmmA = tmm1;
    Xbyak::Tmm tmmB0 = tmm2;
    Xbyak::Tmm tmmB1 = tmm3;
    Xbyak::Tmm tmmB2 = tmm4;
    Xbyak::Tmm tmmB3 = tmm5;
    Xbyak::Tmm tmmB4 = tmm6;
    Xbyak::Tmm tmmB5 = tmm7;

    void generate() override;
};

#endif

}  // namespace ov::Extensions::Cpu
