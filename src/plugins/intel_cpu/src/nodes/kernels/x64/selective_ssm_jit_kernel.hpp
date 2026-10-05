// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "jit_kernel_base.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::kernel {

// Supported state-size limit. Large states use a runtime loop over bounded groups of vectors.
constexpr size_t max_selective_ssm_jit_state_size = 4096;

enum class jit_selective_ssm_state_mode : std::uint8_t { in_place, separate, no_store };

struct jit_selective_ssm_compile_params {
    ov::element::Type data_precision = ov::element::dynamic;
    ov::element::Type state_precision = ov::element::dynamic;
    size_t state_size = 0;
    jit_selective_ssm_state_mode state_mode = jit_selective_ssm_state_mode::in_place;
};

struct jit_selective_ssm_step {
    float decay = 0.F;
    float delta = 0.F;
};

struct jit_selective_ssm_call_args {
    const void* input_state = nullptr;
    const float* input_projection = nullptr;
    const float* output_projection = nullptr;
    const void* x = nullptr;
    void* output = nullptr;
    const jit_selective_ssm_step* steps = nullptr;
    size_t row_count = 0;
    void* output_state = nullptr;
    size_t token_count = 1;
    size_t input_stride = 0;       // Bytes between consecutive tokens in x/output.
    size_t projection_stride = 0;  // Bytes between consecutive FP32 B/C projections.
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
class jit_selective_ssm_kernel : public JitKernel<jit_selective_ssm_compile_params, jit_selective_ssm_call_args> {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_selective_ssm_kernel)

    explicit jit_selective_ssm_kernel(const jit_selective_ssm_compile_params& jcp) : JitKernel(jit_name(), jcp, isa) {}

private:
    using Vmm = std::conditional_t<(isa & dnnl::impl::cpu::x64::zmm_bit) != 0, Xbyak::Zmm, Xbyak::Ymm>;

    static constexpr size_t vector_size = dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::vlen / sizeof(float);
    static constexpr size_t max_row_tile = 4;
    // Keep the common 128-element state fully unrolled on AVX2 and AVX-512.
    static constexpr size_t max_unrolled_vectors = 16;
    // An even chunk length preserves accumulator-bank parity when the runtime vector loop restarts at index 0.
    static_assert(max_unrolled_vectors % 2 == 0);

    // Preparation scratch aliases the first accumulator, before any accumulation bank is initialized.
    static constexpr size_t delta_scratch_vmm_idx = max_row_tile;
    // Output packing reuses state registers only after every state store and scalar reduction is complete.
    static constexpr size_t packed_output_low_vmm_idx = 0;
    static constexpr size_t packed_output_high_vmm_idx = 1;

    // Conversion emitters own this contiguous range while load/store helpers are active.
    static constexpr size_t aux_vmm_base = isa == dnnl::impl::cpu::x64::avx2 ? 3 * max_row_tile + 3 : 20;
    static constexpr size_t aux_vmm_count = isa == dnnl::impl::cpu::x64::avx2 ? 1 : 6;
    static constexpr size_t second_accumulator_vmm_base = aux_vmm_base + aux_vmm_count;
    // AVX2 reduction reuses state registers after all state stores; recurrence uses all 16 YMM registers.
    static constexpr size_t reduction_tmp0_vmm_idx = isa == dnnl::impl::cpu::x64::avx2 ? 0 : 3 * max_row_tile + 3;
    static constexpr size_t reduction_tmp1_vmm_idx = isa == dnnl::impl::cpu::x64::avx2 ? 1 : 3 * max_row_tile + 4;

    void generate() override;
    void emit_row_tile(size_t rows);
    void emit_state_vector(size_t rows,
                           size_t active_lanes,
                           size_t vector_index,
                           size_t projection_offset,
                           size_t state_vector_offset);
    void load_projections(size_t active_lanes, size_t projection_offset);
    void load_state(const Vmm& destination, size_t active_lanes, size_t offset);
    void compute_state(size_t row, size_t active_lanes);
    void accumulate_output(size_t row, size_t active_lanes, size_t vector);
    void store_state(const Vmm& source, size_t active_lanes, size_t offset);
    void advance_row_pointers(size_t rows);
    void advance_state_pointers(int64_t elements);
    // Initialize both banks for one row on AVX-512, or the single bank on AVX2.
    void zero_accumulators(size_t row);
    // Fold the odd bank into the even bank for one row; AVX2 needs no fold.
    void fold_partial_accumulators(size_t row);
    void reduce_to_scalar(const Vmm& accumulator);
    void store_output(const Vmm& source, int element_count, size_t offset = 0);
    void prepare_row_scales();
    void store_row_tile();
    void load(const Vmm& destination,
              const Xbyak::Reg64& source,
              const ov::element::Type& source_precision,
              int element_count,
              size_t offset = 0,
              bool zero_fill = true);
    void store(const Xbyak::Reg64& destination,
               const Vmm& source,
               const ov::element::Type& destination_precision,
               int element_count,
               size_t offset = 0);

    static Vmm state_vmm(size_t row) {
        return Vmm(row);
    }
    // Output dot products alternate banks for even/odd state vectors on AVX-512 to hide FMA latency.
    // The first bank follows the state registers; the second follows the conversion-emitter auxiliaries.
    // AVX2 has only 16 registers and keeps a single bank to leave room for state, scales and projections.
    static Vmm accumulator_vmm(size_t row, size_t vector = 0) {
        if constexpr ((isa & dnnl::impl::cpu::x64::zmm_bit) != 0) {
            if (vector % 2 != 0) {
                return Vmm(second_accumulator_vmm_base + row);
            }
        }
        return Vmm(max_row_tile + row);
    }
    static Vmm input_scale_vmm(size_t row) {
        return Vmm(2 * max_row_tile + row);
    }

    const Xbyak::Reg64 reg_args = rbx;
    const Xbyak::Reg64 reg_input_state = r8;
    const Xbyak::Reg64 reg_output_state = rbp;
    const Xbyak::Reg64 reg_input_projection = r9;
    const Xbyak::Reg64 reg_output_projection = r10;
    const Xbyak::Reg64 reg_x = r11;
    const Xbyak::Reg64 reg_output = r12;
    const Xbyak::Reg64 reg_rows = r13;
    const Xbyak::Reg64 reg_vector_chunks = rdx;
    const Xbyak::Reg64 reg_steps = rsi;
    const Xbyak::Reg64 reg_tokens = rdi;

    const Vmm vmm_decay = Vmm(3 * max_row_tile);
    const Vmm vmm_input_projection = Vmm(3 * max_row_tile + 1);
    const Vmm vmm_output_projection = Vmm(3 * max_row_tile + 2);
    const Vmm vmm_reduce_tmp0 = Vmm(reduction_tmp0_vmm_idx);
    const Vmm vmm_reduce_tmp1 = Vmm(reduction_tmp1_vmm_idx);
    // Shared load/store emitters use k1; keep the recurrence tail mask in a separate register.
    const Xbyak::Opmask k_tail = k2;

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
    const std::vector<size_t> pool_aux_gpr_idxs = {static_cast<size_t>(rax.getIdx()),
                                                   static_cast<size_t>(r14.getIdx()),
                                                   static_cast<size_t>(r15.getIdx())};
    const std::vector<size_t> pool_aux_vmm_idxs = [] {
        std::vector<size_t> indices(aux_vmm_count);
        for (size_t index = 0; index < aux_vmm_count; ++index) {
            indices[index] = aux_vmm_base + index;
        }
        return indices;
    }();
};

bool is_selective_ssm_jit_precision_supported(const ov::element::Type& precision);

std::shared_ptr<JitKernelBase> create_selective_ssm_jit_kernel(
    const ov::element::Type& data_precision,
    size_t state_size,
    const ov::element::Type& state_precision = ov::element::f32,
    jit_selective_ssm_state_mode state_mode = jit_selective_ssm_state_mode::in_place);

}  // namespace ov::intel_cpu::kernel
