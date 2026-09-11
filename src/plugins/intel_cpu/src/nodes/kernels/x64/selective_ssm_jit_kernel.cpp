// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm_jit_kernel.hpp"

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "cpu/x64/jit_generator.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "nodes/kernels/x64/jit_kernel_base.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::kernel {

#define GET_OFF(field) offsetof(jit_selective_ssm_call_args, field)

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::load(const Vmm& destination,
                                         const Xbyak::Reg64& source,
                                         const ov::element::Type& source_precision,
                                         int element_count,
                                         size_t offset,
                                         bool zero_fill) {
    const bool fill_tail = zero_fill && static_cast<size_t>(element_count) < vector_size;
    const auto seed = load_emitter_params(source_precision, ov::element::f32, element_count, fill_tail, "zero").hash();
    auto& emitter = emitters[seed];
    if (!emitter) {
        constexpr cpu_isa_t emitter_isa = (isa & zmm_bit) != 0 ? avx512_core : isa;
        emitter = std::make_unique<jit_load_emitter>(this,
                                                     emitter_isa,
                                                     source_precision,
                                                     ov::element::f32,
                                                     element_count,
                                                     ov::element::f32,
                                                     fill_tail,
                                                     "zero");
    }
    emitter->emit_code({static_cast<size_t>(source.getIdx()), offset},
                       {static_cast<size_t>(destination.getIdx())},
                       pool_aux_vmm_idxs,
                       pool_aux_gpr_idxs);
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::store(const Xbyak::Reg64& destination,
                                          const Vmm& source,
                                          const ov::element::Type& destination_precision,
                                          int element_count,
                                          size_t offset) {
    const auto seed = store_emitter_params(ov::element::f32, destination_precision, element_count).hash();
    auto& emitter = emitters[seed];
    if (!emitter) {
        constexpr cpu_isa_t emitter_isa = (isa & zmm_bit) != 0 ? avx512_core : isa;
        emitter = std::make_unique<jit_store_emitter>(this,
                                                      emitter_isa,
                                                      ov::element::f32,
                                                      destination_precision,
                                                      element_count);
    }
    emitter->emit_code({static_cast<size_t>(source.getIdx())},
                       {static_cast<size_t>(destination.getIdx()), offset},
                       pool_aux_vmm_idxs,
                       pool_aux_gpr_idxs);
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::reduce_to_scalar(const Vmm& accumulator) {
    const Xbyak::Xmm accumulator_xmm(accumulator.getIdx());
    // AVX2 decode uses all 16 SIMD registers for a four-row tile. State vectors are dead once reduction starts, so
    // their low registers can safely host the VEX-encoded 128-bit reduction instructions.
    const auto tmp0_idx = isa == avx2 ? 0 : vmm_reduce_tmp0.getIdx();
    const auto tmp1_idx = isa == avx2 ? 1 : vmm_reduce_tmp1.getIdx();
    const Xbyak::Xmm tmp0_xmm(tmp0_idx);
    const Xbyak::Xmm tmp1_xmm(tmp1_idx);

    if constexpr (isa == avx2) {
        vextractf128(tmp0_xmm, Xbyak::Ymm(accumulator.getIdx()), 1);
        vaddps(accumulator_xmm, accumulator_xmm, tmp0_xmm);
    } else {
        vextractf32x8(Xbyak::Ymm(tmp0_idx), Xbyak::Zmm(accumulator.getIdx()), 1);
        vaddps(Xbyak::Ymm(accumulator.getIdx()), Xbyak::Ymm(accumulator.getIdx()), Xbyak::Ymm(tmp0_idx));
        vextractf128(tmp0_xmm, Xbyak::Ymm(accumulator.getIdx()), 1);
        vaddps(accumulator_xmm, accumulator_xmm, tmp0_xmm);
    }

    vpermilps(tmp1_xmm, accumulator_xmm, 0xB1);
    vaddps(accumulator_xmm, accumulator_xmm, tmp1_xmm);
    vpermilps(tmp1_xmm, accumulator_xmm, 0x4E);
    vaddps(accumulator_xmm, accumulator_xmm, tmp1_xmm);
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::clear_inactive_lanes(const Vmm& value, size_t active_lanes) {
    const auto active_mask = static_cast<uint16_t>((uint32_t{1} << active_lanes) - 1U);
    const auto vector_mask = static_cast<uint16_t>((uint32_t{1} << vector_size) - 1U);
    const auto inactive_mask = static_cast<uint16_t>(vector_mask & ~active_mask);
    uni_vpxor(vmm_reduce_tmp0, vmm_reduce_tmp0, vmm_reduce_tmp0);
    if constexpr (isa == avx2) {
        vblendps(value, value, vmm_reduce_tmp0, static_cast<uint8_t>(inactive_mask));
    } else {
        mov(r14.cvt32(), inactive_mask);
        kmovw(k1, r14.cvt32());
        vblendmps(value | k1, value, vmm_reduce_tmp0);
    }
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::store_output(const Vmm& source, int element_count, size_t offset) {
    store(reg_output, source, m_jcp.data_precision, element_count, offset);
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::prepare_row_scales() {
    const Xbyak::Xmm packed_scales(vmm_input_projection.getIdx());
    const Xbyak::Xmm delta(accumulator_vmm(0).getIdx());
    load(vmm_input_projection, reg_x, m_jcp.data_precision, max_row_tile, 0, false);
    vbroadcastss(delta, ptr[reg_steps + offsetof(jit_selective_ssm_step, delta)]);
    vmulps(packed_scales, packed_scales, delta);

    for (size_t row = 0; row < max_row_tile; ++row) {
        const Xbyak::Xmm scale(input_scale_vmm(row).getIdx());
        vpermilps(scale, packed_scales, static_cast<uint8_t>(row * 0x55U));
    }
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::store_row_tile() {
    const Xbyak::Xmm packed_output(state_vmm(0).getIdx());
    const Xbyak::Xmm packed_output_high(state_vmm(1).getIdx());
    vunpcklps(packed_output, Xbyak::Xmm(accumulator_vmm(0).getIdx()), Xbyak::Xmm(accumulator_vmm(1).getIdx()));
    vunpcklps(packed_output_high, Xbyak::Xmm(accumulator_vmm(2).getIdx()), Xbyak::Xmm(accumulator_vmm(3).getIdx()));
    vshufps(packed_output, packed_output, packed_output_high, 0x44);
    store_output(state_vmm(0), max_row_tile);
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::store_state(const Vmm& source, int element_count, size_t offset) {
    const auto& destination =
        m_jcp.state_mode == jit_selective_ssm_state_mode::separate ? reg_output_state : reg_input_state;
    store(destination, source, m_jcp.state_precision, element_count, offset);
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::emit_state_vector(size_t rows,
                                                      size_t active_lanes,
                                                      size_t projection_offset,
                                                      size_t state_vector_offset) {
    const auto state_element_size = m_jcp.state_precision.size();
    const auto state_row_bytes = m_jcp.state_size * state_element_size;
    const bool is_full_vector = active_lanes == vector_size;

    if (is_full_vector) {
        vmovups(vmm_input_projection, ptr[reg_input_projection + projection_offset]);
        vmovups(vmm_output_projection, ptr[reg_output_projection + projection_offset]);
    } else {
        load(vmm_input_projection,
             reg_input_projection,
             ov::element::f32,
             static_cast<int>(active_lanes),
             projection_offset);
        load(vmm_output_projection,
             reg_output_projection,
             ov::element::f32,
             static_cast<int>(active_lanes),
             projection_offset);
    }

    const auto emit_store = [&](size_t row) {
        const auto state = state_vmm(row);
        const auto state_offset = row * state_row_bytes + state_vector_offset;
        if (is_full_vector && m_jcp.state_precision == ov::element::f32) {
            const auto& destination =
                m_jcp.state_mode == jit_selective_ssm_state_mode::separate ? reg_output_state : reg_input_state;
            vmovups(ptr[destination + state_offset], state);
        } else {
            store_state(state, static_cast<int>(active_lanes), state_offset);
        }
    };

    for (size_t row = 0; row < rows; ++row) {
        const auto state = state_vmm(row);
        const auto state_offset = row * state_row_bytes + state_vector_offset;
        if (is_full_vector && m_jcp.state_precision == ov::element::f32) {
            vmovups(state, ptr[reg_input_state + state_offset]);
        } else {
            load(state, reg_input_state, m_jcp.state_precision, static_cast<int>(active_lanes), state_offset);
        }

        // state[p, n] = decay * state[p, n] + (delta * x[p]) * B[n]
        vmulps(state, state, vmm_decay);
        vfmadd231ps(state, vmm_input_projection, input_scale_vmm(row));
        if (!is_full_vector) {
            clear_inactive_lanes(state, active_lanes);
        }

        if (m_jcp.state_mode == jit_selective_ssm_state_mode::in_place) {
            emit_store(row);
        }
        // output[p] = sum_n(state[p, n] * C[n])
        const auto vector = projection_offset / (vector_size * sizeof(float));
        vfmadd231ps(accumulator_vmm(row, vector), state, vmm_output_projection);
        if constexpr (isa == avx2) {
            if (m_jcp.state_mode == jit_selective_ssm_state_mode::separate) {
                emit_store(row);
            }
        }
    }

    if constexpr (isa != avx2) {
        if (m_jcp.state_mode == jit_selective_ssm_state_mode::separate) {
            for (size_t row = 0; row < rows; ++row) {
                emit_store(row);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::emit_row_tile(size_t rows) {
    const auto data_size = m_jcp.data_precision.size();
    const auto state_element_size = m_jcp.state_precision.size();
    const auto full_vectors = m_jcp.state_size / vector_size;
    const auto tail = m_jcp.state_size % vector_size;

    const bool use_packed_rows = m_jcp.data_precision != ov::element::f32 && rows == max_row_tile;
    if (use_packed_rows) {
        prepare_row_scales();
    }
    for (size_t row = 0; row < rows; ++row) {
        const auto scale = input_scale_vmm(row);
        const auto accumulator = accumulator_vmm(row);
        if (!use_packed_rows) {
            load(scale, reg_x, m_jcp.data_precision, 1, row * data_size, false);
            vmulss(Xbyak::Xmm(scale.getIdx()),
                   Xbyak::Xmm(scale.getIdx()),
                   ptr[reg_steps + offsetof(jit_selective_ssm_step, delta)]);
        }
        vbroadcastss(scale, Xbyak::Xmm(scale.getIdx()));
        uni_vpxor(accumulator, accumulator, accumulator);
        if constexpr ((isa & zmm_bit) != 0) {
            const auto second = accumulator_vmm(row, 1);
            uni_vpxor(second, second, second);
        }
    }

    const auto loop_chunks = full_vectors > max_unrolled_vectors ? full_vectors / max_unrolled_vectors : 0;
    const auto loop_vectors = loop_chunks * max_unrolled_vectors;
    if (loop_chunks > 0) {
        Xbyak::Label vector_loop;
        mov(reg_vector_chunks, loop_chunks);
        align(16);
        L(vector_loop);
        for (size_t vector = 0; vector < max_unrolled_vectors; ++vector) {
            emit_state_vector(rows,
                              vector_size,
                              vector * vector_size * sizeof(float),
                              vector * vector_size * state_element_size);
        }
        advance_state_pointers(max_unrolled_vectors * vector_size);
        dec(reg_vector_chunks);
        jnz(vector_loop, T_NEAR);
    }

    for (size_t vector = 0; vector < full_vectors - loop_vectors; ++vector) {
        const auto projection_offset = vector * vector_size * sizeof(float);
        const auto state_vector_offset = vector * vector_size * state_element_size;
        emit_state_vector(rows, vector_size, projection_offset, state_vector_offset);
    }

    if (tail > 0) {
        const auto projection_offset = (full_vectors - loop_vectors) * vector_size * sizeof(float);
        const auto state_vector_offset = (full_vectors - loop_vectors) * vector_size * state_element_size;
        emit_state_vector(rows, tail, projection_offset, state_vector_offset);
    }

    if (loop_vectors > 0) {
        advance_state_pointers(-static_cast<int64_t>(loop_vectors * vector_size));
    }

    for (size_t row = 0; row < rows; ++row) {
        const auto accumulator = accumulator_vmm(row);
        if constexpr ((isa & zmm_bit) != 0) {
            vaddps(accumulator, accumulator, accumulator_vmm(row, 1));
        }
        reduce_to_scalar(accumulator);
        if (!use_packed_rows) {
            store_output(accumulator, 1, row * data_size);
        }
    }
    if (use_packed_rows) {
        store_row_tile();
    }
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::advance_state_pointers(int64_t elements) {
    // Xbyak takes the bits of a sign-extended imm32; the state-size limit keeps both offsets representable.
    const auto state_bytes = static_cast<uint32_t>(elements * static_cast<int64_t>(m_jcp.state_precision.size()));
    const auto projection_bytes = static_cast<uint32_t>(elements * static_cast<int64_t>(sizeof(float)));
    add(reg_input_state, state_bytes);
    if (m_jcp.state_mode == jit_selective_ssm_state_mode::separate) {
        add(reg_output_state, state_bytes);
    }
    add(reg_input_projection, projection_bytes);
    add(reg_output_projection, projection_bytes);
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::advance_row_pointers(size_t rows) {
    add(reg_input_state, rows * m_jcp.state_size * m_jcp.state_precision.size());
    if (m_jcp.state_mode == jit_selective_ssm_state_mode::separate) {
        add(reg_output_state, rows * m_jcp.state_size * m_jcp.state_precision.size());
    }
    add(reg_x, rows * m_jcp.data_precision.size());
    add(reg_output, rows * m_jcp.data_precision.size());
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::generate() {
    this->preamble();

    mov(reg_args, abi_param1);
    mov(reg_input_state, ptr[reg_args + GET_OFF(input_state)]);
    if (m_jcp.state_mode == jit_selective_ssm_state_mode::separate) {
        mov(reg_output_state, ptr[reg_args + GET_OFF(output_state)]);
    }
    mov(reg_input_projection, ptr[reg_args + GET_OFF(input_projection)]);
    mov(reg_output_projection, ptr[reg_args + GET_OFF(output_projection)]);
    mov(reg_x, ptr[reg_args + GET_OFF(x)]);
    mov(reg_output, ptr[reg_args + GET_OFF(output)]);
    mov(reg_steps, ptr[reg_args + GET_OFF(steps)]);
    Xbyak::Label token_loop;
    Xbyak::Label kernel_exit;
    if (m_jcp.state_mode == jit_selective_ssm_state_mode::in_place) {
        mov(reg_tokens, ptr[reg_args + GET_OFF(token_count)]);
        test(reg_tokens, reg_tokens);
        jz(kernel_exit, T_NEAR);
    }
    L(token_loop);
    mov(reg_rows, ptr[reg_args + GET_OFF(row_count)]);
    vbroadcastss(vmm_decay, ptr[reg_steps + offsetof(jit_selective_ssm_step, decay)]);

    Xbyak::Label main_loop;
    Xbyak::Label tail_loop;
    Xbyak::Label end;

    constexpr size_t active_row_tile = max_row_tile;
    cmp(reg_rows, active_row_tile);
    jb(tail_loop, T_NEAR);
    align(16);
    L(main_loop);
    emit_row_tile(active_row_tile);
    advance_row_pointers(active_row_tile);
    sub(reg_rows, active_row_tile);
    cmp(reg_rows, active_row_tile);
    jae(main_loop, T_NEAR);

    L(tail_loop);
    test(reg_rows, reg_rows);
    jz(end, T_NEAR);
    emit_row_tile(1);
    advance_row_pointers(1);
    dec(reg_rows);
    jnz(tail_loop, T_NEAR);

    L(end);
    if (m_jcp.state_mode == jit_selective_ssm_state_mode::in_place) {
        dec(reg_tokens);
        jz(kernel_exit, T_NEAR);
        // Row traversal advances x/output; restore their token bases before applying the token stride.
        mov(rax, ptr[reg_args + GET_OFF(row_count)]);
        imul(rax, rax, static_cast<int>(m_jcp.data_precision.size()));
        sub(reg_x, rax);
        sub(reg_output, rax);
        add(reg_x, ptr[reg_args + GET_OFF(input_stride)]);
        add(reg_output, ptr[reg_args + GET_OFF(input_stride)]);
        add(reg_input_projection, ptr[reg_args + GET_OFF(projection_stride)]);
        add(reg_output_projection, ptr[reg_args + GET_OFF(projection_stride)]);
        mov(reg_input_state, ptr[reg_args + GET_OFF(input_state)]);
        add(reg_steps, sizeof(jit_selective_ssm_step));
        jmp(token_loop, T_NEAR);
    }
    L(kernel_exit);
    this->postamble();
    for (const auto& emitter : emitters) {
        if (emitter.second) {
            emitter.second->emit_data();
        }
    }
}

bool is_selective_ssm_jit_precision_supported(const ov::element::Type& precision) {
    if (precision == ov::element::f32) {
        return mayiuse(avx2);
    }
    if (precision == ov::element::f16) {
        return mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2);
    }
    if (precision == ov::element::bf16) {
        return mayiuse(avx512_core_bf16) || mayiuse(avx2_vnni_2);
    }
    return false;
}

std::shared_ptr<JitKernelBase> create_selective_ssm_jit_kernel(const ov::element::Type& data_precision,
                                                               size_t state_size,
                                                               const ov::element::Type& state_precision,
                                                               jit_selective_ssm_state_mode state_mode) {
    if (!is_selective_ssm_jit_precision_supported(data_precision)) {
        return nullptr;
    }
    if (state_size == 0 || state_size > max_selective_ssm_jit_state_size) {
        return nullptr;
    }
    if (state_precision != ov::element::f32 && state_precision != data_precision) {
        return nullptr;
    }
    if (state_mode != jit_selective_ssm_state_mode::in_place && state_mode != jit_selective_ssm_state_mode::separate &&
        state_mode != jit_selective_ssm_state_mode::no_store) {
        return nullptr;
    }
    if (state_precision != ov::element::f32 && state_mode == jit_selective_ssm_state_mode::in_place) {
        return nullptr;
    }

    const jit_selective_ssm_compile_params compile_params{
        data_precision,
        state_precision,
        state_size,
        state_mode,
    };
    // Vector width selects the recurrence kernel; the emitters select native FP16/BF16 conversions.
    if (mayiuse(avx512_core)) {
        auto result = std::make_shared<jit_selective_ssm_kernel<avx512_core>>(compile_params);
        result->create_kernel();
        return result;
    }
    if (mayiuse(avx2)) {
        auto result = std::make_shared<jit_selective_ssm_kernel<avx2>>(compile_params);
        result->create_kernel();
        return result;
    }
    return nullptr;
}

template class jit_selective_ssm_kernel<avx2>;
template class jit_selective_ssm_kernel<avx512_core>;

}  // namespace ov::intel_cpu::kernel
