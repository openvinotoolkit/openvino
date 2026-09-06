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
#include "emitters/plugin/x64/jit_bf16_emitters.hpp"
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
                                          size_t offset,
                                          const ov::element::Type& source_precision) {
    const auto seed = store_emitter_params(source_precision, destination_precision, element_count).hash();
    auto& emitter = emitters[seed];
    if (!emitter) {
        constexpr cpu_isa_t emitter_isa = (isa & zmm_bit) != 0 ? avx512_core : isa;
        emitter = std::make_unique<jit_store_emitter>(this,
                                                      emitter_isa,
                                                      source_precision,
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
    if (m_jcp.data_precision == ov::element::bf16) {
        if constexpr (isa == avx2) {
            if (!mayiuse(avx2_vnni_2)) {
                // State registers are dead here. Pass them to the conversion emitter to avoid spills.
                if (!bf16_output_converter) {
                    bf16_output_converter = std::make_unique<jit_uni_vcvtneps2bf16>(this, avx2);
                }
                const auto source_idx = static_cast<size_t>(source.getIdx());
                bf16_output_converter->emit_code({source_idx},
                                                 {source_idx},
                                                 {static_cast<size_t>(state_vmm(2).getIdx())},
                                                 pool_aux_gpr_idxs);
                store(reg_output, source, ov::element::bf16, element_count, offset, ov::element::bf16);
                return;
            }
        }
        store_bf16(reg_output, source, element_count, offset);
    } else {
        store(reg_output, source, m_jcp.data_precision, element_count, offset);
    }
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::prepare_row_scales() {
    const Xbyak::Xmm packed_scales(vmm_input_projection.getIdx());
    const Xbyak::Xmm delta(accumulator_vmm(0).getIdx());
    load(vmm_input_projection, reg_x, m_jcp.data_precision, max_row_tile, 0, false);
    vbroadcastss(delta, ptr[reg_args + GET_OFF(delta)]);
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
void jit_selective_ssm_kernel<isa>::emit_bf16_subnormal_store(const Xbyak::Reg64& destination,
                                                              const Vmm& source,
                                                              int element_count,
                                                              size_t offset) {
    if constexpr (isa == avx2) {
        store_avx2_bf16(destination, source, element_count, offset);
        return;
    }
    uni_vmovups(vmm_reduce_tmp0, source);
    uni_vmovups(vmm_reduce_tmp1, source);
    uni_vpsrld(vmm_reduce_tmp1, vmm_reduce_tmp1, 16);
    vpsllw(vmm_reduce_tmp1, vmm_reduce_tmp1, 15);
    uni_vpaddd(vmm_reduce_tmp0, vmm_reduce_tmp0, vmm_reduce_tmp1);
    uni_vpsrld(vmm_reduce_tmp0, vmm_reduce_tmp0, 16);

    const Xbyak::Zmm packed(vmm_reduce_tmp0.getIdx());
    if (static_cast<size_t>(element_count) == vector_size) {
        vpmovdw(ptr[destination + offset], packed);
    } else {
        const auto active_mask = static_cast<uint16_t>((uint32_t{1} << element_count) - 1U);
        mov(r14.cvt32(), active_mask);
        kmovw(k1, r14.cvt32());
        vpmovdw(ptr[destination + offset], packed | k1);
    }
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::store_avx2_bf16(const Xbyak::Reg64& destination,
                                                    const Vmm& source,
                                                    int element_count,
                                                    size_t offset) {
    if constexpr (isa == avx2) {
        const Xbyak::Ymm rounded(vmm_reduce_tmp0.getIdx());
        const Xbyak::Ymm rounding_bit(source.getIdx());
        uni_vmovups(rounded, source);
        // Compute ((bits >> 16) & 1) << 15. The word shift discards every bit except the retained BF16 LSB.
        uni_vpsrld(rounding_bit, rounding_bit, 16);
        vpsllw(rounding_bit, rounding_bit, 15);
        uni_vpaddd(rounded, rounded, rounding_bit);
        uni_vpsrld(rounded, rounded, 16);

        const Xbyak::Xmm rounded_low(rounded.getIdx());
        const Xbyak::Xmm rounded_high(rounding_bit.getIdx());
        vextracti128(rounded_high, rounded, 1);
        vpackusdw(rounded_low, rounded_low, rounded_high);
        if (static_cast<size_t>(element_count) == vector_size) {
            vmovups(ptr[destination + offset], rounded_low);
        } else {
            for (int lane = 0; lane < element_count; ++lane) {
                vpextrw(word[destination + offset + lane * sizeof(uint16_t)], rounded_low, lane);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::store_state(const Vmm& source, int element_count, size_t offset) {
    if (m_jcp.state_precision == ov::element::f32) {
        const auto& destination =
            m_jcp.state_mode == jit_selective_ssm_state_mode::separate ? reg_output_state : reg_input_state;
        store(destination, source, m_jcp.state_precision, element_count, offset);
        return;
    }
    if (m_jcp.state_precision != ov::element::bf16) {
        store(reg_output_state, source, m_jcp.state_precision, element_count, offset);
        return;
    }

    if constexpr (isa == avx2) {
        if (m_jcp.state_mode == jit_selective_ssm_state_mode::separate &&
            static_cast<size_t>(element_count) == vector_size) {
            // The caller has already accumulated the output, so the state register is dead and can hold the rounding
            // bit. This avoids the spills and constant-table setup required by the generic BF16 store emitter.
            store_avx2_bf16(reg_output_state, source, element_count, offset);
        } else {
            store_bf16(reg_output_state, source, element_count, offset);
        }
        return;
    }

    store_bf16(reg_output_state, source, element_count, offset);
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::store_bf16(const Xbyak::Reg64& destination,
                                               const Vmm& source,
                                               int element_count,
                                               size_t offset) {
    if constexpr (isa == avx2) {
        if (!mayiuse(avx2_vnni_2)) {
            store(destination, source, ov::element::bf16, element_count, offset);
            return;
        }
    }

    const auto emit_regular_store = [&]() {
        // OpenVINO bfloat16 conversion depends only on the retained LSB and the highest discarded bit. Clearing the
        // lower 15 bits before the native round-to-nearest-even conversion preserves that behavior.
        if constexpr (isa == avx2) {
            vpsrld(vmm_reduce_tmp0, source, 15);
            vpslld(vmm_reduce_tmp0, vmm_reduce_tmp0, 15);
        } else {
            vpandd(vmm_reduce_tmp0, source, vmm_bf16_round_mask);
        }
        store(destination, vmm_reduce_tmp0, ov::element::bf16, element_count, offset);
    };

    if (mayiuse(avx512_core_bf16) || mayiuse(avx2_vnni_2)) {
        // Keep the uncommon conversion block out of the generated hot path.
        auto& fallback = deferred_bf16_subnormal_stores.emplace_back(destination, source, element_count, offset);
        if constexpr (isa == avx2) {
            // A state register is free after its output contribution has been accumulated.
            const auto zero = source.getIdx() == 0 ? state_vmm(1) : state_vmm(0);
            uni_vpxor(zero, zero, zero);
            vpsrld(vmm_reduce_tmp0, source, 23);
            vpslld(vmm_reduce_tmp0, vmm_reduce_tmp0, 24);
            vpcmpeqd(vmm_reduce_tmp0, vmm_reduce_tmp0, zero);
            vpmovmskb(r14.cvt32(), vmm_reduce_tmp0);
            const auto active_bytes = static_cast<uint32_t>((uint64_t{1} << (4 * element_count)) - 1U);
            test(r14.cvt32(), active_bytes);
        } else {
            constexpr uint8_t fpclass_subnormal = 1U << 5;
            vfpclassps(k1, source, fpclass_subnormal);
            kortestw(k1, k1);
        }
        jnz(fallback.entry, T_NEAR);
        emit_regular_store();
        L(fallback.continuation);
    } else {
        emit_regular_store();
    }
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
        vfmadd231ps(accumulator_vmm(row), state, vmm_output_projection);
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
            vmulss(Xbyak::Xmm(scale.getIdx()), Xbyak::Xmm(scale.getIdx()), ptr[reg_args + GET_OFF(delta)]);
        }
        vbroadcastss(scale, Xbyak::Xmm(scale.getIdx()));
        uni_vpxor(accumulator, accumulator, accumulator);
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
    const auto state_bytes = elements * static_cast<int64_t>(m_jcp.state_precision.size());
    const auto projection_bytes = elements * static_cast<int64_t>(sizeof(float));
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
    mov(reg_rows, ptr[reg_args + GET_OFF(row_count)]);
    vbroadcastss(vmm_decay, ptr[reg_args + GET_OFF(decay)]);
    if constexpr ((isa & zmm_bit) != 0) {
        if (m_jcp.state_precision == ov::element::bf16 || m_jcp.data_precision == ov::element::bf16) {
            mov(r14.cvt32(), 0xFFFF8000);
            vmovd(Xbyak::Xmm(vmm_bf16_round_mask.getIdx()), r14.cvt32());
            vpbroadcastd(vmm_bf16_round_mask, Xbyak::Xmm(vmm_bf16_round_mask.getIdx()));
        }
    }

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

    Xbyak::Label kernel_exit;
    L(end);
    if (!deferred_bf16_subnormal_stores.empty()) {
        jmp(kernel_exit, T_NEAR);
        for (auto& fallback : deferred_bf16_subnormal_stores) {
            L(fallback.entry);
            emit_bf16_subnormal_store(fallback.destination, fallback.source, fallback.element_count, fallback.offset);
            jmp(fallback.continuation, T_NEAR);
        }
    }
    L(kernel_exit);
    this->postamble();
    if (bf16_output_converter) {
        bf16_output_converter->emit_data();
    }
    for (const auto& emitter : emitters) {
        if (emitter.second) {
            emitter.second->emit_data();
        }
    }
}

std::shared_ptr<JitKernelBase> create_selective_ssm_jit_kernel(const ov::element::Type& data_precision,
                                                               size_t state_size,
                                                               const ov::element::Type& state_precision,
                                                               jit_selective_ssm_state_mode state_mode) {
    if (data_precision != ov::element::f32 && data_precision != ov::element::f16 &&
        data_precision != ov::element::bf16) {
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
