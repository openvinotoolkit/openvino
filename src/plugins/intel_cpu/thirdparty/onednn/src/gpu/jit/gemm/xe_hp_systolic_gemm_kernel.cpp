/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <assert.h>

#include "gpu/jit/gemm/xe_hp_systolic_gemm_kernel.hpp"

using namespace ngen;

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using dnnl::impl::utils::one_of;

static constexpr uint16_t flag_c_bias_row = FlagCORow;
static constexpr uint16_t flag_c_bias_col = FlagCOColumn;

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::barrier_prep(
        const InstructionModifier &swsb, const GRF &header) {
    and_<uint32_t>(1 | swsb, header[2], r0_save[2],
            uint32_t((hw >= HW::XeHPG) ? 0xFFFF0000 : 0x7F000000));
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::mul_constant(
        const InstructionModifier &mod, const RegData &dst, const RegData &src0,
        int32_t src1) {
    if (src1 == 0)
        mov(mod, dst, uint16_t(0));
    else if (src1 == 1) {
        if (dst != src0) mov(mod, dst, src0);
    } else if (ngen::utils::is_zero_or_pow2(src1))
        shl(mod, dst, src0, uint16_t(ngen::utils::log2(src1)));
    else
        mul(mod, dst, src0, src1);
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::zero_c() {
    for (int ii = 0; ii < (cfg.tile_m / 8); ii++) {
        for (int o = 0; o < cfg.tile_n; o += 4) {
            mov<float>(16, c_regs[ii * acc_stride + o], 0.0f);
            mov<uint32_t>(16, c_regs[ii * acc_stride + o + 2], 0);
        }
    }
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::scattered_setup_c(int stride, bool load) {
    // Set up SIMD16 scattered access pointers to emulate block access to C
    //   (2 columns x 8 regs/column).
    auto a_elem_bytes = getBytes(cfg.a_type);
    mov<uint16_t>(4, uheaders[15],
            Immediate::uv(0 * stride, 1 * stride, 2 * stride, 3 * stride, 0, 0,
                    0, 0));
    mov<uint16_t>(4, uheaders[15][0](4), uheaders[15][0](1));
    eadd<uint64_t>(4, uheaders[0], uc_base, uheaders[15].uw(0)(4));
    eadd<uint64_t>(4, uheaders[1], uheaders[0], stride * 4);
    eadd<uint64_t>(8, uheaders[2], uheaders[0], stride * 8);
    if (cfg.c_packed && a_elem_bytes == 1) {
        eadd<uint64_t>(8, uheaders[4], uheaders[0], stride * 16);
        eadd<uint64_t>(8, uheaders[6], uheaders[0], stride * 24);
        for (int q = 8; q < 16; q += 2)
            eadd<uint64_t>(8, uheaders[q], uheaders[q - 8], stride * 32);
    } else if (cfg.c_packed && a_elem_bytes == 2) {
        eadd<uint64_t>(8, uheaders[4], uheaders[0], stride * 16 * cfg.tile_n);
        eadd<uint64_t>(
                8, uheaders[6], uheaders[0], stride * (16 * cfg.tile_n + 8));
        for (int q = 8; q < 16; q += 2)
            eadd<uint64_t>(8, uheaders[q], uheaders[q - 8], stride * 16);
    } else {
        eadd<uint64_t>(8, uheaders[4], uheaders[0], stride * 16);
        eadd<uint64_t>(8, uheaders[6], uheaders[0], stride * 24);
        for (int q = 8; q < 16; q += 2)
            eadd<uint64_t>(8, uheaders[q], uheaders[q - 8], uldc);
    }
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::block_setup_c(bool remainder, bool load) {
    auto a_elem_bytes = getBytes(cfg.a_type);
    auto c_elem_bytes = getBytes(cfg.c_type);
    if (remainder) {
        // 8 blocks, each 16x1.
        emov<uint64_t>(1, uheaders[0][0], uc_base);
        if (cfg.c_packed && a_elem_bytes == 1) {
            eadd<uint64_t>(1, uheaders[1][0], uc_base, c_elem_bytes * 16);
            for (int q = 2; q < 16; q += 2)
                eadd<uint64_t>(
                        8, uheaders[q], uheaders[0], c_elem_bytes * 32 * q / 2);
        } else if (cfg.c_packed && a_elem_bytes == 2) {
            eadd<uint64_t>(
                    1, uheaders[1][0], uc_base, c_elem_bytes * 16 * cfg.tile_n);
            for (int q = 2; q < 16; q += 2)
                eadd<uint64_t>(
                        8, uheaders[q], uheaders[0], c_elem_bytes * 16 * q / 2);
        } else {
            eadd<uint64_t>(1, uheaders[1][0], uc_base, c_elem_bytes * 16);
            eadd<uint64_t>(8, uheaders[2], uheaders[0], uldc);
            eadd<uint64_t>(8, uheaders[4], uheaders[0], uldc_x2);
            eadd<uint64_t>(8, uheaders[6], uheaders[2], uldc_x2);
            for (int q = 8; q < 16; q += 2)
                eadd<uint64_t>(8, uheaders[q], uheaders[q - 8], uldc_x4);
        }
    } else {
        // 4 blocks, each 32x1.
        emov<uint64_t>(1, uheaders[0][0], uc_base);
        if (cfg.c_packed && a_elem_bytes == 1) {
            eadd<uint64_t>(1, uheaders[1][0], uc_base, c_elem_bytes * 32);
            eadd<uint64_t>(8, uheaders[2], uheaders[0], c_elem_bytes * 32 * 2);
            eadd<uint64_t>(8, uheaders[4], uheaders[0], c_elem_bytes * 32 * 4);
            eadd<uint64_t>(8, uheaders[6], uheaders[0], c_elem_bytes * 32 * 6);
        } else if (cfg.c_packed && a_elem_bytes == 2) {
            eadd<uint64_t>(
                    1, uheaders[1][0], uc_base, c_elem_bytes * 16 * cfg.tile_n);
            eadd<uint64_t>(8, uheaders[2], uheaders[0], c_elem_bytes * 16 * 2);
            eadd<uint64_t>(8, uheaders[4], uheaders[0], c_elem_bytes * 16 * 4);
            eadd<uint64_t>(8, uheaders[6], uheaders[0], c_elem_bytes * 16 * 6);
        } else {
            eadd<uint64_t>(1, uheaders[1][0], uc_base, uldc);
            eadd<uint64_t>(8, uheaders[2], uheaders[0], uldc_x2);
            eadd<uint64_t>(8, uheaders[4], uheaders[0], uldc_x4);
            eadd<uint64_t>(8, uheaders[6], uheaders[2], uldc_x4);
        }
    }
}

template <HW hw>
int xehp_systolic_gemm_kernel_t<hw>::interleave(int j) {
    // Convert logical column index in C to the corresponding interleaved index.
    int half_n = cfg.tile_n / 2;
    bool second = (j >= half_n);
    if (second) j -= half_n;
    return ((j & ~3) << 1) + (int(second) << 2) + (j & 3);
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::load_update_c_internal(
        bool remainder, bool c_align16) {
    Label done;

    // Get configuration options.
    bool alpha1 = cfg.alpha1;
    bool beta0 = cfg.beta0;
    bool beta1 = cfg.beta1;
    bool float_update = !(alpha1 && (beta0 || beta1)) || cfg.have_post_op();

    const auto a_elem_bytes = getBytes(cfg.a_type);
    const auto c_elem_bytes = getBytes(cfg.c_type);
    bool c32 = (c_elem_bytes == 4);

    if (beta0 && alpha1 && !cfg.have_post_op()) return; // Nothing to do.

    // Get the bank ID for a given register.
    auto bank = [](const RegData &r) { return (r.getBase() & 2) >> 1; };

    // Load a 32x4 block of C and increment pointers.
    auto c_load = [&](int j0) {
        Label skip;

        if (beta0) return;

        if (remainder) {
            // Check load j0 + 1.
            cmp(1 | gt | f1[1], null.ud(), un_rem, uint16_t(j0 + 1));
        }

        for (int j = j0; j < j0 + 4; j++) {
            auto jj = (j & 7);

            if (remainder) {
                // Skip this load if masked off. Otherwise, prepare next flag.
                jmpi(1 | ~f1[j & 1], skip);
                if (j + 2 < cfg.tile_n)
                    cmp(1 | gt | f1[j & 1], null.ud(), un_rem, uint16_t(j + 2));
            }

            if (c_align16) {
                if (remainder) {
                    // Block read with masks.
                    assert(c32);
                    load(16 | f0[0], utemp[jj * 4 + 0], block_oword(4), A64,
                            uheaders[2 * jj + 0]);
                    load(16 | f0[1], utemp[jj * 4 + 2], block_oword(4), A64,
                            uheaders[2 * jj + 1]);
                } else {
                    // Block read.
                    load(16, utemp[jj * 4 + 4 - c_elem_bytes],
                            aligned_block_oword(c_elem_bytes * 2), A64,
                            uheaders[jj]);
                }
            } else {
                // Scattered byte or dword load, possibly masked.
                auto j1 = (j & 1);
                InstructionModifier mod0 = 16;
                InstructionModifier mod1 = 16;
                if (remainder) {
                    mod0 = mod0 | f0[0];
                    mod1 = mod1 | f0[1];
                }
                if (c32) {
                    load(mod0, utemp[jj * 4 + 0], scattered_dword(1), A64,
                            uheaders[8 * j1 + 0]);
                    load(mod1, utemp[jj * 4 + 2], scattered_dword(1), A64,
                            uheaders[8 * j1 + 4]);
                } else {
                    load(mod0, utemp[jj * 4 + 0], scattered_byte(c_elem_bytes),
                            A64, uheaders[8 * j1 + 0]);
                    load(mod1, utemp[jj * 4 + 2], scattered_byte(c_elem_bytes),
                            A64, uheaders[8 * j1 + 4]);
                }
                if (j + 2 < cfg.tile_n) {
                    for (int q = 0; q < 8; q += 2) {
                        if (cfg.c_packed)
                            eadd<uint64_t>(8, uheaders[8 * j1 + q],
                                    uheaders[8 * j1 + q],
                                    c_elem_bytes * packed_ldc() * 2);
                        else
                            eadd<uint64_t>(8, uheaders[8 * j1 + q],
                                    uheaders[8 * j1 + q], uldc_x2);
                    }
                }
            }
        }

        if (c_align16 && (j0 + 8 < cfg.tile_n)) {
            // Increment pointers.
            for (int j = j0; j < j0 + 4; j++) {
                auto jj = (j & 7);
                if (remainder) {
                    if (cfg.c_packed) {
                        eadd<uint64_t>(1, uheaders[2 * jj + 0],
                                uheaders[2 * jj + 0],
                                c_elem_bytes * packed_ldc() * 8);
                        eadd<uint64_t>(1, uheaders[2 * jj + 1],
                                uheaders[2 * jj + 1],
                                c_elem_bytes * packed_ldc() * 8);
                    } else {
                        eadd<uint64_t>(1, uheaders[2 * jj + 0],
                                uheaders[2 * jj + 0], uldc_x8);
                        eadd<uint64_t>(1, uheaders[2 * jj + 1],
                                uheaders[2 * jj + 1], uldc_x8);
                    }
                } else {
                    if (cfg.c_packed)
                        eadd<uint64_t>(1, uheaders[jj], uheaders[jj],
                                c_elem_bytes * packed_ldc() * 8);
                    else
                        eadd<uint64_t>(1, uheaders[jj], uheaders[jj], uldc_x8);
                }
            }
        }

        mark(skip);
    };

    if (remainder) {
        // Do the first n compare.
        cmp(1 | gt | f1[0], null.ud(), un_rem, uint32_t(0));
    }

    // Set up headers.
    if (c_align16)
        block_setup_c(remainder, true);
    else
        scattered_setup_c(c_elem_bytes, true);

    // Get first load ready.
    c_load(0);

    for (int j0 = 0; j0 < cfg.tile_n; j0 += 4) {
        int j0_4 = j0 & 4;

        // Get (sub)register in loaded C submatrix at offset (ii*8, jj).
        auto get_load_reg = [&](DataType dt, int ii, int jj) {
            if (cfg.c_packed && c_align16 && !remainder && a_elem_bytes == 2) {
                int bi = (ii & 2) >> 1;
                int bj = (jj & 1) << 1;
                ii = (ii & ~2) | bj;
                jj = (jj & ~1) | bi;
            }
            auto bytes = c_align16 ? getBytes(dt) : 4;
            auto stride = bytes / getBytes(dt);
            auto per_reg = 4 / bytes;
            auto reg = utemp[j0_4 * 4 + (4 - bytes) + (ii / per_reg) + jj * 4];
            auto off = (ii % per_reg) * 8;

            return reg.sub(off, dt)(stride);
        };

        // Get register in accumulated C submatrix at offset (ii*8, jj).
        auto get_acc_reg = [&](DataType dt, int ii, int jj) {
            auto acc_base = c_regs[interleave(j0)];
            return acc_base.advance(jj + ii * acc_stride).retype(dt);
        };

        // Load C block ahead of time for next loop, and check for loop exit.
        if ((j0 + 4) < cfg.tile_n) {
            c_load(j0 + 4);
            if (remainder)
                cmp(1 | gt | f1[1], null.ud(), un_rem, uint16_t(j0 + 4));
        }

        // If accumulator not single precision, convert to single precision, unless alpha = 1 and beta is 0 or 1.
        auto cur_acc_type = cfg.acc_type;
        if (float_update && (cfg.acc_type != DataType::f)) {
            for (int ii = 0; ii < (cfg.tile_m / 8); ii++) {
                for (int jj = 0; jj < 4; jj += 2) {
                    auto acc_orig = get_acc_reg(cfg.acc_type, ii, jj);
                    auto acc_f = get_acc_reg(DataType::f, ii, jj);
                    mov(16, acc_f, acc_orig);
                }
            }
            cur_acc_type = DataType::f;
        }

        // Premultiply by alpha if both alpha is not 1, unless beta = 1 (use FMA later instead).
        if (!alpha1 && !beta1) {
            for (int ii = 0; ii < (cfg.tile_m / 8); ii++) {
                for (int jj = 0; jj < 4; jj += 2) {
                    auto a_reg = get_acc_reg(cur_acc_type, ii, jj);
                    mul(16, a_reg, a_reg, ualpha_regs[!bank(a_reg)]);
                }
            }
        }

        // Half-precision C must be upconverted to single precision separately (no hf/f mixed mode support in XeHP).
        // Similarly integer C must be upconverted if alpha or beta float.
        auto old_type = cfg.c_type;
        if (!beta0) {
            if ((float_update && one_of(cfg.c_type, DataType::d, DataType::ud))
                    || (cfg.c_type == DataType::hf)) {
                for (int jj = 0; jj < 4; jj++) {
                    for (int ii = 0; ii < (cfg.tile_m / 8); ii += 2) {
                        auto old_c = get_load_reg(cfg.c_type, ii, jj);
                        auto old_f = get_load_reg(cur_acc_type, ii, jj);
                        mov(16, old_f, old_c);
                    }
                }
                old_type = cur_acc_type;
            }
        }

        // Non-packed bfloat16 must also be upconverted. Can't use mov for this.
        if (!beta0 && !c_align16 && (cfg.c_type == DataType::bf)) {
            for (int jj = 0; jj < 4; jj++) {
                for (int ii = 0; ii < (cfg.tile_m / 8); ii += 2) {
                    auto old_ud = get_load_reg(DataType::ud, ii, jj);
                    shl(16, old_ud, old_ud, uint16_t(16));
                }
            }
            old_type = DataType::f;
        }

        // Main alpha/beta scaling.
        for (int ii = 0; ii < (cfg.tile_m / 8); ii++) {
            for (int jj = 0; jj < 4; jj++) {
                auto a_reg = get_acc_reg(cur_acc_type, ii, jj);
                auto o_reg = get_load_reg(old_type, ii, jj);
                int b = !bank(a_reg);

                if (beta0) {
                    /* no op */
                } else if (beta1) {
                    if (alpha1)
                        add(8, a_reg, a_reg, o_reg);
                    else
                        mad(8, a_reg, o_reg, a_reg, ualpha_regs[b]);
                } else
                    mad(8, a_reg, a_reg, o_reg, ubeta_regs[b]);
            }
        }

        // Post-ops.
        if (cfg.have_post_op()) {
            for (int ii = 0; ii < (cfg.tile_m / 8); ii++) {
                auto range_start = get_acc_reg(cur_acc_type, ii, 0);
                post_op_injector->compute(GRFRange(range_start.getBase(), 4));
            }
        }

        // Convert back from single precision, if needed.
        if (cfg.acc_type != cur_acc_type) {
            for (int ii = 0; ii < (cfg.tile_m / 8); ii++) {
                for (int jj = 0; jj < 4; jj += 2) {
                    auto acc_cur = get_acc_reg(cur_acc_type, ii, jj);
                    auto acc_orig = get_acc_reg(cfg.acc_type, ii, jj);
                    mov(16, acc_orig, acc_cur);
                }
            }
            cur_acc_type = cfg.acc_type;
        }

        // Early exit if no more columns to load.
        if (remainder && (j0 + 4 < cfg.tile_n)) jmpi(1 | ~f1[1], done);
    }

    mark(done);
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::load_update_c(
        bool remainder, bool c_align16) {
    if (cfg.early_c_bias) add_c_bias();
    load_update_c_internal(remainder, c_align16);
    if (!cfg.early_c_bias) add_c_bias();
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::store_c(bool remainder, bool c_align16) {
    Label done;

    const auto a_elem_bytes = getBytes(cfg.a_type);
    const auto c_elem_bytes = getBytes(cfg.c_type);
    bool c32 = (c_elem_bytes == 4);

    if (remainder) {
        // Do the first two n compares.
        cmp(1 | gt | f1[0], un_rem, uint16_t(0));
        cmp(1 | gt | f1[1], un_rem, uint16_t(1));
    }

    // Set up headers. TODO: reuse headers from load where possible.
    if (c_align16)
        block_setup_c(remainder, false);
    else
        scattered_setup_c(c_elem_bytes, false);

    for (int j0 = 0; j0 < cfg.tile_n; j0 += 4) {
        int j0_4 = j0 & 4;

        auto acc = c_regs[interleave(j0)];

        // Get (sub)register in stored C submatrix at offset (ii*8, jj).
        auto get_store_reg = [&](int ii, int jj) {
            if (cfg.c_packed && c_align16 && !remainder && a_elem_bytes == 2) {
                int bi = (ii & 2) >> 1;
                int bj = (jj & 1) << 1;
                ii = (ii & ~2) | bj;
                jj = (jj & ~1) | bi;
            }
            auto bytes = c_align16 ? c_elem_bytes : 4;
            auto stride = bytes / c_elem_bytes;
            auto per_reg = 4 / bytes;
            auto reg = utemp[j0_4 * 4 + (ii / per_reg) + jj * 4];
            auto off = (ii % per_reg) * 8;

            return reg.sub(off, cfg.c_type)(stride);
        };

        // 4x4 transpose of 8x1 blocks, downconverting if necessary.
        for (int ii = 0; ii < (cfg.tile_m / 8); ii++) {
            for (int jj = 0; jj < 4; jj++) {
                GRF a_reg = acc.advance(jj + ii * acc_stride)
                                    .retype(cfg.acc_type);
                auto dreg = get_store_reg(ii, jj);

                if (a_reg.getType() == dreg.getType()) {
                    // Use float moves for raw moves.
                    a_reg.setType(DataType::f);
                    dreg.setType(DataType::f);
                }

                mov<float>(8, dreg, a_reg);
            }
        }

        // Store C.
        for (int j = j0; j < j0 + 4; j++) {
            auto jj = (j & 7);

            if (remainder) {
                // Skip this store if masked off. Otherwise, prepare next flag.
                if (!cfg.c_packed) {
                    jmpi(1 | ~f1[j & 1], done);
                } else {
                    // Zero out instead of masking off.
                    Label zero_out_done;
                    jmpi(1 | f1[j & 1], zero_out_done);
                    mov<uint32_t>(16, utemp[jj * 4 + 0], 0u);
                    mov<uint32_t>(16, utemp[jj * 4 + 2], 0u);
                    mark(zero_out_done);
                }
                if (j + 2 < cfg.tile_n)
                    cmp(1 | gt | f1[j & 1], un_rem, uint16_t(j + 2));
            }

            if (c_align16) {
                if (remainder) {
                    // Block write with masks.
                    assert(c32);
                    InstructionModifier mod0 = 16 | f0[0];
                    InstructionModifier mod1 = 16 | f0[1];
                    if (cfg.c_packed) {
                        mod0 = mod0 | any16h;
                        mod1 = mod1 | any16h;

                        // Zero-out remainders in registers.
                        mov<uint32_t>(16 | ~f0[0], utemp[jj * 4 + 0], 0u);
                        mov<uint32_t>(16 | ~f0[1], utemp[jj * 4 + 2], 0u);
                    }

                    store(mod0, block_oword(4), A64, uheaders[2 * jj + 0],
                            utemp[jj * 4 + 0]);
                    store(mod1, block_oword(4), A64, uheaders[2 * jj + 1],
                            utemp[jj * 4 + 2]);
                } else {
                    // Block write.
                    store(16, block_oword(2 * c_elem_bytes), A64, uheaders[jj],
                            utemp[jj * 4]);
                }

                if ((jj == 7) && (j0 + 8 < cfg.tile_n)) {
                    // Increment all block write pointers.
                    sync(SyncFunction::allrd);
                    for (int q = 0; q < (remainder ? 16 : 8); q += 2) {
                        if (cfg.c_packed)
                            eadd<uint64_t>(8, uheaders[q], uheaders[q],
                                    c_elem_bytes * packed_ldc() * 8);
                        else
                            eadd<uint64_t>(
                                    8, uheaders[q], uheaders[q], uldc_x8);
                    }
                }
            } else {
                // Scattered dword or byte store, possibly masked.
                auto j1 = (j & 1);
                InstructionModifier mod0 = 16;
                InstructionModifier mod1 = 16;
                if (remainder) {
                    mod0 = mod0 | f0[0];
                    mod1 = mod1 | f0[1];
                }
                if (remainder && cfg.c_packed) {
                    mod0 = mod0 | any16h;
                    mod1 = mod1 | any16h;

                    // Zero-out remainders in registers.
                    mov<uint32_t>(16 | ~f0[0], utemp[jj * 4 + 0], 0u);
                    mov<uint32_t>(16 | ~f0[1], utemp[jj * 4 + 2], 0u);
                }

                if (c32) {
                    store(mod0, scattered_dword(1), A64, uheaders[8 * j1 + 0],
                            utemp[jj * 4 + 0]);
                    store(mod1, scattered_dword(1), A64, uheaders[8 * j1 + 4],
                            utemp[jj * 4 + 2]);
                } else {
                    store(mod0, scattered_byte(c_elem_bytes), A64,
                            uheaders[8 * j1 + 0], utemp[jj * 4 + 0]);
                    store(mod1, scattered_byte(c_elem_bytes), A64,
                            uheaders[8 * j1 + 4], utemp[jj * 4 + 2]);
                }
                if ((j1 == 1) && (j + 2 < cfg.tile_n)) {
                    // Increment all scattered pointers at once.
                    sync(SyncFunction::allrd);
                    for (int q = 0; q < 16; q += 2) {
                        if (cfg.c_packed)
                            eadd<uint64_t>(8, uheaders[q], uheaders[q],
                                    c_elem_bytes * packed_ldc() * 2);
                        else
                            eadd<uint64_t>(
                                    8, uheaders[q], uheaders[q], uldc_x2);
                    }
                }
            }
        }
    }

    mark(done);
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::load_c_bias(bool remainder) {
    if (cfg.c_bias == bias_t::runtime) {
        Label label_row, label_col, label_done;
        and_(1 | nz | f1[0], null.ud(), uflags, flag_c_bias_col);
        and_(1 | nz | f1[1], null.ud(), uflags, flag_c_bias_row);
        jmpi(1 | f1[0], label_col);
        jmpi(1 | f1[1], label_row);

        load_c_bias(bias_t::fixed, remainder);
        jmpi(1, label_done);

        mark(label_col);
        load_c_bias(bias_t::column, remainder);
        jmpi(1, label_done);

        mark(label_row);
        load_c_bias(bias_t::row, remainder);

        mark(label_done);
    } else
        load_c_bias(cfg.c_bias, remainder);
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::load_c_bias(
        bias_t c_bias, bool remainder) {
    if (!remainder && getBytes(cfg.co_type) == 4)
        load_c_bias_block(c_bias);
    else
        load_c_bias_scattered(c_bias, remainder);
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::load_c_bias_block(bias_t c_bias) {
    assert(uoff_co.getOffset() == 2 && uoff_co2.getOffset() == 2);

    switch (c_bias) {
        case bias_t::none: break;
        case bias_t::fixed:
            mov(1, uheaders[0].ud(0), uoff_co);
            load(1, uoffset[0], surface_dword(ChannelMask::r),
                    Surface(co_surface), uheaders[0]);
            break;
        case bias_t::row:
        case bias_t::column: {
            int tile = (c_bias == bias_t::column) ? cfg.tile_n : cfg.tile_m;
            if (tile > 32) {
                add(1, uoff_co2, uoff_co, uint16_t(32 * 4));
                load(16, uoffset[0], aligned_block_oword(8),
                        Surface(co_surface), uoff_co);
                load(16, uoffset[4], aligned_block_oword(4),
                        Surface(co_surface), uoff_co2);
            } else
                load(16, uoffset[0], aligned_block_oword(8),
                        Surface(co_surface), uoff_co);
            break;
        }
        case bias_t::runtime: assert(!"Bias type not specified");
    }
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::load_c_bias_scattered(
        bias_t c_bias, bool remainder) {
    auto bytes = getBytes(cfg.co_type);
    auto lg2_bytes = ngen::utils::log2(bytes);

    switch (c_bias) {
        case bias_t::none: break;
        case bias_t::fixed:
            mov(1, uheaders[0].ud(0), uoff_co);
            load(1, uoffset[0], scattered_byte(bytes), Surface(co_surface),
                    uheaders[0]);
            break;
        case bias_t::row:
        case bias_t::column: {
            bool column = (c_bias == bias_t::column);
            auto index_vec = uheaders[6].uw();
            auto index_plus_16 = uheaders[7].uw();
            auto index_plus_32 = uheaders[8].uw();
            int tile = column ? cfg.tile_n : cfg.tile_m;

            mov(8, index_vec[0](1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
            mov(8, index_vec[8](1),
                    Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));

            InstructionModifier flag[3];
            if (remainder) {
                if (column) {
                    add(16, index_plus_16, index_vec, uint16_t(16));
                    add(16, index_plus_32, index_vec, uint16_t(32));
                    cmp(32 | gt | f1[0], un_rem.uw(), index_vec);
                    flag[0] |= f1[0];
                    flag[1] |= f1[1];
                    flag[2] |= f1[0];
                } else {
                    // Reuse saved row mask.
                    flag[0] |= f0[0];
                    flag[1] |= f0[1];
                }
            }

            if (bytes > 1)
                shl<uint16_t>(16, index_vec, index_vec, uint16_t(lg2_bytes));

            add(16, uheaders[0].ud(), uoff_co, index_vec);
            add3(16, uheaders[2].ud(), uoff_co, index_vec,
                    uint16_t(16 * bytes));
            if (tile > 32)
                add3(16, uheaders[4].ud(), uoff_co, index_vec,
                        uint16_t(32 * bytes));

            load(16 | flag[0], uoffset[0], scattered_byte(bytes),
                    Surface(co_surface), uheaders[0]);
            load(16 | flag[1], uoffset[2], scattered_byte(bytes),
                    Surface(co_surface), uheaders[2]);

            if (tile > 32) {
                if (column && remainder)
                    cmp(16 | gt | f1[0], un_rem.uw(), index_plus_32);
                load(16 | flag[2], uoffset[4], scattered_byte(bytes),
                        Surface(co_surface), uheaders[4]);
            }

            break;
        }
        case bias_t::runtime: assert(!"Bias type not specified");
    }
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::convert_c_bias(
        bias_t c_bias, ngen::DataType dst_type) {
    auto src_type = cfg.co_type;

    auto convert = [&](int simd, RegData dst, RegData src) {
        if (src_type == DataType::bf && dst_type == DataType::f) {
            // Must convert bf->f by hand.
            src.setType(DataType::uw);
            dst.setType(DataType::ud);
            shl(simd, dst, src, uint16_t(16));
        } else
            mov(simd, dst, src);
    };

    if (src_type != dst_type) {
        switch (c_bias) {
            case bias_t::none: break;
            case bias_t::fixed:
                convert(1, uoffset[0].sub(0, dst_type),
                        uoffset[0].sub(0, src_type));
                break;
            case bias_t::row:
            case bias_t::column: {
                assert(getBytes(dst_type) == 4);
                int nreg
                        = ((c_bias == bias_t::column) ? cfg.tile_n : cfg.tile_m)
                        / 8;
                auto src_stride = 4 / getBytes(src_type);
                for (int q = 0; q < nreg; q += 2)
                    convert(16, uoffset[q].sub(0, dst_type)(1),
                            uoffset[q].sub(0, src_type)(src_stride));
                break;
            }
            case bias_t::runtime: assert(!"Bias type not specified");
        }
    }
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::add_c_bias() {
    if (cfg.c_bias == bias_t::runtime) {
        Label label_row, label_col, label_done;
        and_(1 | nz | f1[0], null.ud(), uflags, flag_c_bias_col);
        and_(1 | nz | f1[1], null.ud(), uflags, flag_c_bias_row);
        jmpi(1 | f1[0], label_col);
        jmpi(1 | f1[1], label_row);

        add_c_bias(bias_t::fixed);
        jmpi(1, label_done);

        mark(label_col);
        add_c_bias(bias_t::column);
        jmpi(1, label_done);

        mark(label_row);
        add_c_bias(bias_t::row);

        mark(label_done);
    } else
        add_c_bias(cfg.c_bias);
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::add_c_bias(bias_t c_bias) {
    auto cur_type = cfg.early_c_bias ? cfg.acc_type : cfg.c_type;

    convert_c_bias(c_bias, cur_type);

    auto co_fixed = uoffset[0].sub(0, cur_type);
    if (merge_abc_bias()) add(1, co_fixed, co_fixed, uao_bo_k);

    for (int ii = 0; ii < (cfg.tile_m / 8); ii++) {
        for (int j = 0; j < cfg.tile_n; j += 2) {
            auto a_reg
                    = c_regs[interleave(j) + ii * acc_stride].retype(cur_type);
            switch (c_bias) {
                case bias_t::none: break;
                case bias_t::fixed: add(16, a_reg, a_reg, co_fixed); break;
                case bias_t::row:
                    if (cur_type != DataType::f) {
                        add(16, a_reg, a_reg,
                                uoffset[ii].sub(0, cur_type)(0, 8, 1));
                    } else {
                        // No region support on FPU pipe.
                        add(8, a_reg, a_reg, uoffset[ii].sub(0, cur_type)(1));
                        a_reg++;
                        add(8, a_reg, a_reg, uoffset[ii].sub(0, cur_type)(1));
                    }
                    break;
                case bias_t::column:
                    if (cur_type != DataType::f) {
                        add(16, a_reg, a_reg,
                                uoffset[j >> 3].sub(j & 7, cur_type)(1, 8, 0));
                    } else {
                        add(8, a_reg, a_reg,
                                uoffset[j >> 3].sub(j & 7, cur_type)(0));
                        a_reg++;
                        add(8, a_reg, a_reg,
                                uoffset[j >> 3].sub((j + 1) & 7, cur_type)(0));
                    }
                    break;
                case bias_t::runtime: assert(!"Bias type not specified");
            }
        }
    }
}

template <HW hw>
bool xehp_systolic_gemm_kernel_t<hw>::merge_abc_bias() {
    return cfg.a_bias && cfg.b_bias && (cfg.c_bias == bias_t::fixed)
            && cfg.alpha1;
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::add_ab_bias() {
    auto a_row_sums = utemp[6] - utemp[9];
    auto b_col_sums = utemp[0] - utemp[5];
    GRF headers[3] = {uheaders[0], uheaders[2], uheaders[4]};

    // Precompute ao * bo * k if needed.
    if (cfg.a_bias && cfg.b_bias) {
        mul(1, uao_bo_k, uk, uao);
        mul(1, uao_bo_k, uao_bo_k, ubo);
    }

    // Load A row sums and B column sums.
    if (cfg.a_bias) {
        mov(1, headers[0].ud(2), off_bsum_save);
        if (cfg.tile_n > 32)
            add(1, headers[1].ud(2), off_bsum_save, uint16_t(32 * 4));
        load(16, b_col_sums[0], aligned_block_oword(8), Surface(bp_surface),
                headers[0]);
        if (cfg.tile_n > 32)
            load(16, b_col_sums[4], aligned_block_oword(4), Surface(bp_surface),
                    headers[1]);
    }
    if (cfg.b_bias) {
        mov(1, headers[2].ud(2), off_asum_save);
        load(16, a_row_sums[0], aligned_block_oword(8), Surface(ap_surface),
                headers[2]);
    }
    // Compute bias contributions.
    if (cfg.a_bias) {
        // ao * b_sum
        for (int j = 0; j < cfg.tile_n; j++) {
            for (int ii = 0; ii < (cfg.tile_m / 8); ii++) {
                auto a_reg = c_regs[interleave(j) + ii * acc_stride].retype(
                        cfg.acc_type);
                mad(8, a_reg, a_reg, b_col_sums[j / 8].d(j % 8), uao);
            }
        }
    }
    if (cfg.b_bias) {
        // a_sum * bo
        for (int ii = 0; ii < (cfg.tile_m / 8); ii++) {
            for (int j = 0; j < cfg.tile_n; j++) {
                auto a_reg = c_regs[interleave(j) + ii * acc_stride].retype(
                        cfg.acc_type);
                mad(8, a_reg, a_reg, a_row_sums[ii].d(), ubo);
            }
        }
    }
    if (cfg.a_bias && cfg.b_bias && !merge_abc_bias()) {
        // ao * bo * k (if not possible to absorb into co)
        for (int ii = 0; ii < (cfg.tile_m / 8); ii++) {
            for (int o = 0; o < cfg.tile_n; o += 2) {
                auto c_reg = c_regs[ii * acc_stride + o].retype(cfg.acc_type);
                add(16, c_reg, c_reg, uao_bo_k);
            }
        }
    }
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::update_c(bool remainder) {
    // C is arranged in 8x8 column major blocks organized in a 4x6 row major array (allowing a tile size of up to 32x48).
    // Each 8x8 block is split in two 8x4 blocks (due to dpasw).
    // Rearrange into contiguous columns and use hword x4 stores, taking 4 columns at a time.
    //   (effectively a 4x4 whole-register transpose)
    // This burns through icache -- consider rewriting with indirect accesses.

    bool c32 = (getBytes(cfg.c_type) == 4);

    if (remainder) {
        // Set up m mask in f0.0:ud.
        auto t0 = r25 - r26;
        mov<uint16_t>(8, t0[0][0](1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        mov<uint16_t>(
                8, t0[0][8](1), Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));
        add<uint16_t>(16, t0[1], t0[0], 16u);
        cmp<uint16_t>(32 | gt | f0[0], um_rem, t0);
    }

    // Set up headers and multiples of LDC (= ldc in bytes). TODO collapse into one instruction.
    if (!cfg.c_packed) {
        shl(1, uldc_x2, uldc, uint16_t(1));
        shl(1, uldc_x4, uldc, uint16_t(2));
        shl(1, uldc_x8, uldc, uint16_t(3));
    }

    // Check whether C pointer has given (power of 2) alignment. Result stored in f1.1.
    auto check_c_align = [&](int align) {
        auto uc_align = r18.ud(0);
        if (cfg.c_packed)
            uc_align = uc_base.ud(0);
        else
            or_(1, uc_align, uldc, uc_base.ud(0));
        and_(1 | ze | f1[1], null.ud(), uc_align, uint16_t(align - 1));
    };

    Label unaligned_c;

    load_c_bias(remainder);
    if (!cfg.c_align16_check) {
        // Assume 16-byte alignment.
        load_update_c(remainder, true);
        store_c(remainder, true);
    } else if (!c32 && remainder) {
        // C not 32-bit, remainder. Only one (unaligned) path.
        load_update_c(remainder, false);
        store_c(remainder, false);
    } else {
        // Two full paths, one with aligned C, one without.
        check_c_align(16);
        jmpi(1 | ~f1[1], unaligned_c);
        load_update_c(remainder, true);
        store_c(remainder, true);
        epilogue();
        mark(unaligned_c);
        load_update_c(remainder, false);
        store_c(remainder, false);
    }
}

// Update C, checking at runtime for remainder handling.
template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::update_c() {
    Label partial_c;

    // Turn on auto-SWSB for the remainder of the kernel.
    setDefaultAutoSWSB();

    // Start 64-bit emulation if needed.
    if (cfg.emulate64) {
        emu_state.temp[0] = uemulate_temp[0];
        emu_state.temp[1] = uemulate_temp[1];
    }

    // Move C pointer to safety. Also bias flags if needed.
    emov(1, uc_base, c_ptr_mem);

    if (cfg.c_bias == bias_t::runtime) mov(1, uflags, flags_save);

    // Pull saved data from accumulators. Note moves to/from accumulator don't support full swizzling, so
    //  the subregister offsets for the following movs must match.
    assert(ubase.getByteOffset() == 0 && base_save.getByteOffset() == 0);
    mov(8, ubase, base_save);

    assert(ualpha_regs[1].getByteOffset() == alpha_save.getByteOffset());
    mov(2, ualpha_regs[1].ud()(1), alpha_save.ud()(1));

    // Add A/B bias terms.
    add_ab_bias();

    // Do remainder check.
    if (!cfg.c_remainder)
        update_c(false);
    else {
        jmpi(1 | f0[0] | any16h, partial_c);
        update_c(false);
        epilogue();

        mark(partial_c);
        update_c(true);
    }
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::dpasw_typed(
        const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount,
        const GRF &c_reg, const GRF &a_reg, const GRF &b_reg) {
    dpasw(mod, sdepth, rcount, c_reg.retype(cfg.acc_type),
            c_reg.retype(cfg.acc_type), a_reg.retype(cfg.a_type),
            b_reg.retype(cfg.b_type));
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::multiply_chunk(int ao, int i0, bool waitb,
        const InstructionModifier &swsb0, const InstructionModifier &swsb_end) {
    int co = i0 * 6;

    if (cfg.tile_n > 32) {
        if (waitb) {
            dpasw_typed(8 | swsb0 | Atomic, 8, 8, c_regs[co], a_regs[ao],
                    b_regs[0]);
            dpasw_typed(8, 8, 8, c_regs[co + 8], a_regs[ao], b_regs[4]);
            dpasw_typed(8 | sb1.dst | Atomic, 8, 8, c_regs[co + 16], a_regs[ao],
                    b_regs[8]);
            dpasw_typed(8, 8, 8, c_regs[co + 24], a_regs[ao], b_regs[12]);
            dpasw_typed(8 | sb2.dst | Atomic, 8, 8, c_regs[co + 32], a_regs[ao],
                    b_regs[16]);
            dpasw_typed(8 | swsb_end, 8, 8, c_regs[co + 40], a_regs[ao],
                    b_regs[20]);
        } else {
            dpasw_typed(8 | swsb0 | Atomic, 8, 8, c_regs[co], a_regs[ao],
                    b_regs[0]);
            dpasw_typed(
                    8 | Atomic, 8, 8, c_regs[co + 8], a_regs[ao], b_regs[4]);
            dpasw_typed(
                    8 | Atomic, 8, 8, c_regs[co + 16], a_regs[ao], b_regs[8]);
            dpasw_typed(
                    8 | Atomic, 8, 8, c_regs[co + 24], a_regs[ao], b_regs[12]);
            dpasw_typed(
                    8 | Atomic, 8, 8, c_regs[co + 32], a_regs[ao], b_regs[16]);
            dpasw_typed(8 | swsb_end, 8, 8, c_regs[co + 40], a_regs[ao],
                    b_regs[20]);
        }
    } else {
        if (waitb) {
            dpasw_typed(8 | swsb0 | Atomic, 8, 8, c_regs[co], a_regs[ao],
                    b_regs[0]);
            dpasw_typed(8, 8, 8, c_regs[co + 8], a_regs[ao], b_regs[4]);
            dpasw_typed(8 | sb1.dst | Atomic, 8, 8, c_regs[co + 16], a_regs[ao],
                    b_regs[8]);
            dpasw_typed(8 | swsb_end, 8, 8, c_regs[co + 24], a_regs[ao],
                    b_regs[12]);
        } else {
            dpasw_typed(8 | swsb0 | Atomic, 8, 8, c_regs[co], a_regs[ao],
                    b_regs[0]);
            dpasw_typed(
                    8 | Atomic, 8, 8, c_regs[co + 8], a_regs[ao], b_regs[4]);
            dpasw_typed(
                    8 | Atomic, 8, 8, c_regs[co + 16], a_regs[ao], b_regs[8]);
            dpasw_typed(8 | swsb_end, 8, 8, c_regs[co + 24], a_regs[ao],
                    b_regs[12]);
        }
    }
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::multiply(int buffer, bool last_multiply) {
    // Load half of A (16x32) -- hopefully broadcast from SLM to this row -- and half of B, interleaved.
    InstructionModifier swsb = last_multiply ? SWSB(1) : dep_addr0;

    mov(1 | swsb, addr0.ud(2), slm_a_offset_load);
    mov(1 | dep_addr1, addr1.ud(2), slm_b_offset_load);
    add(1 | dep_addr2, addr2.ud(2), slm_b_offset_load, uint16_t(8 * 32 / 16));
    add(1 | dep_addr3, addr3.ud(2), slm_b_offset_load, uint16_t(16 * 32 / 16));

    if (cfg.alt_barriers) barrierwait();

    if (cfg.fulsim) sync(SyncFunction::nop, SWSB<int64_t>(1));
    sync.nop(sb5.src);
    load(16 | SWSB(sb3, 4), a_regs[0], block_oword(16), SLM, addr0);
    load(16 | SWSB(sb0, 3), b_regs[0], block_oword(16), SLM, addr1);
    load(16 | SWSB(sb1, 2), b_regs[8], block_oword(16), SLM, addr2);
    if (cfg.tile_n > 32)
        load(16 | SWSB(sb2, 1), b_regs[16], block_oword(16), SLM, addr3);

    add(1 | sb3.src, addr0.ud(2), slm_a_offset_load, uint16_t(8 * 32 / 16));
    add(1 | sb0.src, addr1.ud(2), slm_a_offset_load, uint16_t(16 * 32 / 16));
    add(1 | sb1.src, addr2.ud(2), slm_a_offset_load, uint16_t(24 * 32 / 16));
    load(16 | SWSB(sb4, 3), a_regs[8], block_oword(16), SLM, addr0);

    // Wait for A data to load.
    sync.allwr(0x18);

    if (cfg.alt_barriers && !last_multiply) {
        barrier_prep(sb2.src, addr3);
        barriermsg(SWSB(sb15, 1), addr3);
    }

    // Rows 0-7
    multiply_chunk(0, 0, true, sb0.dst, sb3);

    // Rows 8-15
    multiply_chunk(8, 8, false, InstructionModifier(), sb4);

    // Load third quarter of A (8x32)
    load(16 | SWSB(sb3, 2), a_regs[0], block_oword(16), SLM, addr1);

    // Rows 16-23
    multiply_chunk(0, 16, false, sb3.dst);

    // Load last quarter of A (8x32)
    load(16 | SWSB(sb4, 1), a_regs[8], block_oword(16), SLM, addr2);

    // Increment A and B to next buffer.
    swsb = cfg.fulsim ? InstructionModifier(sb3.src) : InstructionModifier();
    if (buffer == 2)
        mov(2 | swsb, slm_a_offset_load(1), slm_a_offset_load_init(1));
    else
        add(2 | swsb, slm_a_offset_load(1), slm_a_offset_load(1),
                uint16_t(slm_buf_size() / 16));

    // Rows 24-31
    multiply_chunk(8, 24, false, sb4.dst, sb5);

    // Remember dependencies for address registers.
    dep_addr0 = InstructionModifier {};
    dep_addr1 = sb3.src;
    dep_addr2 = sb4.src;
    dep_addr3 = cfg.alt_barriers ? sb15.src : sb2.src;
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::copy_load(int store_buffer, bool use_c) {
    // Load new A and B and increment load pointers
    if (!cfg.emulate64) {
        sync(SyncFunction::nop,
                SWSB<uint64_t>(1)); // SWSB doesn't cover $.src + RegDist
        mov(1 | dep_addr0, addr0.uq(0), a_ptr_mem);
        mov(1 | dep_addr1, addr1.uq(0), b_ptr_mem);
        add(1 | dep_addr2, addr2.uq(0), b_ptr_mem, uint16_t(8 * 32));
    } else {
        sync(SyncFunction::nop, SWSB<uint32_t>(1));
        mov(1 | dep_addr2, addr2.ud(1), b_ptr_mem.ud(1));
        add(1 | ov | f1[1], addr2.ud(0), b_ptr_mem.ud(0), uint16_t(8 * 32));
        mov(2 | dep_addr0, addr0.ud(0)(1), a_ptr_mem.ud()(1));
        mov(2 | dep_addr1, addr1.ud(0)(1), b_ptr_mem.ud()(1));
        add(1 | f1[1] | SWSB(4), addr2.ud(1), addr2.ud(1), 1);
    }

    if (use_c) {
        load(16 | SWSB(sb11, 3), c_regs[0], block_hword(8), A64, addr0);
        load(16 | SWSB(sb12, 2), c_regs[8], block_hword(8), A64, addr1);
        if (cfg.tile_n > 32)
            load(16 | SWSB(sb13, 1), c_regs[16], block_hword(4), A64, addr2);
        dep_addr0 = sb11.src;
        dep_addr1 = sb12.src;
        if (cfg.tile_n > 32) dep_addr2 = sb13.src;
        if (cfg.fulsim)
            sync.allrd(0x3000); // Unnecessary syncs to pacify Fulsim.
    } else {
        // Stronger than necessary dependencies... can load as soon as prev. store inputs are read.
        int load_buffer = cfg.global_3x_buf ? store_buffer : 0;
        int t0 = 8 + load_buffer * 2;
        load(16 | SWSB(SBID(t0), 3), a_copy[load_buffer][0], block_hword(8),
                A64, addr0);
        load(16 | SWSB(SBID(t0 + 1), 2), b_copy[load_buffer][0], block_hword(8),
                A64, addr1);
        if (cfg.tile_n > 32)
            load(16 | SWSB(SBID(t0 + 2), 1), b_copy[load_buffer][8],
                    block_hword(4), A64, addr2);
        dep_addr0 = SBID(t0).src;
        dep_addr1 = SBID(t0 + 1).src;
        if (cfg.tile_n > 32) dep_addr2 = SBID(t0 + 2).src;
        if (cfg.fulsim) sync.allrd(0x6 << t0);
    }

    if (!cfg.emulate64) {
        add(1 | SWSB(3), a_ptr_mem, a_ptr_mem, uint16_t(cfg.tile_m * 32));
        add(1 | SWSB(3), b_ptr_mem, b_ptr_mem, uint16_t(cfg.tile_n * 32));
    } else {
        add(1 | ov | f1[0] | SWSB(3), a_ptr_mem.ud(0), a_ptr_mem.ud(0),
                uint16_t(cfg.tile_m * 32));
        add(1 | ov | f1[1] | SWSB(3), b_ptr_mem.ud(0), b_ptr_mem.ud(0),
                uint16_t(cfg.tile_n * 32));
        add(1 | f1[0], a_ptr_mem.ud(1), a_ptr_mem.ud(1), 1);
        add(1 | f1[1], b_ptr_mem.ud(1), b_ptr_mem.ud(1), 1);
    }
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::copy_store(int store_buffer, bool first) {
    auto aoffset = first ? slm_a_offset_store_init : slm_a_offset_store;
    auto boffset = first ? slm_b_offset_store_init : slm_b_offset_store;

    // Store A and B and advance store pointers to next buffer.
    mov(1 | dep_addr0, addr0.ud(2), aoffset);
    mov(1 | dep_addr1, addr1.ud(2), boffset);
    add(1 | dep_addr2, addr2.ud(2), boffset, uint16_t(8 * 32 / 16));

    if (first && !cfg.global_3x_buf) {
        store(16 | SWSB(sb11, 3), block_oword(16), SLM, addr0, c_regs[0]);
        store(16 | SWSB(sb12, 2), block_oword(16), SLM, addr1, c_regs[8]);
        if (cfg.tile_n > 32)
            store(16 | SWSB(sb13, 1), block_oword(8), SLM, addr2, c_regs[16]);
        dep_addr0 = sb11.src;
        dep_addr1 = sb12.src;
        if (cfg.tile_n > 32) dep_addr2 = sb13.src;
        if (cfg.fulsim) sync.allrd(0x3000);
    } else {
        int load_buffer = cfg.global_3x_buf ? store_buffer : 0;
        int t0 = 8 + load_buffer * 2;
        store(16 | SWSB(SBID(t0), 3), block_oword(16), SLM, addr0,
                a_copy[load_buffer][0]);
        store(16 | SWSB(SBID(t0 + 1), 2), block_oword(16), SLM, addr1,
                b_copy[load_buffer][0]);
        if (cfg.tile_n > 32)
            store(16 | SWSB(SBID(t0 + 2), 1), block_oword(8), SLM, addr2,
                    b_copy[load_buffer][8]);
        dep_addr0 = SBID(t0).src;
        dep_addr1 = SBID(t0 + 1).src;
        if (cfg.tile_n > 32) dep_addr2 = SBID(t0 + 2).src;
        if (cfg.fulsim) sync.allrd(0x6 << t0);
    }

    if (store_buffer == 2)
        mov(2, slm_a_offset_store(1), slm_a_offset_store_init(1));
    else
        add(2, slm_a_offset_store(1), aoffset(1),
                uint16_t(slm_buf_size() / 16));
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::store_signal(bool force_fence) {
    if (cfg.use_slm_fence || force_fence) {
        // Signal SLM data ready once memory fence returns, asynchronously
        sync(SyncFunction::nop, dep_addr0);
        barrier_prep(dep_addr3, addr3);

        slmfence(SWSB(sb15, 1), addr0);
        barriermsg(SWSB(sb15), addr3);
        dep_addr0 = dep_addr3 = sb15.src;
    } else {
        barrier_prep(dep_addr3, addr3);
        barriermsg(SWSB(sb15, 1), addr3);
        dep_addr3 = sb15.src;
    }
}

template <HW hw>
void xehp_systolic_gemm_kernel_t<hw>::body() {
    Label top, bottom, skip_main, rem_top, rem_bottom;

    if (!cfg.global_3x_buf) {
        cmp(1 | lt | f1[1], k_counter, 3);
        add(1 | le | f0[1], k_counter, k_counter, -5);

        jmpi(1 | f1[1], skip_main);

        copy_load(0, true); // L0 -> C
        copy_load(1); // L1
        copy_store(0, true); // S0 <- C
        store_signal(true); // Signal 0 ready
        zero_c();
        sync.nop(SWSB<AllPipes>(1));
        copy_store(1); // S1

        if (!cfg.alt_barriers) {
            barrierwait(); // Wait 0 ready
            store_signal(); // Signal 1 ready
        }

        jmpi(1 | f0[1], bottom); // Zero-trip loop check

        mark(top);
        add(1 | gt | f0[1], k_counter, k_counter, -3);

        copy_load(2); // L2
        multiply(0); // M0
        if (!cfg.alt_barriers) barrierwait(); // Wait 0 ready
        copy_store(2); // S2
        if (!cfg.alt_barriers) store_signal(); // Signal 2 ready

        copy_load(0); // L0
        multiply(1); // M1
        if (!cfg.alt_barriers) barrierwait(); // Wait 2 ready
        copy_store(0); // S0
        if (!cfg.alt_barriers) store_signal(); // Signal 0 ready

        copy_load(1); // L1
        multiply(2); // M2
        if (!cfg.alt_barriers) barrierwait(); // Wait 0 ready
        copy_store(1); // S1
        if (!cfg.alt_barriers) store_signal(); // Signal 1 ready

        jmpi(1 | f0[1], top);
        mark(bottom);

        copy_load(2); // L2
        multiply(0); // M0
        if (!cfg.alt_barriers) barrierwait(); // Wait 1 ready
        copy_store(2); // S2
        if (!cfg.alt_barriers) store_signal(); // Signal 2 ready

        multiply(1); // M1

        if (!cfg.alt_barriers) barrierwait(); // Wait 2 ready

        multiply(2, true); // M2

        add(1 | le | f0[1], k_counter, k_counter, 2);
        jmpi(1 | f0[1], rem_bottom);
        jmpi(1, rem_top);

        mark(skip_main);

        zero_c();
        add(1, k_counter, k_counter, 5);

        mov(2, slm_a_offset_store(1), slm_a_offset_store_init(1));
        sync.nop(SWSB<AllPipes>(1));

        mark(rem_top);

        cmp(1 | lt | f0[1], k_counter, 2);
        copy_load(0);
        copy_store(0);
        store_signal(true);
        if (!cfg.alt_barriers) barrierwait();
        multiply(0, true);

        jmpi(1 | f0[1], rem_bottom);
        copy_load(1);
        copy_store(1);
        store_signal(true);
        if (!cfg.alt_barriers) barrierwait();
        multiply(1, true);

        mark(rem_bottom);
    } else {
        // Triple-buffered global memory load + SLM pipeline.
        cmp(1 | lt | f1[1], k_counter, 4);
        add(1 | le | f0[1], k_counter, k_counter, -6);

        jmpi(1 | f1[1], skip_main);

        copy_load(0); // L0
        copy_load(1); // L1
        copy_load(2); // L2
        copy_store(0, true); // S0
        store_signal(true); // Signal 0 ready
        zero_c();
        copy_load(0); // L0
        sync.nop(SWSB<uint32_t>(1));
        copy_store(1); // S1

        if (!cfg.alt_barriers) {
            barrierwait(); // Wait 0 ready
            store_signal(); // Signal 1 ready
        }

        jmpi(1 | f0[1], bottom); // Zero-trip loop check

        mark(top);
        add(1 | gt | f0[1], k_counter, k_counter, -3);

        copy_load(1); // L1
        multiply(0); // M0
        if (!cfg.alt_barriers) barrierwait(); // Wait 0 ready
        copy_store(2); // S2
        if (!cfg.alt_barriers) store_signal(); // Signal 2 ready

        copy_load(2); // L2
        multiply(1); // M1
        if (!cfg.alt_barriers) barrierwait(); // Wait 2 ready
        copy_store(0); // S0
        if (!cfg.alt_barriers) store_signal(); // Signal 0 ready

        copy_load(0); // L0
        multiply(2); // M2
        if (!cfg.alt_barriers) barrierwait(); // Wait 0 ready
        copy_store(1); // S1
        if (!cfg.alt_barriers) store_signal(); // Signal 1 ready

        jmpi(1 | f0[1], top);
        mark(bottom);

        multiply(0); // M0
        if (!cfg.alt_barriers) barrierwait(); // Wait 1 ready
        copy_store(2); // S2
        if (!cfg.alt_barriers) store_signal(); // Signal 2 ready

        multiply(1); // M1
        if (!cfg.alt_barriers) barrierwait(); // Wait 2 ready
        copy_store(0); // S0
        if (!cfg.alt_barriers) store_signal(); // Signal 0 ready

        multiply(2); // M2

        if (!cfg.alt_barriers) barrierwait(); // Wait 0 ready

        multiply(0, true); // M0

        add(1 | le | f0[1], k_counter, k_counter, 2);
        jmpi(1 | f0[1], rem_bottom);
        jmpi(1, rem_top);

        mark(skip_main);

        zero_c();
        add(1 | le | f0[1], k_counter, k_counter, 5);

        mov(2, slm_a_offset_store(1), slm_a_offset_store_init(1));
        sync.nop(SWSB<uint32_t>(1));

        copy_load(0);
        copy_store(0);
        store_signal(true);
        if (!cfg.alt_barriers) barrierwait();
        multiply(0, true);

        jmpi(1 | f0[1], rem_bottom);

        mark(rem_top);

        cmp(1 | lt | f0[1], k_counter, 2);

        copy_load(1);
        copy_store(1);
        store_signal(true);
        if (!cfg.alt_barriers) barrierwait();
        multiply(1, true);

        jmpi(1 | f0[1], rem_bottom);

        copy_load(2);
        copy_store(2);
        store_signal(true);
        if (!cfg.alt_barriers) barrierwait();
        multiply(2, true);

        mark(rem_bottom);
    }

    sync(SyncFunction::allwr); // Wait for systolic ops to finish

    update_c();
}

template <HW hw>
xehp_systolic_gemm_kernel_t<hw>::xehp_systolic_gemm_kernel_t(config_t cfg_)
    : cfg(cfg_) {
    if (!cfg.valid()) assert(!"Invalid configuration");

    if (cfg.have_post_op()) {
        auto inj_ptr = new injector_t(this, data_type::f32, cfg.post_ops,
                upost_op_scratch, cfg.post_op_is_fwd);
        assert(inj_ptr);
        post_op_injector.reset(inj_ptr);
    }

    setDefaultNoMask();

    // Signature:
    //   kernel void gemm_kernel(global char *ap, global uchar *bp, global int *C,
    //                           int k, int ldc,
    //                           long offset_a, long offset_b, long offset_c,
    //                           int m, int n,
    //                           float alpha, float beta,
    //                           int lda, int ldb [, uint abo]
    //                           [, int *co, int offset_co [, uint flags]]
    //                           [, int stride_a, int stride_b, int stride_c);

    externalName("xehp_systolic_gemm_kernel");
    newArgument("ap", ExternalArgumentType::GlobalPtr);
    newArgument("bp", ExternalArgumentType::GlobalPtr);
    newArgument("c", ExternalArgumentType::GlobalPtr);
    newArgument("offset_a", DataType::q);
    newArgument("offset_b", DataType::q);
    newArgument("offset_c", DataType::q);
    newArgument("lda", DataType::d);
    newArgument("ldb", DataType::d);
    newArgument("ldc", DataType::d);
    newArgument("m", DataType::d);
    newArgument("n", DataType::d);
    newArgument("k", DataType::d);
    newArgument("alpha", cfg.scale_type);
    newArgument("beta", cfg.scale_type);
    if (cfg.a_bias || cfg.b_bias) newArgument("abo", DataType::ud);
    if (cfg.c_bias != bias_t::none) {
        newArgument("co", ExternalArgumentType::GlobalPtr);
        newArgument("offset_co", DataType::d);
        if (cfg.c_bias == bias_t::runtime) newArgument("flags", DataType::ud);
    }
    if (cfg.batch) {
        newArgument("stride_a", DataType::d);
        newArgument("stride_b", DataType::d);
        newArgument("stride_c", DataType::d);
    }
    requireBarrier();
    requireDPAS();
    requireGRF(256);
    requireLocalID(2);
    requireSIMD(8);
    requireSLM(slm_buf_size() * 3);
    finalizeInterface();

    // Inputs.
    auto global_id_x = r0.ud(1);
    auto global_id_y = r0.ud(6);
    auto global_id_z = r0.ud(7);
    auto local_id_x = r1.uw(0);
    auto local_id_y = r2.uw(0);
    auto ap = getArgument("ap");
    auto bp = getArgument("bp");
    auto c_ptr = getArgument("c");
    auto in_offset_a = getArgument("offset_a");
    auto in_offset_b = getArgument("offset_b");
    auto in_offset_c = getArgument("offset_c");
    auto k = getArgument("k");
    auto ldc = getArgument("ldc");
    auto m = getArgument("m");
    auto n = getArgument("n");
    auto alpha = getArgument("alpha");
    auto beta = getArgument("beta");
    auto lda = getArgument("lda");
    auto ldb = getArgument("ldb");
    auto abo = getArgumentIfExists("abo");
    auto in_offset_co = getArgumentIfExists("offset_co");
    auto stride_a = getArgumentIfExists("stride_a");
    auto stride_b = getArgumentIfExists("stride_b");
    auto stride_c = getArgumentIfExists("stride_c");

    ap_surface = getArgumentSurface("ap");
    bp_surface = getArgumentSurface("bp");
    if (cfg.c_bias != bias_t::none) co_surface = getArgumentSurface("co");
    auto flags = getArgumentIfExists("flags");

    // Temporaries
    auto n0 = r10.ud(0);
    auto m0 = r10.ud(1);
    auto offset_a = r12.uq(0);
    auto offset_b = r12.uq(1);
    auto offset_c = r12.uq(2);
    // reserved: r12.uq(3)
    auto boffset_a = r13.q(0);
    auto boffset_b = r13.q(1);
    auto boffset_c = r13.q(2);
    auto offset_asum = r14.ud(0);
    auto offset_bsum = r14.ud(2);
    auto global_n0 = r18.ud(0);
    auto global_m0 = r18.ud(1);
    auto local_n0 = r20.ud(0);
    auto local_m0 = r20.ud(1);
    auto suboffset_a = r26.ud(0);
    auto suboffset_b = r26.ud(1);
    auto thd1_adjust = r27.ud(0);
    auto temp = r28.ud(0);
    auto temp2 = r28.ud(1);
    auto save_copy = r32.ud();
    auto k_counter_copy = r32.ud(0);
    auto ldc_copy = r32.ud(1);
    auto off_co_copy = r32.ud(2);
    auto k_copy = r32.ud(3);
    auto mrem_copy = r32.uw(8);
    auto nrem_copy = r32.uw(9);
    auto abo_copy = r32.ud(5);
    auto alpha_copy = r32.f(6);
    auto beta_copy = r32.f(7);

    if (cfg.emulate64) {
        emu_strategy.emulate64 = true;
        emu_state.temp[0] = r22;
        emu_state.temp[1] = r23;
        emu_state.flag = f1[0];
    }

    setDefaultAutoSWSB(true);
    prologue();

    // Enable IEEE f32->s32 rounding and f32/f16 denorms.
    or_(1, cr0, cr0, uint16_t(0x1480));

    // Find our threadgroup's position within the matrix.
    if (cfg.walk_n_first) std::swap(global_id_x, global_id_y);
    mul_constant(1, global_m0, global_id_x, 4 * cfg.tile_m);
    mul_constant(1, global_n0, global_id_y, 4 * cfg.tile_n);

    // Adjust barrier ID field on DG2.
    if (hw == HW::XeHPG) mov<uint8_t>(1, r0[10], r0[11]);

    // Find our position within the threadgroup. Fixed threadgroup size: 4x4.
    mul_constant(1, local_m0, local_id_x, cfg.tile_m / 8);
    mul_constant(1, local_n0, local_id_y, cfg.tile_n);
    add(2, n0(1), local_n0(1), global_n0(1));

    // Batch handling.
    if (cfg.batch) {
        emul(1, boffset_a, stride_a, global_id_z);
        emul(1, boffset_b, stride_b, global_id_z);
        emul(1, boffset_c, stride_c, global_id_z);
        eadd(1, in_offset_a, in_offset_a, boffset_a);
        eadd(1, in_offset_b, in_offset_b, boffset_b);
        eadd(1, in_offset_c, in_offset_c, boffset_c);
    }

    // Compute starting addresses:
    //   - suboffset_a = local_id_Y * (tile_m / 4) * 32
    //   - suboffset_b = local_id_X/8 * (tile_n / 4) * 32
    //   - slm_a_offset_load_init = local_m0 * 32 [36 with padding]
    //   - slm_b_offset_load_init = 128 * 32 [36 w/ padding] + local_n0 * 32 + ((tile_n / 2) * 32 if fused)
    //   - slm_a_offset_store_init = slm_a_offset_load_init + suboffset_a
    //   - slm_b_offset_store_init = slm_b_offset_load_init + suboffset_b
    //   - Ap += m0 * lda + suboffset_a
    //   - Bp += n0 * ldb + suboffset_b
    //   - C += m0 + n0 * ldc [save for later]
    uint16_t lg2_a_elem_bytes = ngen::utils::log2(getBytes(cfg.a_type));
    uint16_t lg2_c_elem_bytes = ngen::utils::log2(getBytes(cfg.c_type));
    uint16_t lg2_co_elem_bytes = ngen::utils::log2(getBytes(cfg.co_type));

    mov(1, k_copy, k);
    add(1, k_counter_copy, k, this_unroll_k() - 1);
    shl(1, ldc_copy, ldc, lg2_c_elem_bytes);
    mul_constant(1, suboffset_a, local_id_y, cfg.tile_m * (32 / 4));
    mul_constant(1, suboffset_b, local_id_x, cfg.tile_n * (32 / 4) / 8);
    if (cfg.c_packed) mul(1, temp2, m0, cfg.tile_n);
    shr(1, k_counter_copy, k_counter_copy, ngen::utils::log2(this_unroll_k()));
    emul(1, offset_c, n0, ldc);
    emul(1, offset_a, m0, lda);
    emul(1, offset_b, n0, ldb);
    switch (cfg.c_bias) {
        case bias_t::none: break;
        case bias_t::fixed: mov(1, off_co_copy, in_offset_co); break;
        case bias_t::row: add(1, off_co_copy, in_offset_co, m0); break;
        case bias_t::column: add(1, off_co_copy, in_offset_co, n0); break;
        case bias_t::runtime: {
            mov(1, off_co_copy, in_offset_co);
            and_(1 | nz | f0[0], null.ud(), flags, flag_c_bias_col);
            and_(1 | nz | f1[0], null.ud(), flags, flag_c_bias_row);
            add(1 | f0[0], off_co_copy, off_co_copy, n0);
            add(1 | f1[0], off_co_copy, off_co_copy, m0);
            break;
        }
    }
    if (cfg.c_packed)
        eadd(1, offset_c, offset_c, temp2);
    else
        eadd(1, offset_c, offset_c, m0);
    eadd(1, offset_a, offset_a, in_offset_a); // TODO: combine
    eadd(1, offset_b, offset_b, in_offset_b);
    eadd(1, offset_c, offset_c, in_offset_c);
    if (getBytes(cfg.a_type) > 1)
        eshl(2, offset_a(1), offset_a(1), lg2_a_elem_bytes); // A, B
    if (cfg.a_bias)
        mad(1, offset_asum, offset_a.ud(), k_counter_copy,
                cfg.tile_m * this_unroll_k());
    if (cfg.b_bias)
        mad(1, offset_bsum, offset_b.ud(), k_counter_copy,
                cfg.tile_n * this_unroll_k());
    eadd(1, offset_a, offset_a, suboffset_a);
    eadd(1, offset_b, offset_b, suboffset_b);
    eshl(1, offset_c, offset_c, lg2_c_elem_bytes);
    eadd(1, a_ptr_mem, ap, offset_a);
    eadd(1, b_ptr_mem, bp, offset_b);
    eadd(1, c_ptr_mem, c_ptr, offset_c);
    if (cfg.c_bias != bias_t::none)
        shl(1, off_co_copy, off_co_copy, uint16_t(lg2_co_elem_bytes));

    and_(1, temp, local_id_x, uint16_t(8));
    shr(2, suboffset_a(1), suboffset_a(1), uint16_t(4));

    if (cfg.pad_a) {
        shl(1, local_n0, local_n0, uint16_t(5 - 4));
        mul(1, local_m0, local_id_x, uint16_t(9));
        mul(1, thd1_adjust, temp, uint16_t((cfg.tile_n / 2) * 32 / (8 * 16)));
        add(1, local_n0, local_n0, uint16_t((128 * 36) / 16));
    } else {
        shl(2, local_n0(1), local_n0(1), uint32_t(5 - 4));
        mul(1, thd1_adjust, temp, uint16_t((cfg.tile_n / 2) * 32 / (8 * 16)));
        add(1, local_n0, local_n0, uint16_t(128 * 32 / 16));
    }

    mov(1, slm_a_offset_load_init.uw(), local_m0.uw());
    add(1, slm_b_offset_load_init.uw(), local_n0.uw(), thd1_adjust.uw());
    add(1, slm_a_offset_store_init.uw(), local_m0.uw(), suboffset_a.uw());
    add(1, slm_b_offset_store_init.uw(), local_n0.uw(), suboffset_b.uw());
    mov(2, slm_a_offset_load(1), slm_a_offset_load_init(1));

    // Compute m, n remainders and save variables for C update.
    // Also compute threshold for m remainder: 2*tile_m for thread 0 of fused pair,
    //  tile_m for thread 1.
    if (cfg.c_remainder) {
        mul_constant(1, temp, temp,
                cfg.tile_m / 8); // temp = local id X (unshifted) & 8
        add(1 | sat, mrem_copy, m, -m0);
        add(1 | sat, nrem_copy, n, -n0);
        add(1, temp, -temp, 2 * cfg.tile_m);
    }

    if (abo.isValid()) mov(1, abo_copy.f(), abo.f());
    mov(1, alpha_copy, alpha);
    mov(1, beta_copy, beta);

    sync(SyncFunction::nop, SWSB<AllPipes>(1));

    mov<uint32_t>(8, r0_save, r0);

    if (cfg.c_remainder) mov(1, f0[0], 0);
    mov(8, base_save, save_copy);

    if (cfg.emulate64) {
        emu_state.temp[0] = invalid;
        emu_state.temp[1] = invalid;
        emu_state.flag = invalid;
    }

    if (cfg.a_bias) mov(1, off_asum_save, offset_asum);
    if (cfg.b_bias) mov(1, off_bsum_save, offset_bsum);
    if (cfg.c_bias == bias_t::runtime) mov(1, flags_save, flags);

    // Check whether to use remainder path, and save in f0.0 for later.
    if (cfg.c_remainder) {
        cmp(1 | lt | f0[0], mrem_copy, temp);
        cmp(1 | lt | f0[0] | M8, nrem_copy, uint32_t(cfg.tile_n));
    }

    setDefaultAutoSWSB(false);

    // Main body.
    body();

    // Epilogue.
    epilogue(r0_save);

    // Kernel padding for instruction prefetch.
    for (int rep = 0; rep < 8; rep++)
        nop();
}

template class xehp_systolic_gemm_kernel_t<HW::XeHP>;
template class xehp_systolic_gemm_kernel_t<HW::XeHPG>;

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
