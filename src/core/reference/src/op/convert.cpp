// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/convert.hpp"

#include "openvino/reference/utils/convert_util.hpp"

#ifdef OV_CORE_USE_XBYAK_JIT
#    include "openvino/reference/utils/jit_generator.hpp"
#    include "openvino/util/os.hpp"
#endif

#ifdef OV_CORE_USE_INTRINSICS
#    include "openvino/reference/utils/convert_x86_intrinsics.hpp"
#endif

namespace ov {
namespace reference {

namespace {
#ifdef OV_CORE_USE_XBYAK_JIT
// vcvtps2ph immediate: force round-to-nearest-even and suppress all FP
// exceptions, so the rounding is independent of caller MXCSR state and
// bit-identical to static_cast<ov::float16>(float). Equivalent to
// _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC from <immintrin.h>.
// (Not 0x04 — that is _MM_FROUND_CUR_DIRECTION, i.e. "use MXCSR".)
inline constexpr uint8_t kVcvtps2phRneNoExc = 0x08;

// Shared FP16 range constants used by multiple JIT kernels (fp32<->fp16 conversion,
// fp16 compression check). ov::float16::operator float() is not constexpr so the
// ov::float16-derived values must be initialised at runtime; the scalar-error /
// mask thresholds are plain literals and stay constexpr. We keep them as named
// file-scope constants to avoid duplicating std::numeric_limits<> lookups and
// literal values across every JIT body.
inline const float kF16MaxPos = std::numeric_limits<ov::float16>::max();
inline const float kF16MaxNeg = std::numeric_limits<ov::float16>::lowest();
inline const float kF16MinPos = ov::float16::from_bits(0x0001);
inline const float kF16MinNeg = -ov::float16::from_bits(0x0001);
inline constexpr float kF16CompressionAbsErrVal = static_cast<float>(ov::reference::f16_compression_max_abs_error);
inline constexpr float kF16CompressionRelErrVal = static_cast<float>(ov::reference::f16_compression_max_rel_error);
inline constexpr uint32_t kAbsMaskVal = 0x7FFFFFFFu;

template <typename src_t, typename dst_t, bool clamp = false>
void jit_convert_vec(jit::Generator&, const Xbyak::RegExp&, const Xbyak::RegExp&) {}

template <typename src_t, typename dst_t, bool clamp = false>
void jit_convert_vec_prepare(jit::Generator&) {}

template <>
void jit_convert_vec<uint8_t, float16>(jit::Generator& gen, const Xbyak::RegExp& src, const Xbyak::RegExp& dst) {
    auto u8vec = gen.xmm1;
    auto i32vec = gen.ymm2;
    auto f16vec = gen.xmm3;
    auto fvec = gen.ymm4;

    gen.movq(u8vec, gen.qword[src]);
    gen.vpmovzxbd(i32vec, u8vec);
    gen.vcvtdq2ps(fvec, i32vec);
    gen.vcvtps2ph(f16vec, fvec, kVcvtps2phRneNoExc);
    gen.vzeroupper();
    gen.vmovdqu(gen.xword[dst], f16vec);
}

template <>
void jit_convert_vec<float16, float>(jit::Generator& gen, const Xbyak::RegExp& src, const Xbyak::RegExp& dst) {
    auto f16vec = gen.xmm3;
    auto f32vec = gen.ymm4;

    gen.vmovdqu(f16vec, gen.xword[src]);
    gen.vcvtph2ps(f32vec, f16vec);
    gen.vmovups(gen.yword[dst], f32vec);
}

template <>
void jit_convert_vec<float, float16>(jit::Generator& gen, const Xbyak::RegExp& src, const Xbyak::RegExp& dst) {
    auto f16vec = gen.xmm3;
    auto f32vec = gen.ymm4;

    gen.vmovups(f32vec, gen.yword[src]);
    gen.vcvtps2ph(f16vec, f32vec, kVcvtps2phRneNoExc);
    gen.vmovdqu(gen.xword[dst], f16vec);
}

template <>
void jit_convert_vec<bfloat16, float16>(jit::Generator& gen, const Xbyak::RegExp& src, const Xbyak::RegExp& dst) {
    const auto f32vec = gen.ymm4;
    const auto f16vec = gen.xmm3;

    gen.vpmovzxwd(f32vec, gen.yword[src]);              // load bf16 into tmp
    gen.vpslld(f32vec, f32vec, 16);                     // convert bf16->f32 by bit shift
    gen.vcvtps2ph(f16vec, f32vec, kVcvtps2phRneNoExc);  // convert f32 -> f16
    gen.vmovdqu(gen.xword[dst], f16vec);                // move result to destination
}

template <>
void jit_convert_vec<bfloat16, float16, true>(jit::Generator& gen, const Xbyak::RegExp& src, const Xbyak::RegExp& dst) {
    const auto f32vec = gen.ymm4;
    const auto f16vec = gen.xmm3;

    auto upper_bound = gen.ymm5;
    auto lower_bound = gen.ymm6;

    gen.vpmovzxwd(f32vec, gen.yword[src]);              // load bf16 into tmp
    gen.vpslld(f32vec, f32vec, 16);                     // convert bf16->f32 by bit shift
    gen.vminps(f32vec, f32vec, upper_bound);            // clamp f16 max
    gen.vmaxps(f32vec, f32vec, lower_bound);            // clamp f16 lowest
    gen.vcvtps2ph(f16vec, f32vec, kVcvtps2phRneNoExc);  // convert f32 -> f16
    gen.vmovdqu(gen.xword[dst], f16vec);                // move result to destination
}

template <>
void jit_convert_vec<bfloat16, float>(jit::Generator& gen, const Xbyak::RegExp& src, const Xbyak::RegExp& dst) {
    const auto f32vec = gen.ymm4;

    gen.vpmovzxwd(f32vec, gen.yword[src]);  // load bf16 into tmp
    gen.vpslld(f32vec, f32vec, 16);         // convert bf16->f32 by bit shift
    gen.vmovdqu(gen.yword[dst], f32vec);    // move result to destination
}

template <>
void jit_convert_vec_prepare<float, float16, true>(jit::Generator& gen) {
    auto upper_bound = gen.ymm5;
    auto lower_bound = gen.ymm6;
    auto addr = gen.r15;

    static const float upper_bounds[8] =
        {kF16MaxPos, kF16MaxPos, kF16MaxPos, kF16MaxPos, kF16MaxPos, kF16MaxPos, kF16MaxPos, kF16MaxPos};
    static const float lower_bounds[8] =
        {kF16MaxNeg, kF16MaxNeg, kF16MaxNeg, kF16MaxNeg, kF16MaxNeg, kF16MaxNeg, kF16MaxNeg, kF16MaxNeg};

    gen.mov(addr, reinterpret_cast<size_t>(upper_bounds));
    gen.vmovdqu(upper_bound, gen.yword[addr]);
    gen.mov(addr, reinterpret_cast<size_t>(lower_bounds));
    gen.vmovdqu(lower_bound, gen.yword[addr]);
}

template <>
void jit_convert_vec_prepare<bfloat16, float16, true>(jit::Generator& gen) {
    jit_convert_vec_prepare<float, float16, true>(gen);
}

template <>
void jit_convert_vec<float, float16, true>(jit::Generator& gen, const Xbyak::RegExp& src, const Xbyak::RegExp& dst) {
    auto f16vec = gen.xmm3;
    auto f32vec = gen.ymm4;
    auto upper_bound = gen.ymm5;
    auto lower_bound = gen.ymm6;

    gen.vmovups(f32vec, gen.yword[src]);
    gen.vminps(f32vec, f32vec, upper_bound);
    gen.vmaxps(f32vec, f32vec, lower_bound);
    gen.vcvtps2ph(f16vec, f32vec, kVcvtps2phRneNoExc);
    gen.vmovdqu(gen.xword[dst], f16vec);
}

template <>
void jit_convert_vec_prepare<float, int8_t>(jit::Generator& gen) {
    auto order = gen.ymm1;
    auto addr = gen.r15;

    static constexpr int8_t offsets[32] = {0,  4,  8,  12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                           -1, -1, -1, -1, 0,  4,  8,  12, -1, -1, -1, -1, -1, -1, -1, -1};

    gen.mov(addr, reinterpret_cast<size_t>(offsets));  // get offsets[] address
    gen.vmovdqu(order, gen.yword[addr]);               // save offsets[] to ymm register
}

template <>
void jit_convert_vec<float, int8_t>(jit::Generator& gen, const Xbyak::RegExp& src, const Xbyak::RegExp& dst) {
    auto order = gen.ymm1;
    auto p32vec = gen.ymm2;
    auto p32vec_lo = gen.xmm2;
    auto p32vec_hi = gen.xmm3;

    gen.vcvttps2dq(p32vec, gen.yword[src]);     // convert 8 floats to 8 ints
    gen.vpshufb(p32vec, p32vec, order);         // Shuffle the bytes according to the order
    gen.vextracti128(p32vec_hi, p32vec, 1);     // extract upper part of p32vec
    gen.vpor(p32vec_lo, p32vec_lo, p32vec_hi);  // p32vec_lo = p32vec_lo | p32vec_hi
    gen.movq(gen.qword[dst], p32vec_lo);        // save the result
}

template <>
void jit_convert_vec_prepare<float16, int8_t>(jit::Generator& gen) {
    jit_convert_vec_prepare<float, int8_t>(gen);
}

template <>
void jit_convert_vec<float16, int8_t>(jit::Generator& gen, const Xbyak::RegExp& src, const Xbyak::RegExp& dst) {
    auto order = gen.ymm1;
    auto p32vec = gen.ymm2;
    auto p32vec_lo = gen.xmm2;
    auto p32vec_hi = gen.xmm3;

    gen.vcvtph2ps(p32vec, gen.xword[src]);      // convert 8 fp16's to 8 floats
    gen.vcvttps2dq(p32vec, p32vec);             // convert 8 floats to 8 ints
    gen.vpshufb(p32vec, p32vec, order);         // Shuffle the bytes according to the order
    gen.vextracti128(p32vec_hi, p32vec, 1);     // extract upper part of p32vec
    gen.vpor(p32vec_lo, p32vec_lo, p32vec_hi);  // p32vec_lo = p32vec_lo | p32vec_hi
    gen.movq(gen.qword[dst], p32vec_lo);        // save the result
}

class jit_convert_array : public jit::Generator {
    typedef struct context {
        struct {
            size_t type_size;
            void (jit::Generator::*copy)(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size);
        } src, dst;
        void (*convert_vec)(jit::Generator&, const Xbyak::RegExp&, const Xbyak::RegExp&);
        void (*prepare)(jit::Generator&);
    } context_t;

    jit_convert_array(const context_t& ctx) {
        using namespace Xbyak;

        const uint32_t vlen = 8u;

        auto reg_src = rax;
        auto reg_dst = rbx;
        auto reg_sz = rdx;

        Label tail, exit;

        preamble();

        ctx.prepare(*this);

        mov(reg_src, ptr[param + offsetof(args_t, src)]);
        mov(reg_dst, ptr[param + offsetof(args_t, out)]);
        mov(reg_sz, ptr[param + offsetof(args_t, count)]);

        xor_(rsi, rsi);
        mov(r8, reg_sz);
        shr(r8, 3);

        foreach (rsi, 1, r8, [&, this](const Xbyak::Reg64& idx) {
            ctx.convert_vec(*this, reg_src, reg_dst);
            add(reg_src, static_cast<uint32_t>(ctx.src.type_size * vlen));
            add(reg_dst, static_cast<uint32_t>(ctx.dst.type_size * vlen));
        })
            ;

        L(tail);

        shl(rsi, 3);
        sub(reg_sz, rsi);
        test(reg_sz, reg_sz);
        jz(exit);

        // allocate array for 8 floats on stack
        sub(rsp, vlen * sizeof(float));
        mov(r8, rsp);

        vpxor(ymm4, ymm4, ymm4);
        vmovups(yword[r8], ymm4);

        // Tail conversion
        (this->*ctx.src.copy)(r8, reg_src, reg_sz);
        ctx.convert_vec(*this, r8, r8);
        (this->*ctx.dst.copy)(reg_dst, r8, reg_sz);

        // Free the array on stack
        add(rsp, vlen * sizeof(float));

        L(exit);

        postamble();
    }

public:
    typedef struct {
        const void* src;
        void* out;
        const size_t count;
    } args_t;

    typedef void (*fn_t)(const args_t*);

    template <typename src_t, typename dst_t, bool clamp = false>
    static fn_t get() {
        if (is_x64() && mayiuse(jit::avx) && mayiuse(jit::avx2) && mayiuse(jit::fp16)) {
            static const jit_convert_array::context_t context{{sizeof(src_t), &jit::Generator::copy<src_t>},
                                                              {sizeof(dst_t), &jit::Generator::copy<dst_t>},
                                                              jit_convert_vec<src_t, dst_t, clamp>,
                                                              jit_convert_vec_prepare<src_t, dst_t, clamp>};

            static jit_convert_array generator(context);

            // Since the JIT code always resides in memory, and ASAN's memory management may remove executable
            // permissions, we need to restore executable permissions for the generated code.
            generator.setProtectModeRE(false);
            return (fn_t)generator.getCode();
        }
        return nullptr;
    }
};

template <typename data_t, typename range_t>
void jit_count_out_of_range_vec_prepare(jit::Generator&) {}

template <typename data_t, typename range_t>
void jit_count_out_of_range_vec(jit::Generator&, const Xbyak::RegExp&);

template <typename data_t, typename range_t>
void jit_count_out_of_range_vec_finalize(jit::Generator&, const Xbyak::RegExp&) {}

class jit_count_out_of_range : public jit::Generator {
    typedef struct context {
        struct {
            size_t type_size;
            void (jit::Generator::*copy)(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size);
        } data;
        void (*prepare)(jit::Generator&);
        void (*count_out_of_range)(jit::Generator&, const Xbyak::RegExp&);
        void (*finalize)(jit::Generator&, const Xbyak::RegExp& dst);
    } context_t;

    jit_count_out_of_range(const context_t& ctx) {
        using namespace Xbyak;

        const uint32_t vlen = 8u;

        auto reg_src = rax;
        auto reg_dst = rbx;
        auto reg_sz = rdx;

        Label tail, exit;

        preamble();

        ctx.prepare(*this);

        mov(reg_src, ptr[param + offsetof(args_t, src)]);
        mov(reg_dst, ptr[param + offsetof(args_t, dst)]);
        mov(reg_sz, ptr[param + offsetof(args_t, count)]);

        xor_(rsi, rsi);
        mov(r8, reg_sz);
        shr(r8, 3);

        foreach (rsi, 1, r8, [&, this](const Xbyak::Reg64& idx) {
            ctx.count_out_of_range(*this, reg_src);
            add(reg_src, static_cast<uint32_t>(ctx.data.type_size * vlen));
        })
            ;

        L(tail);

        shl(rsi, 3);
        sub(reg_sz, rsi);
        test(reg_sz, reg_sz);
        jz(exit);

        // allocate array for 8 floats on stack
        sub(rsp, vlen * sizeof(float));
        mov(r8, rsp);

        auto tmp_vec = ymm2;  // reuse mask_vec
        vpxor(tmp_vec, tmp_vec, tmp_vec);
        vmovups(yword[r8], tmp_vec);

        // Tail conversion
        (this->*ctx.data.copy)(r8, reg_src, reg_sz);
        ctx.count_out_of_range(*this, r8);

        // Free the array on stack
        add(rsp, vlen * sizeof(float));

        L(exit);

        ctx.finalize(*this, reg_dst);

        postamble();
    }

public:
    typedef struct {
        const void* src;
        void* dst;
        const size_t count;
    } args_t;

    typedef void (*fn_t)(const args_t*);

    template <typename data_t, typename range_t>
    static fn_t get() {
        if (is_x64() && mayiuse(jit::avx2)) {
            static const jit_count_out_of_range::context_t context{
                {sizeof(data_t), &jit::Generator::copy<data_t>},
                jit_count_out_of_range_vec_prepare<data_t, range_t>,
                jit_count_out_of_range_vec<data_t, range_t>,
                jit_count_out_of_range_vec_finalize<data_t, range_t>};

            static jit_count_out_of_range generator(context);

            // Since the JIT code always resides in memory, and ASAN's memory management may remove executable
            // permissions, we need to restore executable permissions for the generated code.
            generator.setProtectModeRE(false);
            return (fn_t)generator.getCode();
        }
        return nullptr;
    }
};

// Combined single-pass JIT kernel for AVX-512: counts out-of-range AND detects lossy f16 compression.
// Processes 16 floats per iteration using zmm + opmask + F16C (vcvtps2ph/vcvtph2ps). Bails immediately
// if any in-range element has significant precision loss (abs error > 1.0). Tail handled via masked load.
class jit_check_f16_compression_avx512 : public jit::Generator {
public:
    typedef struct {
        const void* src;
        void* oor_dst;    // size_t* — output: combined out-of-range + high-relative-error count
        void* lossy_dst;  // size_t* — output: lossy count (0 or non-zero)
        const size_t count;
    } args_t;

    typedef void (*fn_t)(const args_t*);

    static fn_t get() {
        if (is_x64() && mayiuse(jit::avx512_core) && mayiuse(jit::fp16)) {
            static jit_check_f16_compression_avx512 generator;

            // Since the JIT code always resides in memory, and ASAN's memory management may remove executable
            // permissions, we need to restore executable permissions for the generated code.
            generator.setProtectModeRE(false);
            return (fn_t)generator.getCode();
        }
        return nullptr;
    }

private:
    jit_check_f16_compression_avx512() : jit::Generator(jit::avx512_core) {
        using namespace Xbyak;

        const uint32_t vlen = 16u;  // 16 floats per zmm

        // GP register allocation
        const auto& reg_src = rax;
        const auto& reg_oor_dst = rbx;    // callee-saved
        const auto& reg_lossy_dst = r12;  // callee-saved
        const auto& reg_count = rdx;
        const auto& reg_oor_accum = r14;  // callee-saved — 64-bit OOR + RelErr accumulator
        const auto& reg_main_iters = r8;
        const auto& reg_idx = rsi;
        const auto& reg_tmp = r9;
        const auto& reg_addr = r15;  // callee-saved — used to load constants

        // ZMM register allocation
        const auto& data_vec = zmm0;        // chunk of 16 floats
        const auto& rt_vec = zmm1;          // f32->f16->f32 roundtripped
        const auto& rt_ymm = ymm1;          // low 256-bit alias for vcvtps2ph dest
        const auto& diff_vec = zmm2;        // |data - roundtripped|
        const auto& abs_data_vec = zmm3;    // |data|
        const auto& rel_thresh_vec = zmm4;  // |data| * 1e-4
        const auto& abs_mask_vec = zmm5;    // 0x7FFFFFFF broadcast
        const auto& abs_err_vec = zmm6;     // 1.0f broadcast
        const auto& rel_err_vec = zmm7;     // 1e-4f broadcast
        const auto& f16_max_pos_vec = zmm8;
        const auto& f16_max_neg_vec = zmm9;
        const auto& f16_min_pos_vec = zmm10;
        const auto& f16_min_neg_vec = zmm11;
        const auto& zero_vec = zmm12;

        // Opmasks (k0 reserved by ISA — represents "no mask")
        const auto& k_oor = k1;    // OOR per chunk (subnormal | overflow), then |= relative-error
        const auto& k_lossy = k2;  // (abs_diff > 1.0) AND NOT k_oor — early exit
        const auto& k_rel = k3;    // (abs_diff > |data|*1e-4) AND NOT k_oor
        const auto& k_tail = k4;   // tail mask (built once before tail iteration)
        const auto& k_tmp1 = k5;
        const auto& k_tmp2 = k6;

        Label main_loop, tail_block, normal_exit, lossy_exit;

        preamble();
        xor_(reg_oor_accum, reg_oor_accum);

        auto bcast = [&, this](const Xbyak::Zmm& vec, const void* p) {
            mov(reg_addr, reinterpret_cast<size_t>(p));
            vbroadcastss(vec, dword[reg_addr]);
        };

        bcast(f16_max_pos_vec, &kF16MaxPos);
        bcast(f16_max_neg_vec, &kF16MaxNeg);
        bcast(f16_min_pos_vec, &kF16MinPos);
        bcast(f16_min_neg_vec, &kF16MinNeg);
        bcast(abs_err_vec, &kF16CompressionAbsErrVal);
        bcast(rel_err_vec, &kF16CompressionRelErrVal);
        bcast(abs_mask_vec, &kAbsMaskVal);
        vpxorq(zero_vec, zero_vec, zero_vec);

        // --- Load args ---
        mov(reg_src, ptr[param + offsetof(args_t, src)]);
        mov(reg_oor_dst, ptr[param + offsetof(args_t, oor_dst)]);
        mov(reg_lossy_dst, ptr[param + offsetof(args_t, lossy_dst)]);
        mov(reg_count, ptr[param + offsetof(args_t, count)]);

        const unsigned char _cmp_lt_os = 1;
        const unsigned char _cmp_neq_uq = 4;
        const unsigned char _cmp_gt_os = 6;

        // Per-chunk emission. If `masked` is true, all opmasks are AND-ed with k_tail
        // to defensively clear out-of-tail lanes (masked load already zeroed them, but
        // this guards against any non-zero residual from sign-bit operations on -0.0).
        auto emit_chunk = [&, this](bool masked) {
            // === Build OOR mask in k_oor ===
            // subnormal-when-rounded: |data| < min_pos AND data != 0
            vcmpps(k_tmp1, data_vec, f16_min_pos_vec, _cmp_lt_os);  // data < min_pos
            vcmpps(k_tmp2, data_vec, f16_min_neg_vec, _cmp_gt_os);  // data > min_neg
            kandw(k_oor, k_tmp1, k_tmp2);                           // in (-min_pos, min_pos)
            vcmpps(k_tmp1, data_vec, zero_vec, _cmp_neq_uq);        // data != 0
            kandw(k_oor, k_oor, k_tmp1);
            // overflow: data > max_pos OR data < max_neg
            vcmpps(k_tmp1, data_vec, f16_max_pos_vec, _cmp_gt_os);
            korw(k_oor, k_oor, k_tmp1);
            vcmpps(k_tmp1, data_vec, f16_max_neg_vec, _cmp_lt_os);
            korw(k_oor, k_oor, k_tmp1);
            if (masked) {
                kandw(k_oor, k_oor, k_tail);
            }

            // === Roundtrip f32 -> f16 -> f32 ===
            vcvtps2ph(rt_ymm, data_vec, kVcvtps2phRneNoExc);
            vcvtph2ps(rt_vec, rt_ymm);

            // === abs_diff = |data - roundtripped| ===
            vsubps(diff_vec, data_vec, rt_vec);
            vpandd(diff_vec, diff_vec, abs_mask_vec);  // mask off sign bit

            // === Lossy check: (abs_diff > 1.0) AND NOT k_oor -> early exit ===
            vcmpps(k_lossy, diff_vec, abs_err_vec, _cmp_gt_os);
            kandnw(k_lossy, k_oor, k_lossy);  // k_lossy = ~k_oor & k_lossy
            if (masked) {
                kandw(k_lossy, k_lossy, k_tail);
            }
            kortestw(k_lossy, k_lossy);
            jnz(lossy_exit, T_NEAR);

            // === Relative-error check: (abs_diff > |data| * 1e-4) AND NOT k_oor ===
            vpandd(abs_data_vec, data_vec, abs_mask_vec);
            vmulps(rel_thresh_vec, abs_data_vec, rel_err_vec);
            vcmpps(k_rel, diff_vec, rel_thresh_vec, _cmp_gt_os);
            kandnw(k_rel, k_oor, k_rel);
            korw(k_oor, k_oor, k_rel);
            if (masked) {
                kandw(k_oor, k_oor, k_tail);
            }

            // === Accumulate popcount(k_oor) into reg_oor_accum ===
            // Branchless 16-bit SWAR popcount — avoids POPCNT (not gated by mayiuse(avx512_core)).
            kmovw(reg_tmp.cvt32(), k_oor);           // x = k_oor (16 bits in low word)
            mov(reg_addr.cvt32(), reg_tmp.cvt32());  // tmp2 = x
            shr(reg_addr.cvt32(), 1);
            and_(reg_addr.cvt32(), 0x5555);
            sub(reg_tmp.cvt32(), reg_addr.cvt32());  // x -= (x >> 1) & 0x5555
            mov(reg_addr.cvt32(), reg_tmp.cvt32());
            and_(reg_addr.cvt32(), 0x3333);
            shr(reg_tmp.cvt32(), 2);
            and_(reg_tmp.cvt32(), 0x3333);
            add(reg_tmp.cvt32(), reg_addr.cvt32());  // x = (x & 0x3333) + ((x >> 2) & 0x3333)
            mov(reg_addr.cvt32(), reg_tmp.cvt32());
            shr(reg_addr.cvt32(), 4);
            add(reg_tmp.cvt32(), reg_addr.cvt32());
            and_(reg_tmp.cvt32(), 0x0F0F);  // x = (x + (x >> 4)) & 0x0F0F
            mov(reg_addr.cvt32(), reg_tmp.cvt32());
            shr(reg_addr.cvt32(), 8);
            add(reg_tmp.cvt32(), reg_addr.cvt32());  // low byte = popcount (max 16)
            and_(reg_tmp, 0xFF);                     // zero upper bits before 64-bit add
            add(reg_oor_accum, reg_tmp);
        };

        // --- Main loop: 16 elements per iteration ---
        mov(reg_main_iters, reg_count);
        shr(reg_main_iters, 4);
        xor_(reg_idx, reg_idx);

        L(main_loop);
        cmp(reg_idx, reg_main_iters);
        jge(tail_block, T_NEAR);

        vmovups(data_vec, zword[reg_src]);
        emit_chunk(/*masked=*/false);
        add(reg_src, vlen * sizeof(float));
        inc(reg_idx);
        jmp(main_loop, T_NEAR);

        // --- Tail: count & 15 elements via opmask ---
        L(tail_block);
        and_(reg_count, 15);
        jz(normal_exit, T_NEAR);

        // k_tail = (1 << reg_count) - 1, using baseline SHL (BMI2 bzhi not gated by mayiuse(avx512_core)).
        // rcx is free at this point: args_t pointer was consumed in the preamble.
        mov(rcx, reg_count);
        mov(reg_tmp, 1);
        shl(reg_tmp, cl);
        dec(reg_tmp);
        kmovw(k_tail, reg_tmp.cvt32());

        // Masked zeroing-load: out-of-mask lanes read as +0.0 (benign — pass none of the predicates)
        vmovups(data_vec | k_tail | T_z, zword[reg_src]);
        emit_chunk(/*masked=*/true);

        // --- Normal exit: write OOR accumulator and lossy=0 ---
        L(normal_exit);
        mov(qword[reg_oor_dst], reg_oor_accum);
        xor_(reg_tmp, reg_tmp);
        mov(qword[reg_lossy_dst], reg_tmp);
        postamble();

        // --- Lossy exit: partial OOR count is meaningless, write 0; lossy=1 ---
        L(lossy_exit);
        xor_(reg_tmp, reg_tmp);
        mov(qword[reg_oor_dst], reg_tmp);
        mov(reg_tmp, 1);
        mov(qword[reg_lossy_dst], reg_tmp);
        postamble();
    }
};

// Combined single-pass JIT kernel: counts out-of-range AND detects lossy f16 compression.
// Processes 8 floats per iteration using AVX2+F16C. Bails immediately if any in-range
// element has significant precision loss (abs error > 1.0).
class jit_check_f16_compression : public jit::Generator {
public:
    typedef struct {
        const void* src;
        void* oor_dst;    // size_t* — output: combined out-of-range + high-relative-error count
        void* lossy_dst;  // size_t* — output: lossy count (0 or non-zero)
        const size_t count;
    } args_t;

    typedef void (*fn_t)(const args_t*);

    static fn_t get() {
        if (is_x64() && mayiuse(jit::avx2) && mayiuse(jit::fp16)) {
            static jit_check_f16_compression generator;

            // Since the JIT code always resides in memory, and ASAN's memory management may remove executable
            // permissions, we need to restore executable permissions for the generated code.
            generator.setProtectModeRE(false);
            return (fn_t)generator.getCode();
        }
        return nullptr;
    }

private:
    jit_check_f16_compression() {
        using namespace Xbyak;

        const uint32_t vlen = 8u;

        // GP register allocation
        const auto& reg_src = rax;
        const auto& reg_oor_dst = rbx;
        const auto& reg_lossy_dst = r12;  // callee-saved, preserved by preamble
        const auto& reg_sz = rdx;
        const auto& reg_saved_rsp = r13;  // callee-saved, for stack cleanup on lossy exit

        // YMM register allocation
        // Range check constants (same as jit_count_out_of_range)
        const auto& data_vec = ymm1;
        const auto& mask_vec = ymm2;  // OOR mask per chunk
        const auto& mask_vec_xmm = xmm2;
        const auto& tmp_vec = ymm3;
        const auto& oor_accum = ymm4;
        const auto& oor_accum_xmm = xmm4;
        const auto& f16_max_pos_vec = ymm5;
        const auto& f16_max_neg_vec = ymm6;
        const auto& f16_min_pos_vec = ymm7;
        const auto& f16_min_neg_vec = ymm8;
        const auto& f16_zero_vec = ymm9;
        const auto& i32_ones_vec = ymm10;
        // Lossy check constants
        const auto& abs_err_vec = ymm11;  // 1.0f broadcast
        const auto& abs_mask_vec = ymm0;  // 0x7FFFFFFF broadcast (for fabs)
        const auto& rel_err_vec = ymm12;  // 1e-4f broadcast (for relative error threshold)
        // Temps for lossy computation (reused per chunk)
        const auto& diff_vec = ymm14;
        const auto& rt_vec_xmm = xmm15;  // for f32->f16->f32 roundtrip

        Label exit_lossy, exit_normal;

        preamble();
        mov(reg_saved_rsp, rsp);  // save rsp for lossy exit stack cleanup

        // --- Load constants ---
        // 256-bit (8-lane) broadcast arrays for AVX2 (no vbroadcastss on ymm here; we use vmovdqu).
        // Arrays filled from FP16-derived constants (non-constexpr — see kF16* above) are static const;
        // arrays filled from literal/constexpr thresholds are static constexpr.
        static const float max_pos_bounds[8] =
            {kF16MaxPos, kF16MaxPos, kF16MaxPos, kF16MaxPos, kF16MaxPos, kF16MaxPos, kF16MaxPos, kF16MaxPos};
        static const float max_neg_bounds[8] =
            {kF16MaxNeg, kF16MaxNeg, kF16MaxNeg, kF16MaxNeg, kF16MaxNeg, kF16MaxNeg, kF16MaxNeg, kF16MaxNeg};
        static const float min_pos_bounds[8] =
            {kF16MinPos, kF16MinPos, kF16MinPos, kF16MinPos, kF16MinPos, kF16MinPos, kF16MinPos, kF16MinPos};
        static const float min_neg_bounds[8] =
            {kF16MinNeg, kF16MinNeg, kF16MinNeg, kF16MinNeg, kF16MinNeg, kF16MinNeg, kF16MinNeg, kF16MinNeg};
        static constexpr int32_t i32_ones[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        static constexpr float abs_errs[8] = {kF16CompressionAbsErrVal,
                                               kF16CompressionAbsErrVal,
                                               kF16CompressionAbsErrVal,
                                               kF16CompressionAbsErrVal,
                                               kF16CompressionAbsErrVal,
                                               kF16CompressionAbsErrVal,
                                               kF16CompressionAbsErrVal,
                                               kF16CompressionAbsErrVal};
        static constexpr float rel_errs[8] = {kF16CompressionRelErrVal,
                                               kF16CompressionRelErrVal,
                                               kF16CompressionRelErrVal,
                                               kF16CompressionRelErrVal,
                                               kF16CompressionRelErrVal,
                                               kF16CompressionRelErrVal,
                                               kF16CompressionRelErrVal,
                                               kF16CompressionRelErrVal};
        static constexpr uint32_t abs_masks[8] =
            {kAbsMaskVal, kAbsMaskVal, kAbsMaskVal, kAbsMaskVal, kAbsMaskVal, kAbsMaskVal, kAbsMaskVal, kAbsMaskVal};

        auto load_vec = [this](Ymm vec, size_t ptr) {
            mov(r15, ptr);
            vmovdqu(vec, yword[r15]);
        };

        load_vec(f16_max_pos_vec, reinterpret_cast<size_t>(max_pos_bounds));
        load_vec(f16_max_neg_vec, reinterpret_cast<size_t>(max_neg_bounds));
        load_vec(f16_min_pos_vec, reinterpret_cast<size_t>(min_pos_bounds));
        load_vec(f16_min_neg_vec, reinterpret_cast<size_t>(min_neg_bounds));
        load_vec(i32_ones_vec, reinterpret_cast<size_t>(i32_ones));
        load_vec(abs_err_vec, reinterpret_cast<size_t>(abs_errs));
        load_vec(rel_err_vec, reinterpret_cast<size_t>(rel_errs));
        load_vec(abs_mask_vec, reinterpret_cast<size_t>(abs_masks));
        vxorps(f16_zero_vec, f16_zero_vec, f16_zero_vec);
        vxorps(oor_accum, oor_accum, oor_accum);

        // Load args
        mov(reg_src, ptr[param + offsetof(args_t, src)]);
        mov(reg_oor_dst, ptr[param + offsetof(args_t, oor_dst)]);
        mov(reg_lossy_dst, ptr[param + offsetof(args_t, lossy_dst)]);
        mov(reg_sz, ptr[param + offsetof(args_t, count)]);

        const unsigned char _cmp_lt_os = 1;
        const unsigned char _cmp_neq_uq = 4;
        const unsigned char _cmp_gt_os = 6;

        // Kernel lambda: emits the combined check code for 8 elements at src_ptr.
        // Called twice (main loop + tail) — each call emits a copy of the instructions.
        auto emit_kernel = [&](const RegExp& src_ptr) {
            // Load 8 floats
            vmovups(data_vec, yword[src_ptr]);

            // === Out-of-range check (same algorithm as jit_count_out_of_range_vec) ===
            // subnormal: data in (-f16_min_pos, f16_min_pos) and != 0
            vcmpps(tmp_vec, data_vec, f16_min_pos_vec, _cmp_lt_os);
            vcmpps(mask_vec, data_vec, f16_min_neg_vec, _cmp_gt_os);
            vandps(mask_vec, mask_vec, tmp_vec);
            vcmpps(tmp_vec, data_vec, f16_zero_vec, _cmp_neq_uq);
            vandps(mask_vec, mask_vec, tmp_vec);
            // overflow: data > f16_max or data < f16_lowest
            vcmpps(tmp_vec, data_vec, f16_max_pos_vec, _cmp_gt_os);
            vorps(mask_vec, mask_vec, tmp_vec);
            vcmpps(tmp_vec, data_vec, f16_max_neg_vec, _cmp_lt_os);
            vorps(mask_vec, mask_vec, tmp_vec);
            // mask_vec now has per-element OOR mask (0xFFFFFFFF for OOR, 0 for in-range)

            // === Lossy check for in-range elements ===
            // Roundtrip: f32 -> f16 -> f32 (f16 part written to low xmm of rt_vec_xmm)
            vcvtps2ph(rt_vec_xmm, data_vec, kVcvtps2phRneNoExc);  // f32 -> f16 (8 values)
            vcvtph2ps(diff_vec, rt_vec_xmm);                      // f16 -> f32 (roundtripped)

            // abs_diff = |data - roundtripped|
            vsubps(diff_vec, data_vec, diff_vec);
            vandps(diff_vec, diff_vec, abs_mask_vec);  // diff_vec = |diff|

            // lossy = |diff| > abs_error (1.0)
            vcmpps(tmp_vec, diff_vec, abs_err_vec, _cmp_gt_os);  // |diff| > 1.0

            // Exclude out-of-range elements: keep only in-range lossy
            vandnps(tmp_vec, mask_vec, tmp_vec);  // tmp_vec = NOT(oor) AND lossy

            // Early exit if ANY element is lossy
            vtestps(tmp_vec, tmp_vec);  // ZF=1 if all zeros
            jnz(exit_lossy, T_NEAR);    // bail if any lossy

            // === Relative error check: count elements where |diff| > |value| * 1e-4 ===
            // (equivalent to abs_diff / |value| > 1e-4, but avoids division)
            vandps(tmp_vec, data_vec, abs_mask_vec);         // tmp_vec = |data|
            vmulps(tmp_vec, tmp_vec, rel_err_vec);           // tmp_vec = |data| * 1e-4
            vcmpps(tmp_vec, diff_vec, tmp_vec, _cmp_gt_os);  // 1 where |diff| > |data|*1e-4
            vandnps(tmp_vec, mask_vec, tmp_vec);             // exclude OOR (avoid double-count)
            vorps(mask_vec, mask_vec, tmp_vec);              // combine OOR + relative-error

            // === Accumulate combined OOR + relative-error count ===
            vandps(mask_vec, mask_vec, i32_ones_vec);
            vphaddd(mask_vec, mask_vec, mask_vec);
            vpermq(mask_vec, mask_vec, 0x08);
            vpmovsxdq(mask_vec, mask_vec_xmm);
            vpaddq(oor_accum, oor_accum, mask_vec);
        };

        // --- Main loop: 8 elements per iteration ---
        Label main_loop, main_loop_exit;
        xor_(rsi, rsi);
        mov(r8, reg_sz);
        shr(r8, 3);

        L(main_loop);
        cmp(rsi, r8);
        jge(main_loop_exit, T_NEAR);

        emit_kernel(reg_src);
        add(reg_src, static_cast<uint32_t>(sizeof(float) * vlen));

        add(rsi, 1);
        jmp(main_loop, T_NEAR);
        L(main_loop_exit);

        // --- Tail: remaining elements (< 8), zero-padded ---
        shl(rsi, 3);
        sub(reg_sz, rsi);
        test(reg_sz, reg_sz);
        jz(exit_normal, T_NEAR);

        sub(rsp, vlen * sizeof(float));
        mov(r8, rsp);

        vpxor(mask_vec, mask_vec, mask_vec);  // zero
        vmovups(yword[r8], mask_vec);         // zero-pad buffer

        copy<float>(r8, reg_src, reg_sz);  // copy remaining elements
        emit_kernel(r8);                   // process (zeros won't trigger OOR or lossy)

        add(rsp, vlen * sizeof(float));  // free stack (only reached if no lossy)

        // --- Normal exit: no lossy elements found ---
        L(exit_normal);
        {
            // Horizontal sum of oor_accum (4 x i64) -> single i64
            const auto& tmp0 = xmm2;  // reuse mask_vec
            const auto& tmp1 = xmm3;  // reuse tmp_vec
            vextractf128(tmp0, oor_accum, 0);
            vextractf128(tmp1, oor_accum, 1);
            vpaddq(oor_accum_xmm, tmp0, tmp1);
            vpermilpd(tmp0, oor_accum_xmm, 0x01);
            vpaddq(oor_accum_xmm, oor_accum_xmm, tmp0);
            vmovq(qword[reg_oor_dst], oor_accum_xmm);

            // lossy = 0
            xor_(rsi, rsi);
            mov(qword[reg_lossy_dst], rsi);
        }
        postamble();

        // --- Lossy exit: bail immediately (jumped from kernel via jnz) ---
        L(exit_lossy);
        mov(rsp, reg_saved_rsp);  // restore rsp (handles both main loop and tail cases)
        xor_(rsi, rsi);
        mov(qword[reg_oor_dst], rsi);  // oor_count = 0 (meaningless, caller checks lossy first)
        mov(rsi, 1);
        mov(qword[reg_lossy_dst], rsi);  // lossy = 1
        postamble();
    }
};

#endif  // OV_CORE_USE_XBYAK_JIT

template <class Clamp, typename TI, typename TO>
void convert_impl(const TI* arg, TO* out, size_t count) {
#ifdef OV_CORE_USE_XBYAK_JIT
    if (util::may_i_use_dynamic_code()) {
        if (auto converter = jit_convert_array::get<TI, TO, Clamp::enabled>()) {
            jit_convert_array::args_t args = {arg, out, count};
            converter(&args);
            return;
        }
    }
#endif  // OV_CORE_USE_XBYAK_JIT
    Converter<TI, TO>::template apply<Clamp>(arg, out, count);
}
}  // namespace

template <>
void convert<uint8_t, float16>(const uint8_t* arg, float16* out, size_t count) {
    convert_impl<NoClamp>(arg, out, count);
}

template <>
void convert<float16, float>(const float16* arg, float* out, size_t count) {
    convert_impl<NoClamp>(arg, out, count);
}

template <>
void convert<float, float16>(const float* arg, float16* out, size_t count) {
    convert_impl<NoClamp>(arg, out, count);
}

template <>
void convert<float, int8_t>(const float* arg, int8_t* out, size_t count) {
    convert_impl<NoClamp>(arg, out, count);
}

template <>
void convert<float16, int8_t>(const float16* arg, int8_t* out, size_t count) {
    convert_impl<NoClamp>(arg, out, count);
}

template <>
void convert<bfloat16, float16>(const bfloat16* arg, float16* out, size_t count) {
    convert_impl<NoClamp>(arg, out, count);
}

template <>
void convert<bfloat16, float>(const bfloat16* arg, float* out, size_t count) {
    convert_impl<NoClamp>(arg, out, count);
}

void convert_from_f32_to_f16_with_clamp(const float* arg, float16* out, size_t count) {
    convert_impl<Clamp<float, float16>>(arg, out, count);
}

template <>
void convert<int32_t, float16>(const int32_t* arg, float16* out, size_t count) {
    Converter<int32_t, float16>::apply<Clamp<int32_t, float16>>(arg, out, count);
}

void convert_from_bf16_to_f16_with_clamp(const bfloat16* arg, float16* out, size_t count) {
    // can re-use Clamp as bf16 is converted to float before clamping
    using clamp_bf16_f16 = Clamp<float, float16>;
    convert_impl<clamp_bf16_f16>(arg, out, count);
    // CVS-125496: duplicate and stub for ARM, provide optimized solution
}

namespace {
// Predicate: true iff the FP16 representation of `v` falls outside the finite FP16
// normal range (subnormal when rounded, or magnitude larger than float16::max()).
// Shared between check_f16_compression() slow-path and count_out_of_f16_range().
inline bool is_out_of_f16_range(float v) {
    return (std::abs(v) < float16::from_bits(0x0001) && v != 0.0f) || v > std::numeric_limits<float16>::max() ||
           v < std::numeric_limits<float16>::lowest();
}
}  // namespace

CompressionCheckResult check_f16_compression(const float* arg, size_t count) {
#ifdef OV_CORE_USE_XBYAK_JIT
    if (util::may_i_use_dynamic_code()) {
        if (auto fn = jit_check_f16_compression_avx512::get()) {
            size_t oor_count = 0;
            size_t lossy_count = 0;
            jit_check_f16_compression_avx512::args_t args = {arg, &oor_count, &lossy_count, count};
            fn(&args);
            return {oor_count, lossy_count > 0};
        }
        if (auto fn = jit_check_f16_compression::get()) {
            size_t oor_count = 0;
            size_t lossy_count = 0;
            jit_check_f16_compression::args_t args = {arg, &oor_count, &lossy_count, count};
            fn(&args);
            return {oor_count, lossy_count > 0};
        }
    }
#endif  // OV_CORE_USE_XBYAK_JIT
    size_t out_of_range = 0;
    for (size_t i = 0; i < count; ++i) {
        const float v = arg[i];
        if (is_out_of_f16_range(v)) {
            ++out_of_range;
        } else {
            const double roundtripped = static_cast<double>(static_cast<float>(static_cast<float16>(v)));
            const double abs_diff = std::abs(v - roundtripped);
            if (abs_diff > f16_compression_max_abs_error) {
                return {0, true};
            }
            if (v != 0.0f && abs_diff / std::abs(v) > f16_compression_max_rel_error) {
                ++out_of_range;
            }
        }
    }
    return {out_of_range, false};
}

size_t count_out_of_f16_range(const float* arg, size_t count) {
    // Backward-compatible helper: strict FP16-range count only (no relative-error accounting).
    // The in-tree FP16 compression path uses check_f16_compression(); this wrapper exists for
    // external developer-package consumers that linked against the pre-PR symbol.
    return std::count_if(arg, arg + count, is_out_of_f16_range);
}

}  // namespace reference
}  // namespace ov
