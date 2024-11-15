// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/convert.hpp"

#include "openvino/reference/utils/convert_util.hpp"

#ifdef OV_CORE_USE_XBYAK_JIT
#    include "openvino/reference/utils/jit_generator.hpp"
#endif

#ifdef OV_CORE_USE_INTRINSICS
#    include "openvino/reference/utils/convert_x86_intrinsics.hpp"
#endif

namespace ov {
namespace reference {

namespace {
#ifdef OV_CORE_USE_XBYAK_JIT
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
    gen.vcvtps2ph(f16vec, fvec, 0);
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
    gen.vcvtps2ph(f16vec, f32vec, 0);
    gen.vmovdqu(gen.xword[dst], f16vec);
}

template <>
void jit_convert_vec<bfloat16, float16>(jit::Generator& gen, const Xbyak::RegExp& src, const Xbyak::RegExp& dst) {
    const auto f32vec = gen.ymm4;
    const auto f16vec = gen.xmm3;

    gen.vpmovzxwd(f32vec, gen.yword[src]);  // load bf16 into tmp
    gen.vpslld(f32vec, f32vec, 16);         // convert bf16->f32 by bit shift
    gen.vcvtps2ph(f16vec, f32vec, 0);       // convert f32 -> f16
    gen.vmovdqu(gen.xword[dst], f16vec);    // move result to destination
}

template <>
void jit_convert_vec<bfloat16, float16, true>(jit::Generator& gen, const Xbyak::RegExp& src, const Xbyak::RegExp& dst) {
    const auto f32vec = gen.ymm4;
    const auto f16vec = gen.xmm3;

    auto upper_bound = gen.ymm5;
    auto lower_bound = gen.ymm6;

    gen.vpmovzxwd(f32vec, gen.yword[src]);    // load bf16 into tmp
    gen.vpslld(f32vec, f32vec, 16);           // convert bf16->f32 by bit shift
    gen.vminps(f32vec, f32vec, upper_bound);  // clamp f16 max
    gen.vmaxps(f32vec, f32vec, lower_bound);  // clamp f16 lowest
    gen.vcvtps2ph(f16vec, f32vec, 0);         // convert f32 -> f16
    gen.vmovdqu(gen.xword[dst], f16vec);      // move result to destination
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

    static const float f16_max = std::numeric_limits<ov::float16>::max();
    static const float f16_min = std::numeric_limits<ov::float16>::lowest();
    static const float upper_bounds[8] = {f16_max, f16_max, f16_max, f16_max, f16_max, f16_max, f16_max, f16_max};
    static const float lower_bounds[8] = {f16_min, f16_min, f16_min, f16_min, f16_min, f16_min, f16_min, f16_min};

    gen.mov(addr, (size_t)upper_bounds);
    gen.vmovdqu(upper_bound, gen.yword[addr]);
    gen.mov(addr, (size_t)lower_bounds);
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
    gen.vcvtps2ph(f16vec, f32vec, 0);
    gen.vmovdqu(gen.xword[dst], f16vec);
}

template <>
void jit_convert_vec_prepare<float, int8_t>(jit::Generator& gen) {
    auto order = gen.ymm1;
    auto addr = gen.r15;

    static const int8_t offsets[32] = {0,  4,  8,  12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                       -1, -1, -1, -1, 0,  4,  8,  12, -1, -1, -1, -1, -1, -1, -1, -1};

    gen.mov(addr, (size_t)offsets);       // get offsets[] address
    gen.vmovdqu(order, gen.yword[addr]);  // save offsets[] to ymm register
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

template <>
void jit_count_out_of_range_vec_prepare<float, float16>(jit::Generator& gen) {
    auto accum_vec = gen.ymm4;
    auto f16_max_pos_vec = gen.ymm5;
    auto f16_max_neg_vec = gen.ymm6;
    auto f16_min_pos_vec = gen.ymm7;
    auto f16_min_neg_vec = gen.ymm8;
    auto f16_zero_vec = gen.ymm9;
    auto i32_ones_vec = gen.ymm10;
    auto addr = gen.r15;

    static const float f16_max_pos = std::numeric_limits<ov::float16>::max();
    static const float f16_max_neg = std::numeric_limits<ov::float16>::lowest();
    static const float f16_min_pos = ov::float16::from_bits(0x0001);
    static const float f16_min_neg = -ov::float16::from_bits(0x0001);
    static const int32_t i32_one = 1;

    static const float max_pos_bounds[8] =
        {f16_max_pos, f16_max_pos, f16_max_pos, f16_max_pos, f16_max_pos, f16_max_pos, f16_max_pos, f16_max_pos};
    static const float max_neg_bounds[8] =
        {f16_max_neg, f16_max_neg, f16_max_neg, f16_max_neg, f16_max_neg, f16_max_neg, f16_max_neg, f16_max_neg};
    static const float min_pos_bounds[8] =
        {f16_min_pos, f16_min_pos, f16_min_pos, f16_min_pos, f16_min_pos, f16_min_pos, f16_min_pos, f16_min_pos};
    static const float min_neg_bounds[8] =
        {f16_min_neg, f16_min_neg, f16_min_neg, f16_min_neg, f16_min_neg, f16_min_neg, f16_min_neg, f16_min_neg};
    static const int32_t i32_ones[8] = {i32_one, i32_one, i32_one, i32_one, i32_one, i32_one, i32_one, i32_one};

    auto load_vec = [&gen, &addr](Xbyak::Ymm vec, size_t ptr) {
        gen.mov(addr, ptr);
        gen.vmovdqu(vec, gen.yword[addr]);
    };

    load_vec(f16_max_pos_vec, (size_t)max_pos_bounds);
    load_vec(f16_max_neg_vec, (size_t)max_neg_bounds);
    load_vec(f16_min_pos_vec, (size_t)min_pos_bounds);
    load_vec(f16_min_neg_vec, (size_t)min_neg_bounds);
    load_vec(i32_ones_vec, (size_t)i32_ones);
    gen.vxorps(f16_zero_vec, f16_zero_vec, f16_zero_vec);
    gen.vxorps(accum_vec, accum_vec, accum_vec);
}

template <>
void jit_count_out_of_range_vec<float, float16>(jit::Generator& gen, const Xbyak::RegExp& data) {
    auto data_vec = gen.ymm1;
    auto mask_vec = gen.ymm2;
    auto mask_vec_xmm = gen.xmm2;
    auto tmp_vec = gen.ymm3;
    auto accum_vec = gen.ymm4;
    auto f16_max_pos_vec = gen.ymm5;
    auto f16_max_neg_vec = gen.ymm6;
    auto f16_min_pos_vec = gen.ymm7;
    auto f16_min_neg_vec = gen.ymm8;
    auto f16_zero_vec = gen.ymm9;
    auto i32_ones_vec = gen.ymm10;

    const unsigned char _cmp_lt_os = 1;
    const unsigned char _cmp_neq_uq = 4;
    const unsigned char _cmp_gt_os = 6;

    // std::abs(data) < ov::float16::from_bits(0x0001)
    gen.vmovups(data_vec, gen.yword[data]);
    gen.vcmpps(tmp_vec, data_vec, f16_min_pos_vec, _cmp_lt_os);
    gen.vcmpps(mask_vec, data_vec, f16_min_neg_vec, _cmp_gt_os);
    gen.vandps(mask_vec, mask_vec, tmp_vec);

    // data != 0.0f
    gen.vcmpps(tmp_vec, data_vec, f16_zero_vec, _cmp_neq_uq);
    gen.vandps(mask_vec, mask_vec, tmp_vec);

    // data > std::numeric_limits<ov::float16>::max()
    gen.vcmpps(tmp_vec, data_vec, f16_max_pos_vec, _cmp_gt_os);
    gen.vorps(mask_vec, mask_vec, tmp_vec);

    // data < std::numeric_limits<ov::float16>::lowest()
    gen.vcmpps(tmp_vec, data_vec, f16_max_neg_vec, _cmp_lt_os);
    gen.vorps(mask_vec, mask_vec, tmp_vec);

    // addition to i64 accumulator
    gen.vandps(mask_vec, mask_vec, i32_ones_vec);
    gen.vphaddd(mask_vec, mask_vec, mask_vec);
    gen.vpermq(mask_vec, mask_vec, 0x08);
    gen.vpmovsxdq(mask_vec, mask_vec_xmm);
    gen.vpaddq(accum_vec, accum_vec, mask_vec);
}

template <>
void jit_count_out_of_range_vec_finalize<float, float16>(jit::Generator& gen, const Xbyak::RegExp& dst) {
    auto tmp_vec_xmm0 = gen.xmm2;  // reuse mask_vec
    auto tmp_vec_xmm1 = gen.xmm3;  // reuse tmp_vec
    auto accum_vec_ymm = gen.ymm4;
    auto accum_vec_xmm = gen.xmm4;

    // horizontal sum of four i64 values
    gen.vextractf128(tmp_vec_xmm0, accum_vec_ymm, 0);
    gen.vextractf128(tmp_vec_xmm1, accum_vec_ymm, 1);
    gen.vpaddq(accum_vec_xmm, tmp_vec_xmm0, tmp_vec_xmm1);
    gen.vpermilpd(tmp_vec_xmm0, accum_vec_xmm, 0x01);
    gen.vpaddq(accum_vec_xmm, accum_vec_xmm, tmp_vec_xmm0);
    gen.vmovq(gen.qword[dst], accum_vec_xmm);
}

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

            return (fn_t)generator.getCode();
        }
        return nullptr;
    }
};

#endif  // OV_CORE_USE_XBYAK_JIT

template <class Clamp, typename TI, typename TO>
void convert_impl(const TI* arg, TO* out, size_t count) {
#ifdef OV_CORE_USE_XBYAK_JIT
    if (auto converter = jit_convert_array::get<TI, TO, Clamp::enabled>()) {
        jit_convert_array::args_t args = {arg, out, count};
        converter(&args);
    } else
#endif
    {
        Converter<TI, TO>::template apply<Clamp>(arg, out, count);
    }
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
    // FIXME CVS-125496: duplicate and stub for ARM, provide optimized solution
}

size_t count_out_of_f16_range(const float* arg, size_t count) {
#ifdef OV_CORE_USE_XBYAK_JIT
    if (auto converter = jit_count_out_of_range::get<float, float16>()) {
        size_t num_out_of_range = 0;
        jit_count_out_of_range::args_t args = {arg, &num_out_of_range, count};
        converter(&args);
        return num_out_of_range;
    }
#endif  // OV_CORE_USE_XBYAK_JIT
    const auto is_out_of_f16_range = [](const float v) {
        return (std::abs(v) < float16::from_bits(0x0001) && v != 0.0f) || (v > std::numeric_limits<float16>::max()) ||
               (v < std::numeric_limits<float16>::lowest());
    };

    return std::count_if(arg, arg + count, is_out_of_f16_range);
}

}  // namespace reference
}  // namespace ov
