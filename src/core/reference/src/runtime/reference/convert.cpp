// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/convert.hpp"

#include "jit_generator.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
namespace {
template <typename src_t, typename dst_t>
void jit_convert_vec(jit::Generator&, const Xbyak::RegExp&, const Xbyak::RegExp&);

template <typename src_t, typename dst_t>
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
    gen.movdqu(gen.xword[dst], f16vec);
}

template <>
void jit_convert_vec<float16, float>(jit::Generator& gen, const Xbyak::RegExp& src, const Xbyak::RegExp& dst) {
    auto f16vec = gen.xmm3;
    auto f32vec = gen.ymm4;

    gen.movdqu(f16vec, gen.xword[src]);
    gen.vcvtph2ps(f32vec, f16vec);
    gen.vmovups(gen.yword[dst], f32vec);
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

    template <typename src_t, typename dst_t>
    static fn_t get() {
        if (is_x64() && mayiuse(avx) && mayiuse(avx2) && mayiuse(fp16)) {
            static const jit_convert_array::context_t context{{sizeof(src_t), &jit::Generator::copy<src_t>},
                                                              {sizeof(dst_t), &jit::Generator::copy<dst_t>},
                                                              jit_convert_vec<src_t, dst_t>,
                                                              jit_convert_vec_prepare<src_t, dst_t>};

            static jit_convert_array generator(context);

            return (fn_t)generator.getCode();
        }
        return nullptr;
    }
};

template <typename TI, typename TO>
void convert_impl(const TI* arg, TO* out, size_t count) {
    auto converter = jit_convert_array::get<TI, TO>();

    if (converter) {
        jit_convert_array::args_t args = {arg, out, count};
        converter(&args);
    } else {
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<TO>(arg[i]);
        }
    }
}
}  // namespace

template <>
void convert<uint8_t, float16>(const uint8_t* arg, float16* out, size_t count) {
    convert_impl(arg, out, count);
}

template <>
void convert<float16, float>(const float16* arg, float* out, size_t count) {
    convert_impl(arg, out, count);
}

template <>
void convert<float, int8_t>(const float* arg, int8_t* out, size_t count) {
    convert_impl(arg, out, count);
}

template <>
void convert<float16, int8_t>(const float16* arg, int8_t* out, size_t count) {
    convert_impl(arg, out, count);
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
