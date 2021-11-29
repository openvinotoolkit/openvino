// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_convert.h"
#include "cpu_memcpy.h"
#include "utils/bfloat16.hpp"
#include <mkldnn_selective_build.h>
#include <ie_parallel.hpp>
#include <openvino/core/type/float16.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <algorithm>
#include <type_traits>
#include <tuple>
#include <cmath>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace {

template <typename src_t, typename dst_t>
class jit_convert_array : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_convert_array)

    void generate() override {
        const size_t vlen = 8u;
        const size_t vlen_log2 = 3;

        auto reg_src = rax;
        auto reg_dst = rbx;
        auto reg_sz = rdx;

        Label tail, exit;

        preamble();

        mov(reg_src, ptr[param1 + offsetof(args_t, src)]);
        mov(reg_dst, ptr[param1 + offsetof(args_t, out)]);
        mov(reg_sz, ptr[param1 + offsetof(args_t, count)]);

        xor_(rsi, rsi);
        mov(r8, reg_sz);
        shr(r8, vlen_log2);

        foreach(rsi, 1, r8, [&, this](const Xbyak::Reg64& idx) {
            convert_vec(reg_src, reg_dst);
            add(reg_src, sizeof(src_t) * vlen);
            add(reg_dst, sizeof(dst_t) * vlen);
        });

        L(tail);

        shl(rsi, vlen_log2);
        sub(reg_sz, rsi);
        test(reg_sz, reg_sz);
        jz(exit);

        // allocate array for 8 floats on stack
        sub(rsp, vlen * sizeof(float));
        mov(r8, rsp);

        vpxor(ymm4, ymm4, ymm4);
        vmovups(yword[r8], ymm4);

        // Tail conversion
        copy<src_t>(r8, reg_src, reg_sz);
        convert_vec(r8, r8);
        copy<dst_t>(reg_dst, r8, reg_sz);

        // Free the array on stack
        add(rsp, vlen * sizeof(float));

        L(exit);

        postamble();
    }

    void foreach(const Xbyak::Reg64& idx,
                 size_t step,
                 const Xbyak::Reg64& end,
                 std::function<void(const Xbyak::Reg64&)> && fn) {
        Label loop, exit;

        L(loop);
        cmp(idx, end);
        jge(exit);

        fn(idx);

        add(idx, step);
        jmp(loop);
        L(exit);
    }

    void convert_vec(const RegExp & src,
                     const RegExp & dst);

    template<typename T>
    void copy(const Xbyak::Reg64& dst,
              const Xbyak::Reg64& src,
              const Xbyak::Reg64& size) {
        push(rsi);
        push(r15);

        xor_(rsi, rsi);

        auto address_frame = [this](size_t size) -> const AddressFrame& {
            switch (size) {
                case 1: return byte;
                case 2: return word;
                case 4: return dword;
                case 8: return qword;
                default:
                    break;
            }
            return ptr;
        };

        const auto & addr_frame = address_frame(sizeof(T));

        foreach(rsi, 1, size, [&, this](const Xbyak::Reg64& idx) {
            mov(r15, addr_frame[src + idx * sizeof(T)]);
            mov(addr_frame[dst + idx * sizeof(T)], r15);
        });

        pop(r15);
        pop(rsi);
    }

public:
    typedef struct {
        const src_t* src;
        dst_t* out;
        const size_t count;
    } args_t;

    typedef void (*fn_t)(const args_t*);

    static fn_t get() {
        if (mayiuse(avx2) && cpu().has(util::Cpu::tF16C)) {
            static jit_convert_array converter;
            auto & generator = static_cast<jit_generator&>(converter);
            generator.create_kernel();
            return (fn_t)generator.jit_ker();
        }
        return nullptr;
    }
};

template <>
void jit_convert_array<ov::float16, float>::convert_vec(const RegExp & src,
                                                        const RegExp & dst) {
    auto f16vec = xmm3;
    auto f32vec = ymm4;

    movdqu(f16vec, xword[src]);
    vcvtph2ps(f32vec, f16vec);
    vmovups(yword[dst], f32vec);
}

template <>
void jit_convert_array<float, ov::float16>::convert_vec(const RegExp & src,
                                                        const RegExp & dst) {
    auto f16vec = xmm3;
    auto f32vec = ymm4;

    vmovups(f32vec, yword[src]);
    vcvtps2ph(f16vec, f32vec, 0);
    movdqu(xword[dst], f16vec);
}

template <typename TI, typename TO>
void jit_convert(const TI* arg, TO* out, size_t count) {
    using jit_impl = jit_convert_array<TI, TO>;
    static auto converter = jit_impl::get();

    if (converter) {
        typename jit_impl::args_t args = { arg, out, count };
        converter(&args);
    } else {
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<TO>(arg[i]);
        }
    }
}

template <Precision::ePrecision p>
struct PrecisionInfo {
    using value_type = typename PrecisionTrait<p>::value_type;
};

template <>
struct PrecisionInfo<Precision::BF16> {
    using value_type = MKLDNNPlugin::bfloat16_t;
};

template <>
struct PrecisionInfo<Precision::FP16> {
    using value_type = ov::float16;
};

template <>
struct PrecisionInfo<Precision::BOOL> {
    using value_type = bool;
};

struct ConvertContext {
    const void *srcPtr;
    void *dstPtr;
    size_t size;
    bool converted;
};

template<typename T>
struct ConvertPrecision;

template<typename src_t, typename dst_t>
struct ConvertPrecision<std::tuple<src_t, dst_t>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const src_t *>(ctx.srcPtr);
        auto dst = static_cast<dst_t *>(ctx.dstPtr);
        parallel_for(ctx.size, [&](size_t i) {
            dst[i] = static_cast<dst_t>(src[i]);
        });
        ctx.converted = true;
    }
};

template<typename src_t>
struct ConvertPrecision<std::tuple<src_t, ov::float16>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const src_t *>(ctx.srcPtr);
        auto dst = static_cast<ov::float16 *>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = (ctx.size + batch - 1) / batch;
        typedef float batch_type[batch];

        parallel_for(iterations, [&](size_t i) {
            batch_type tmp;
            const size_t offset = i * batch;
            const size_t current_batch_size = std::min(ctx.size - offset, batch);
            for (size_t j = 0; j < current_batch_size; ++j)         // src_t -> fp32
                tmp[j] = static_cast<float>(src[offset + j]);
            jit_convert(tmp, dst + offset, current_batch_size);     // fp32 -> fp16
        });

        ctx.converted = true;
    }
};

template<typename dst_t>
struct ConvertPrecision<std::tuple<ov::float16, dst_t>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const ov::float16 *>(ctx.srcPtr);
        auto dst = static_cast<dst_t *>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = (ctx.size + batch - 1) / batch;
        typedef float batch_type[batch];

        parallel_for(iterations, [&](size_t i) {
            batch_type tmp;
            const size_t offset = i * batch;
            const size_t current_batch_size = std::min(ctx.size - offset, batch);
            jit_convert(src + offset, tmp, current_batch_size);     // fp16 -> fp32
            for (size_t j = 0; j < current_batch_size; ++j)         // fp32 -> dst_t
                dst[offset + j] = static_cast<dst_t>(tmp[j]);
        });

        ctx.converted = true;
    }
};

template<typename T, typename U>
std::tuple<T, T> commonRange() {
    if (std::is_integral<T>::value && std::is_integral<U>::value) {
        int64_t lbound = static_cast<int64_t>(std::numeric_limits<T>::lowest());
        lbound = std::max(lbound, static_cast<int64_t>(std::numeric_limits<U>::lowest()));
        uint64_t ubound = static_cast<uint64_t>(std::numeric_limits<T>::max());
        ubound = std::min(ubound, static_cast<uint64_t>(std::numeric_limits<U>::max()));
        return std::make_tuple(static_cast<T>(lbound), static_cast<T>(ubound));
    }

    double lbound = static_cast<double>(std::numeric_limits<T>::lowest());
    lbound = std::max(lbound, static_cast<double>(std::numeric_limits<U>::lowest()));
    double ubound = static_cast<double>(std::numeric_limits<T>::max());
    ubound = std::min(ubound, static_cast<double>(std::numeric_limits<U>::max()));
    return std::make_tuple(static_cast<T>(lbound), static_cast<T>(ubound));
}

struct TruncateContext {
    const void *srcPtr;
    void *dstPtr;
    size_t size;
    Precision interimPrc;
    bool converted;

    template<typename src_t, typename dst_t>
    std::tuple<src_t, src_t> range() const {
        std::tuple<src_t, src_t> range_0;
        switch (interimPrc) {
            case Precision::U8:
                range_0 = commonRange<src_t, PrecisionInfo<Precision::U8>::value_type>();
                break;
            case Precision::I8:
                range_0 = commonRange<src_t, PrecisionInfo<Precision::I8>::value_type>();
                break;
            case Precision::U16:
                range_0 = commonRange<src_t, PrecisionInfo<Precision::U16>::value_type>();
                break;
            case Precision::I16:
                range_0 = commonRange<src_t, PrecisionInfo<Precision::I16>::value_type>();
                break;
            case Precision::U32:
                range_0 = commonRange<src_t, PrecisionInfo<Precision::U32>::value_type>();
                break;
            case Precision::I32:
                range_0 = commonRange<src_t, PrecisionInfo<Precision::I32>::value_type>();
                break;
            case Precision::U64:
                range_0 = commonRange<src_t, PrecisionInfo<Precision::U64>::value_type>();
                break;
            case Precision::I64:
                range_0 = commonRange<src_t, PrecisionInfo<Precision::I64>::value_type>();
                break;
            case Precision::FP32:
                range_0 = commonRange<src_t, PrecisionInfo<Precision::FP32>::value_type>();
                break;
            case Precision::FP16:
                range_0 = commonRange<src_t, PrecisionInfo<Precision::FP16>::value_type>();
                break;
            case Precision::BF16:
                range_0 = commonRange<src_t, PrecisionInfo<Precision::BF16>::value_type>();
                break;
            case Precision::FP64:
                range_0 = commonRange<src_t, PrecisionInfo<Precision::FP64>::value_type>();
                break;
            case Precision::BOOL:
                range_0 = commonRange<src_t, PrecisionInfo<Precision::BOOL>::value_type>();
                break;
            default:
                IE_THROW() << "Unsupported precision";
        }

        auto range_1 = commonRange<src_t, dst_t>();

        return std::make_pair(std::max(std::get<0>(range_0), std::get<0>(range_1)),
                              std::min(std::get<1>(range_0), std::get<1>(range_0)));
    }
};

template<typename T>
struct TruncatePrecision;

template<typename src_t, typename dst_t>
struct TruncatePrecision<std::tuple<src_t, dst_t>> {
    void operator()(TruncateContext & ctx) {
        auto src = static_cast<const src_t *>(ctx.srcPtr);
        auto dst = static_cast<dst_t *>(ctx.dstPtr);
        src_t lbound, ubound;
        std::tie(lbound, ubound) = ctx.range<src_t, dst_t>();

        if (std::is_integral<src_t>::value
            || ctx.interimPrc.is_float()
            || std::is_integral<dst_t>::value) {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<dst_t>(std::max(std::min(src[i], ubound), lbound));
            });
        } else {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<dst_t>(std::trunc(std::max(std::min(src[i], ubound), lbound)));
            });
        }

        ctx.converted = true;
    }
};

bool isConversionTruncatesRange(const Precision & from, const Precision & to) {
    return to.bitsSize() < from.bitsSize()
            || (from.is_float() && !to.is_float())      // float -> integral
            || (from.isSigned() != to.isSigned())       // signed <-> unsigned
            || (to == Precision::BOOL && from != to);   // T -> bool
}

}   // namespace

#define MKLDNN_CVT(ST, DT) OV_CASE2(Precision::ST, Precision::DT, PrecisionInfo<Precision::ST>::value_type, PrecisionInfo<Precision::DT>::value_type)

#define MKLDNN_CVT_LIST                                                                             \
    MKLDNN_CVT(U8, I8),     MKLDNN_CVT(U8, U16),    MKLDNN_CVT(U8, I16),    MKLDNN_CVT(U8, U32),    \
    MKLDNN_CVT(U8, I32),    MKLDNN_CVT(U8, U64),    MKLDNN_CVT(U8, I64),    MKLDNN_CVT(U8, FP32),   \
    MKLDNN_CVT(U8, FP16),   MKLDNN_CVT(U8, BF16),   MKLDNN_CVT(U8, FP64),   MKLDNN_CVT(U8, BOOL),   \
    MKLDNN_CVT(I8, U8),     MKLDNN_CVT(I8, U16),    MKLDNN_CVT(I8, I16),    MKLDNN_CVT(I8, U32),    \
    MKLDNN_CVT(I8, I32),    MKLDNN_CVT(I8, U64),    MKLDNN_CVT(I8, I64),    MKLDNN_CVT(I8, FP32),   \
    MKLDNN_CVT(I8, FP16),   MKLDNN_CVT(I8, BF16),   MKLDNN_CVT(I8, FP64),   MKLDNN_CVT(I8, BOOL),   \
    MKLDNN_CVT(U16, U8),    MKLDNN_CVT(U16, I8),    MKLDNN_CVT(U16, I16),   MKLDNN_CVT(U16, U32),   \
    MKLDNN_CVT(U16, I32),   MKLDNN_CVT(U16, U64),   MKLDNN_CVT(U16, I64),   MKLDNN_CVT(U16, FP32),  \
    MKLDNN_CVT(U16, FP16),  MKLDNN_CVT(U16, BF16),  MKLDNN_CVT(U16, FP64),  MKLDNN_CVT(U16, BOOL),  \
    MKLDNN_CVT(I16, U8),    MKLDNN_CVT(I16, I8),    MKLDNN_CVT(I16, U16),   MKLDNN_CVT(I16, U32),   \
    MKLDNN_CVT(I16, I32),   MKLDNN_CVT(I16, U64),   MKLDNN_CVT(I16, I64),   MKLDNN_CVT(I16, FP32),  \
    MKLDNN_CVT(I16, FP16),  MKLDNN_CVT(I16, BF16),  MKLDNN_CVT(I16, FP64),  MKLDNN_CVT(I16, BOOL),  \
    MKLDNN_CVT(U32, U8),    MKLDNN_CVT(U32, I8),    MKLDNN_CVT(U32, U16),   MKLDNN_CVT(U32, I16),   \
    MKLDNN_CVT(U32, I32),   MKLDNN_CVT(U32, U64),   MKLDNN_CVT(U32, I64),   MKLDNN_CVT(U32, FP32),  \
    MKLDNN_CVT(U32, FP16),  MKLDNN_CVT(U32, BF16),  MKLDNN_CVT(U32, FP64),  MKLDNN_CVT(U32, BOOL),  \
    MKLDNN_CVT(I32, U8),    MKLDNN_CVT(I32, I8),    MKLDNN_CVT(I32, U16),   MKLDNN_CVT(I32, I16),   \
    MKLDNN_CVT(I32, U32),   MKLDNN_CVT(I32, U64),   MKLDNN_CVT(I32, I64),   MKLDNN_CVT(I32, FP32),  \
    MKLDNN_CVT(I32, FP16),  MKLDNN_CVT(I32, BF16),  MKLDNN_CVT(I32, FP64),  MKLDNN_CVT(I32, BOOL),  \
    MKLDNN_CVT(U64, U8),    MKLDNN_CVT(U64, I8),    MKLDNN_CVT(U64, U16),   MKLDNN_CVT(U64, I16),   \
    MKLDNN_CVT(U64, U32),   MKLDNN_CVT(U64, I32),   MKLDNN_CVT(U64, I64),   MKLDNN_CVT(U64, FP32),  \
    MKLDNN_CVT(U64, FP16),  MKLDNN_CVT(U64, BF16),  MKLDNN_CVT(U64, FP64),  MKLDNN_CVT(U64, BOOL),  \
    MKLDNN_CVT(I64, U8),    MKLDNN_CVT(I64, I8),    MKLDNN_CVT(I64, U16),   MKLDNN_CVT(I64, I16),   \
    MKLDNN_CVT(I64, U32),   MKLDNN_CVT(I64, I32),   MKLDNN_CVT(I64, U64),   MKLDNN_CVT(I64, FP32),  \
    MKLDNN_CVT(I64, FP16),  MKLDNN_CVT(I64, BF16),  MKLDNN_CVT(I64, FP64),  MKLDNN_CVT(I64, BOOL),  \
    MKLDNN_CVT(FP32, U8),   MKLDNN_CVT(FP32, I8),   MKLDNN_CVT(FP32, U16),  MKLDNN_CVT(FP32, I16),  \
    MKLDNN_CVT(FP32, U32),  MKLDNN_CVT(FP32, I32),  MKLDNN_CVT(FP32, U64),  MKLDNN_CVT(FP32, I64),  \
    MKLDNN_CVT(FP32, FP16), MKLDNN_CVT(FP32, BF16), MKLDNN_CVT(FP32, FP64), MKLDNN_CVT(FP32, BOOL), \
    MKLDNN_CVT(FP16, U8),   MKLDNN_CVT(FP16, I8),   MKLDNN_CVT(FP16, U16),  MKLDNN_CVT(FP16, I16),  \
    MKLDNN_CVT(FP16, U32),  MKLDNN_CVT(FP16, I32),  MKLDNN_CVT(FP16, U64),  MKLDNN_CVT(FP16, I64),  \
    MKLDNN_CVT(FP16, FP32), MKLDNN_CVT(FP16, BF16), MKLDNN_CVT(FP16, FP64), MKLDNN_CVT(FP16, BOOL), \
    MKLDNN_CVT(BF16, U8),   MKLDNN_CVT(BF16, I8),   MKLDNN_CVT(BF16, U16),  MKLDNN_CVT(BF16, I16),  \
    MKLDNN_CVT(BF16, U32),  MKLDNN_CVT(BF16, I32),  MKLDNN_CVT(BF16, U64),  MKLDNN_CVT(BF16, I64),  \
    MKLDNN_CVT(BF16, FP32), MKLDNN_CVT(BF16, FP16), MKLDNN_CVT(BF16, FP64), MKLDNN_CVT(BF16, BOOL), \
    MKLDNN_CVT(FP64, U8),   MKLDNN_CVT(FP64, I8),   MKLDNN_CVT(FP64, U16),  MKLDNN_CVT(FP64, I16),  \
    MKLDNN_CVT(FP64, U32),  MKLDNN_CVT(FP64, I32),  MKLDNN_CVT(FP64, U64),  MKLDNN_CVT(FP64, I64),  \
    MKLDNN_CVT(FP64, FP32), MKLDNN_CVT(FP64, FP16), MKLDNN_CVT(FP64, BF16), MKLDNN_CVT(FP64, BOOL), \
    MKLDNN_CVT(BOOL, U8),   MKLDNN_CVT(BOOL, I8),   MKLDNN_CVT(BOOL, U16),  MKLDNN_CVT(BOOL, I16),  \
    MKLDNN_CVT(BOOL, U32),  MKLDNN_CVT(BOOL, I32),  MKLDNN_CVT(BOOL, U64),  MKLDNN_CVT(BOOL, I64),  \
    MKLDNN_CVT(BOOL, FP32), MKLDNN_CVT(BOOL, FP16), MKLDNN_CVT(BOOL, BF16), MKLDNN_CVT(BOOL, FP64)

#define MKLDNN_TRUNC_LIST                                                                           \
    MKLDNN_CVT(U8, U8),     MKLDNN_CVT(I8, I8),     MKLDNN_CVT(U16, U16),   MKLDNN_CVT(I16, I16),   \
    MKLDNN_CVT(U32, U32),   MKLDNN_CVT(I32, I32),   MKLDNN_CVT(U64, U64),   MKLDNN_CVT(I64, I64),   \
    MKLDNN_CVT(FP32, FP32), MKLDNN_CVT(FP16, FP16), MKLDNN_CVT(BF16, BF16), MKLDNN_CVT(FP64, FP64), \
    MKLDNN_CVT(BOOL, BOOL)

void cpu_convert(const void *srcPtr, void *dstPtr, Precision srcPrc, Precision dstPrc, const size_t size) {
    if (srcPtr == nullptr || dstPtr == nullptr)
        IE_THROW() << "cpu_convert has null data pointer";

    if (srcPrc == dstPrc) {
        cpu_memcpy(dstPtr, srcPtr, size*dstPrc.size());
    } else {
        ConvertContext ctx = { srcPtr, dstPtr, size, false };
        OV_SWITCH(MKLDNNPlugin, ConvertPrecision, ctx, std::tie(srcPrc, dstPrc), MKLDNN_CVT_LIST);
        if (!ctx.converted)
            IE_THROW() << "cpu_convert can't convert from: " << srcPrc << " precision to: " << dstPrc;
    }
}

void cpu_convert(const void *srcPtr,
                 void *dstPtr,
                 InferenceEngine::Precision srcPrc,
                 InferenceEngine::Precision interimPrc,
                 InferenceEngine::Precision dstPrc,
                 const size_t size) {
    if (!isConversionTruncatesRange(srcPrc, interimPrc)) {
        cpu_convert(srcPtr, dstPtr, srcPrc, dstPrc, size);
    } else {
        if (srcPtr == nullptr || dstPtr == nullptr)
            IE_THROW() << "cpu_convert has null data pointer";
        TruncateContext ctx = {
            srcPtr,
            dstPtr,
            size,
            interimPrc,
            false
        };
        OV_SWITCH(MKLDNNPlugin, TruncatePrecision, ctx, std::tie(srcPrc, dstPrc), MKLDNN_CVT_LIST);

        if (!ctx.converted)
            OV_SWITCH(MKLDNNPlugin, TruncatePrecision, ctx, std::tie(srcPrc, dstPrc), MKLDNN_TRUNC_LIST);

        if (!ctx.converted)
            IE_THROW() << "cpu_convert can't convert from: " << srcPrc << " precision to: " << dstPrc;
    }
}

#undef MKLDNN_CVT
#undef MKLDNN_CVT_LIST
