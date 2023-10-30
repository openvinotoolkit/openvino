// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_convert.h"
#include "cpu_memcpy.h"
#include <ie_parallel.hpp>
#include <utils/bfloat16.hpp>
#include <utils/general_utils.h>
#include <selective_build.h>
#include <openvino/core/type/float16.hpp>
#include <algorithm>
#include <type_traits>
#include <tuple>
#include <cmath>
#include <onednn/dnnl.h>
#if defined(OPENVINO_ARCH_X86_64)
#include "nodes/kernels/x64/jit_kernel.hpp"
#include <cpu/x64/jit_generator.hpp>
#endif

using namespace InferenceEngine;


namespace ov {
namespace intel_cpu {
namespace {

#if defined(OPENVINO_ARCH_X86_64)

using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

template <typename src_t, typename dst_t>
void convert_vec(jit_generator & gen,
                 const RegExp & src,
                 const RegExp & dst);

template <>
void convert_vec<ov::float16, float>(jit_generator & gen,
                                     const RegExp & src,
                                     const RegExp & dst) {
    auto const & f16vec = gen.xmm3;
    auto const & f32vec = gen.ymm4;

    gen.movdqu(f16vec, gen.xword[src]);
    gen.vcvtph2ps(f32vec, f16vec);
    gen.vmovups(gen.yword[dst], f32vec);
}

template <>
void convert_vec<float, ov::float16>(jit_generator & gen,
                                     const RegExp & src,
                                     const RegExp & dst) {
    auto const & f16vec = gen.xmm3;
    auto const & f32vec = gen.ymm4;

    gen.vmovups(f32vec, gen.yword[src]);
    gen.vcvtps2ph(f16vec, f32vec, 0);
    gen.movdqu(gen.xword[dst], f16vec);
}

class jit_convert_array : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_convert_array)

    void generate() override {
        constexpr size_t vlen = 8u;
        constexpr size_t vlen_log2 = 3;

        preamble();

        // Get arguments addresses
        auto src = arg(&args_t::src);
        auto dst = arg(&args_t::out);
        auto size = arg(&args_t::count);

        size >>= vlen_log2;

        foreach(0, size, [&, this](const Xbyak::Reg64& idx) {
            _convert_vec(*this, src, dst);
            src += _src_size * vlen;
            dst += _dst_size * vlen;
        });

        mov(size, argPtr(&args_t::count));
        size &= vlen - 1;

        // Tail conversion
        _if(size != 0)
        ._then([&] {
            auto tmp = stack(vlen * sizeof(float));
            tmp.clear();

            auto tail_size = var<size_t>();

            tail_size = size;
            tail_size <<= static_cast<size_t>(std::logb(_src_size)) - 1;
            copy<uint16_t>(tmp.pointer(), src, tail_size);

            _convert_vec(*this, tmp.pointer(), tmp.pointer());

            tail_size = size;
            tail_size <<= static_cast<size_t>(std::logb(_dst_size)) - 1;
            copy<uint16_t>(dst, tmp.pointer(), tail_size);
        });

        postamble();
    }

public:
    typedef struct {
        const void* src;
        void* out;
        const size_t count;
    } args_t;

    typedef void (*fn_t)(const args_t*);

    typedef void (*convert_vec_t)(jit_generator &,
                                  const RegExp &,
                                  const RegExp &);

    jit_convert_array(convert_vec_t convert_vec,
                      size_t src_size,
                      size_t dst_size)
        : jit_kernel(jit_name())
        , _convert_vec(convert_vec)
        , _src_size(src_size)
        , _dst_size(dst_size) {}

    template<typename src_t, typename dst_t>
    static fn_t get() {
        if (mayiuse(cpu_isa_t::avx2)
            && dnnl::impl::cpu::x64::cpu().has(Xbyak::util::Cpu::tF16C)) {
            static jit_convert_array converter(convert_vec<src_t, dst_t>, sizeof(src_t), sizeof(dst_t));
            auto & generator = static_cast<jit_generator&>(converter);
            generator.create_kernel();
            return (fn_t)generator.jit_ker();
        }
        return nullptr;
    }

private:
    convert_vec_t _convert_vec;
    size_t _src_size;
    size_t _dst_size;
};

template <typename TI, typename TO>
void jit_convert(const TI* arg, TO* out, size_t count) {
    using jit_impl = jit_convert_array;
    static auto converter = jit_impl::get<TI, TO>();

    if (converter) {
        typename jit_impl::args_t args = { arg, out, count };
        converter(&args);
    } else {
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<TO>(arg[i]);
        }
    }
}

#endif

template <Precision::ePrecision p>
struct PrecisionInfo {
    using value_type = typename PrecisionTrait<p>::value_type;
};

template <>
struct PrecisionInfo<Precision::BF16> {
    using value_type = ov::intel_cpu::bfloat16_t;
};

template <>
struct PrecisionInfo<Precision::FP16> {
    using value_type = ov::float16;
};

template <>
struct PrecisionInfo<Precision::BOOL> {
    using value_type = uint8_t;
};

template<typename T,
         typename U = typename std::conditional<
                        std::is_same<ov::float16, T>::value
                        || std::is_same<ov::intel_cpu::bfloat16_t, T>::value,
                        float, T>::type>
struct Range {
    const std::tuple<U, U> & fit(const Precision & prec);

private:
    std::tuple<U, U> _range {
        std::numeric_limits<T>::lowest(),
        std::numeric_limits<T>::max()
    };
};

template<typename T, typename U>
const std::tuple<U, U> & Range<T, U>::fit(const Precision & prec) {
    if (prec.is_float()) {
        double lbound, ubound;
        switch (prec) {
            case Precision::BF16:
                lbound = static_cast<double>(std::numeric_limits<ov::intel_cpu::bfloat16_t>::lowest());
                ubound = static_cast<double>(std::numeric_limits<ov::intel_cpu::bfloat16_t>::max());
                break;
            case Precision::FP16:
                lbound = static_cast<double>(std::numeric_limits<ov::float16>::lowest());
                ubound = static_cast<double>(std::numeric_limits<ov::float16>::max());
                break;
            case Precision::FP32:
                lbound = static_cast<double>(std::numeric_limits<float>::lowest());
                ubound = static_cast<double>(std::numeric_limits<float>::max());
                break;
            case Precision::FP64:
                lbound = std::numeric_limits<double>::lowest();
                ubound = std::numeric_limits<double>::max();
                break;
            default:
                OPENVINO_THROW("Unsupported precision");
        }
        // If U is integral, its range always less than float, so not need update _range
        // Else it will be overflow, for example static_cast double to int64_t:
        //         int64_t ubound = 9223372036854775807
        //         double  dd_ubound = static_cast<double>(ubbound)
        //         static_cast<int64_t>(dd_ubound) will return -9223372036854775808
        if (!std::is_integral<U>::value) {
                std::get<0>(_range) = static_cast<U>(std::max(static_cast<double>(std::get<0>(_range)), lbound));
                std::get<1>(_range) = static_cast<U>(std::min(static_cast<double>(std::get<1>(_range)), ubound));
        }
    } else {
        int64_t lbound;
        uint64_t ubound;
        switch (prec) {
            case Precision::BOOL:
            case Precision::U8:
                lbound = static_cast<int64_t>(std::numeric_limits<uint8_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<uint8_t>::max());
                break;
            case Precision::I8:
                lbound = static_cast<int64_t>(std::numeric_limits<int8_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<int8_t>::max());
                break;
            case Precision::U16:
                lbound = static_cast<int64_t>(std::numeric_limits<uint16_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<uint16_t>::max());
                break;
            case Precision::I16:
                lbound = static_cast<int64_t>(std::numeric_limits<int16_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<int16_t>::max());
                break;
            case Precision::U32:
                lbound = static_cast<int64_t>(std::numeric_limits<uint32_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
                break;
            case Precision::I32:
                lbound = static_cast<int64_t>(std::numeric_limits<int32_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
                break;
            case Precision::U64:
                lbound = static_cast<int64_t>(std::numeric_limits<uint64_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<uint64_t>::max());
                break;
            case Precision::I64:
                lbound = static_cast<int64_t>(std::numeric_limits<int64_t>::lowest());
                ubound = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
                break;
            default:
                OPENVINO_THROW("Unsupported precision");
        }
        using ltype = typename std::conditional<
                            std::is_floating_point<U>::value,
                            double, int64_t>::type;
        using utype = typename std::conditional<
                            std::is_floating_point<U>::value,
                            double, uint64_t>::type;
        std::get<0>(_range) = static_cast<U>(std::max(static_cast<ltype>(std::get<0>(_range)), static_cast<ltype>(lbound)));
        std::get<1>(_range) = static_cast<U>(std::min(static_cast<utype>(std::get<1>(_range)), static_cast<utype>(ubound)));
    }
    return _range;
}

struct ConvertContext {
    const void *srcPtr;
    void *dstPtr;
    size_t size;
    Precision interimPrc;
    Precision dstPrc;
    bool converted;

    template<typename T>
    std::tuple<T, T> range() const {
        Range<T> r;
        r.fit(interimPrc);
        return r.fit(dstPrc);
    }
};

template<typename T>
struct ConvertPrecision;

template<typename src_t, typename dst_t>
struct ConvertPrecision<std::tuple<src_t, dst_t>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const src_t *>(ctx.srcPtr);
        auto dst = static_cast<dst_t *>(ctx.dstPtr);
        src_t lbound, ubound;
        std::tie(lbound, ubound) = ctx.range<src_t>();

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

template<>
struct ConvertPrecision<std::tuple<float, ov::intel_cpu::bfloat16_t>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const float *>(ctx.srcPtr);
        auto dst = static_cast<ov::intel_cpu::bfloat16_t *>(ctx.dstPtr);

        if (ctx.interimPrc.is_float()) {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<ov::intel_cpu::bfloat16_t>(src[i]);
            });
        } else {
            float lbound, ubound;
            std::tie(lbound, ubound) = ctx.range<float>();
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<ov::intel_cpu::bfloat16_t>(std::trunc(std::max(std::min(src[i], ubound), lbound)));
            });
        }

        ctx.converted = true;
    }
};

template<>
struct ConvertPrecision<std::tuple<ov::intel_cpu::bfloat16_t, float>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const ov::intel_cpu::bfloat16_t *>(ctx.srcPtr);
        auto dst = static_cast<float *>(ctx.dstPtr);

        if (ctx.interimPrc.is_float()) {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<float>(src[i]);
            });
        } else {
            float lbound, ubound;
            std::tie(lbound, ubound) = ctx.range<ov::intel_cpu::bfloat16_t>();
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = std::trunc(std::max(std::min(static_cast<float>(src[i]), ubound), lbound));
            });
        }

        ctx.converted = true;
    }
};

#if defined(OPENVINO_ARCH_X86_64)
template<typename src_t>
struct ConvertPrecision<std::tuple<src_t, ov::float16>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const src_t *>(ctx.srcPtr);
        auto dst = static_cast<ov::float16 *>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);
        typedef float batch_type[batch];

        src_t lbound, ubound;
        std::tie(lbound, ubound) = ctx.range<src_t>();

        if (std::is_integral<src_t>::value
            || ctx.interimPrc.is_float()) {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                for (size_t j = 0; j < current_batch_size; ++j)         // src_t -> fp32
                    tmp[j] = static_cast<float>(std::max(std::min(src[offset + j], ubound), lbound));
                jit_convert(tmp, dst + offset, current_batch_size);     // fp32 -> fp16
            });
        } else {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                for (size_t j = 0; j < current_batch_size; ++j)         // src_t -> fp32
                    tmp[j] = static_cast<float>(std::trunc(std::max(std::min(src[offset + j], ubound), lbound)));
                jit_convert(tmp, dst + offset, current_batch_size);     // fp32 -> fp16
            });
        }

        ctx.converted = true;
    }
};

template<typename dst_t>
struct ConvertPrecision<std::tuple<ov::float16, dst_t>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const ov::float16 *>(ctx.srcPtr);
        auto dst = static_cast<dst_t *>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);
        typedef float batch_type[batch];

        float lbound, ubound;
        std::tie(lbound, ubound) = ctx.range<ov::float16>();

        if (ctx.interimPrc.is_float()
            || std::is_integral<dst_t>::value) {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, tmp, current_batch_size);     // fp16 -> fp32
                for (size_t j = 0; j < current_batch_size; ++j)         // fp32 -> dst_t
                    dst[offset + j] = static_cast<dst_t>(std::max(std::min(tmp[j], ubound), lbound));
            });
        } else {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, tmp, current_batch_size);     // fp16 -> fp32
                for (size_t j = 0; j < current_batch_size; ++j)         // fp32 -> dst_t
                    dst[offset + j] = static_cast<dst_t>(std::trunc(std::max(std::min(tmp[j], ubound), lbound)));
            });
        }

        ctx.converted = true;
    }
};

template<>
struct ConvertPrecision<std::tuple<ov::float16, ov::float16>> {
    void operator()(ConvertContext & ctx) {
        auto src = static_cast<const ov::float16 *>(ctx.srcPtr);
        auto dst = static_cast<ov::float16 *>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);
        typedef float batch_type[batch];

        float lbound, ubound;
        std::tie(lbound, ubound) = ctx.range<ov::float16>();

        if (ctx.interimPrc.is_float()) {
            cpu_memcpy(dst, src, ctx.size * sizeof(ov::float16));
        } else {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, tmp, current_batch_size);     // fp16 -> fp32
                for (size_t j = 0; j < current_batch_size; ++j)         // truncate fp32
                    tmp[j] = std::trunc(std::max(std::min(tmp[j], ubound), lbound));
                jit_convert(tmp, dst + offset, current_batch_size);     // fp32 -> fp16
            });
        }

        ctx.converted = true;
    }
};
#endif

}   // namespace

#define INTEL_CPU_CVT(ST, DT) OV_CASE2(Precision::ST, Precision::DT, PrecisionInfo<Precision::ST>::value_type, PrecisionInfo<Precision::DT>::value_type)

#define INTEL_CPU_CVT_LIST                                                                                      \
    INTEL_CPU_CVT(U8, I8),     INTEL_CPU_CVT(U8, U16),    INTEL_CPU_CVT(U8, I16),    INTEL_CPU_CVT(U8, U32),    \
    INTEL_CPU_CVT(U8, I32),    INTEL_CPU_CVT(U8, U64),    INTEL_CPU_CVT(U8, I64),    INTEL_CPU_CVT(U8, FP32),   \
    INTEL_CPU_CVT(U8, FP16),   INTEL_CPU_CVT(U8, BF16),   INTEL_CPU_CVT(U8, FP64),   INTEL_CPU_CVT(U8, BOOL),   \
    INTEL_CPU_CVT(I8, U8),     INTEL_CPU_CVT(I8, U16),    INTEL_CPU_CVT(I8, I16),    INTEL_CPU_CVT(I8, U32),    \
    INTEL_CPU_CVT(I8, I32),    INTEL_CPU_CVT(I8, U64),    INTEL_CPU_CVT(I8, I64),    INTEL_CPU_CVT(I8, FP32),   \
    INTEL_CPU_CVT(I8, FP16),   INTEL_CPU_CVT(I8, BF16),   INTEL_CPU_CVT(I8, FP64),   INTEL_CPU_CVT(I8, BOOL),   \
    INTEL_CPU_CVT(U16, U8),    INTEL_CPU_CVT(U16, I8),    INTEL_CPU_CVT(U16, I16),   INTEL_CPU_CVT(U16, U32),   \
    INTEL_CPU_CVT(U16, I32),   INTEL_CPU_CVT(U16, U64),   INTEL_CPU_CVT(U16, I64),   INTEL_CPU_CVT(U16, FP32),  \
    INTEL_CPU_CVT(U16, FP16),  INTEL_CPU_CVT(U16, BF16),  INTEL_CPU_CVT(U16, FP64),  INTEL_CPU_CVT(U16, BOOL),  \
    INTEL_CPU_CVT(I16, U8),    INTEL_CPU_CVT(I16, I8),    INTEL_CPU_CVT(I16, U16),   INTEL_CPU_CVT(I16, U32),   \
    INTEL_CPU_CVT(I16, I32),   INTEL_CPU_CVT(I16, U64),   INTEL_CPU_CVT(I16, I64),   INTEL_CPU_CVT(I16, FP32),  \
    INTEL_CPU_CVT(I16, FP16),  INTEL_CPU_CVT(I16, BF16),  INTEL_CPU_CVT(I16, FP64),  INTEL_CPU_CVT(I16, BOOL),  \
    INTEL_CPU_CVT(U32, U8),    INTEL_CPU_CVT(U32, I8),    INTEL_CPU_CVT(U32, U16),   INTEL_CPU_CVT(U32, I16),   \
    INTEL_CPU_CVT(U32, I32),   INTEL_CPU_CVT(U32, U64),   INTEL_CPU_CVT(U32, I64),   INTEL_CPU_CVT(U32, FP32),  \
    INTEL_CPU_CVT(U32, FP16),  INTEL_CPU_CVT(U32, BF16),  INTEL_CPU_CVT(U32, FP64),  INTEL_CPU_CVT(U32, BOOL),  \
    INTEL_CPU_CVT(I32, U8),    INTEL_CPU_CVT(I32, I8),    INTEL_CPU_CVT(I32, U16),   INTEL_CPU_CVT(I32, I16),   \
    INTEL_CPU_CVT(I32, U32),   INTEL_CPU_CVT(I32, U64),   INTEL_CPU_CVT(I32, I64),   INTEL_CPU_CVT(I32, FP32),  \
    INTEL_CPU_CVT(I32, FP16),  INTEL_CPU_CVT(I32, BF16),  INTEL_CPU_CVT(I32, FP64),  INTEL_CPU_CVT(I32, BOOL),  \
    INTEL_CPU_CVT(U64, U8),    INTEL_CPU_CVT(U64, I8),    INTEL_CPU_CVT(U64, U16),   INTEL_CPU_CVT(U64, I16),   \
    INTEL_CPU_CVT(U64, U32),   INTEL_CPU_CVT(U64, I32),   INTEL_CPU_CVT(U64, I64),   INTEL_CPU_CVT(U64, FP32),  \
    INTEL_CPU_CVT(U64, FP16),  INTEL_CPU_CVT(U64, BF16),  INTEL_CPU_CVT(U64, FP64),  INTEL_CPU_CVT(U64, BOOL),  \
    INTEL_CPU_CVT(I64, U8),    INTEL_CPU_CVT(I64, I8),    INTEL_CPU_CVT(I64, U16),   INTEL_CPU_CVT(I64, I16),   \
    INTEL_CPU_CVT(I64, U32),   INTEL_CPU_CVT(I64, I32),   INTEL_CPU_CVT(I64, U64),   INTEL_CPU_CVT(I64, FP32),  \
    INTEL_CPU_CVT(I64, FP16),  INTEL_CPU_CVT(I64, BF16),  INTEL_CPU_CVT(I64, FP64),  INTEL_CPU_CVT(I64, BOOL),  \
    INTEL_CPU_CVT(FP32, U8),   INTEL_CPU_CVT(FP32, I8),   INTEL_CPU_CVT(FP32, U16),  INTEL_CPU_CVT(FP32, I16),  \
    INTEL_CPU_CVT(FP32, U32),  INTEL_CPU_CVT(FP32, I32),  INTEL_CPU_CVT(FP32, U64),  INTEL_CPU_CVT(FP32, I64),  \
    INTEL_CPU_CVT(FP32, FP16), INTEL_CPU_CVT(FP32, BF16), INTEL_CPU_CVT(FP32, FP64), INTEL_CPU_CVT(FP32, BOOL), \
    INTEL_CPU_CVT(FP16, U8),   INTEL_CPU_CVT(FP16, I8),   INTEL_CPU_CVT(FP16, U16),  INTEL_CPU_CVT(FP16, I16),  \
    INTEL_CPU_CVT(FP16, U32),  INTEL_CPU_CVT(FP16, I32),  INTEL_CPU_CVT(FP16, U64),  INTEL_CPU_CVT(FP16, I64),  \
    INTEL_CPU_CVT(FP16, FP32), INTEL_CPU_CVT(FP16, BF16), INTEL_CPU_CVT(FP16, FP64), INTEL_CPU_CVT(FP16, BOOL), \
    INTEL_CPU_CVT(BF16, U8),   INTEL_CPU_CVT(BF16, I8),   INTEL_CPU_CVT(BF16, U16),  INTEL_CPU_CVT(BF16, I16),  \
    INTEL_CPU_CVT(BF16, U32),  INTEL_CPU_CVT(BF16, I32),  INTEL_CPU_CVT(BF16, U64),  INTEL_CPU_CVT(BF16, I64),  \
    INTEL_CPU_CVT(BF16, FP32), INTEL_CPU_CVT(BF16, FP16), INTEL_CPU_CVT(BF16, FP64), INTEL_CPU_CVT(BF16, BOOL), \
    INTEL_CPU_CVT(FP64, U8),   INTEL_CPU_CVT(FP64, I8),   INTEL_CPU_CVT(FP64, U16),  INTEL_CPU_CVT(FP64, I16),  \
    INTEL_CPU_CVT(FP64, U32),  INTEL_CPU_CVT(FP64, I32),  INTEL_CPU_CVT(FP64, U64),  INTEL_CPU_CVT(FP64, I64),  \
    INTEL_CPU_CVT(FP64, FP32), INTEL_CPU_CVT(FP64, FP16), INTEL_CPU_CVT(FP64, BF16), INTEL_CPU_CVT(FP64, BOOL), \
    INTEL_CPU_CVT(BOOL, U8),   INTEL_CPU_CVT(BOOL, I8),   INTEL_CPU_CVT(BOOL, U16),  INTEL_CPU_CVT(BOOL, I16),  \
    INTEL_CPU_CVT(BOOL, U32),  INTEL_CPU_CVT(BOOL, I32),  INTEL_CPU_CVT(BOOL, U64),  INTEL_CPU_CVT(BOOL, I64),  \
    INTEL_CPU_CVT(BOOL, FP32), INTEL_CPU_CVT(BOOL, FP16), INTEL_CPU_CVT(BOOL, BF16), INTEL_CPU_CVT(BOOL, FP64), \
    INTEL_CPU_CVT(U8, U8),     INTEL_CPU_CVT(I8, I8),     INTEL_CPU_CVT(U16, U16),   INTEL_CPU_CVT(I16, I16),   \
    INTEL_CPU_CVT(U32, U32),   INTEL_CPU_CVT(I32, I32),   INTEL_CPU_CVT(U64, U64),   INTEL_CPU_CVT(I64, I64),   \
    INTEL_CPU_CVT(FP32, FP32), INTEL_CPU_CVT(FP16, FP16), INTEL_CPU_CVT(BF16, BF16), INTEL_CPU_CVT(FP64, FP64), \
    INTEL_CPU_CVT(BOOL, BOOL)

#define INTEL_CPU_CVT_FROM_BIN(DT) OV_CASE(Precision::DT, PrecisionInfo<Precision::DT>::value_type)

#define INTEL_CPU_CVT_FROM_BIN_LIST                                                                 \
    INTEL_CPU_CVT_FROM_BIN(FP32), INTEL_CPU_CVT_FROM_BIN(FP16), INTEL_CPU_CVT_FROM_BIN(BF16),       \
    INTEL_CPU_CVT_FROM_BIN(FP64), INTEL_CPU_CVT_FROM_BIN(I16), INTEL_CPU_CVT_FROM_BIN(U8),          \
    INTEL_CPU_CVT_FROM_BIN(I8), INTEL_CPU_CVT_FROM_BIN(U16), INTEL_CPU_CVT_FROM_BIN(I32),           \
    INTEL_CPU_CVT_FROM_BIN(U32), INTEL_CPU_CVT_FROM_BIN(I64), INTEL_CPU_CVT_FROM_BIN(U64),          \
    INTEL_CPU_CVT_FROM_BIN(BOOL)

struct ConvertFromBinContext {
    const void *srcPtr;
    void *dstPtr;
    size_t size;
    bool converted;
};

template<typename T>
struct ConvertFromBinPrecision {
    void operator()(ConvertFromBinContext &ctx) {
        auto src = static_cast<const uint8_t *>(ctx.srcPtr);
        auto dst = static_cast<T *>(ctx.dstPtr);
        const size_t nBits = 8;
        const size_t nBytes = rnd_up(ctx.size, nBits);
        parallel_for(nBytes, [&](size_t byteIndex) {
            auto currentBitNum = std::min(nBits, ctx.size - byteIndex * nBits);
            for (size_t bitIndex = 0; bitIndex < currentBitNum; ++bitIndex) {
                dst[byteIndex * nBits + bitIndex] = static_cast<T>((src[byteIndex] & (1 << bitIndex)) >> bitIndex);
            }
        });
        ctx.converted = true;
    }
};


void cpu_convert(const void *srcPtr, void *dstPtr, Precision srcPrc, Precision dstPrc, const size_t size) {
    cpu_convert(srcPtr, dstPtr, srcPrc, dstPrc, dstPrc, size);
}

void cpu_convert(const void *srcPtr,
                 void *dstPtr,
                 InferenceEngine::Precision srcPrc,
                 InferenceEngine::Precision interimPrc,
                 InferenceEngine::Precision dstPrc,
                 const size_t size) {
    if (srcPtr == nullptr || dstPtr == nullptr)
        OPENVINO_THROW("cpu_convert has null data pointer");

    if (srcPrc == dstPrc && srcPrc == interimPrc) {
        const size_t L2_cache_size = dnnl::utils::get_cache_size(2, true);
        const size_t totalSize = size * dstPrc.size();
        if (totalSize >= L2_cache_size) {
            auto src = static_cast<const uint8_t *>(srcPtr);
            auto dst = static_cast<uint8_t *>(dstPtr);
            parallel_nt(0, [&](const size_t ithr, const size_t nthr) {
                size_t start = 0, end = 0;
                splitter(totalSize, nthr, ithr, start, end);
                cpu_memcpy(dst + start, src + start, end - start);
            });
        } else {
            cpu_memcpy(dstPtr, srcPtr, size * dstPrc.size());
        }
    } else if (srcPrc == Precision::BIN) {
        if (srcPrc.bitsSize() != 1)
            OPENVINO_THROW("cpu_convert can't convert from: ",
                           srcPrc,
                           " <bitsSize == ",
                           srcPrc.bitsSize(),
                           "> precision to: ",
                           dstPrc,
                           ". Not implemented.");
        ConvertFromBinContext ctx {
                srcPtr,
                dstPtr,
                size,
                false
        };
        OV_SWITCH(intel_cpu, ConvertFromBinPrecision, ctx, dstPrc, INTEL_CPU_CVT_FROM_BIN_LIST);
        if (!ctx.converted)
            OPENVINO_THROW("cpu_convert can't convert from: ",
                           srcPrc,
                           " <bitsSize == ",
                           srcPrc.bitsSize(),
                           "> precision to: ",
                           dstPrc);
    } else {
        ConvertContext ctx {
            srcPtr,
            dstPtr,
            size,
            interimPrc,
            dstPrc,
            false
        };
        OV_SWITCH(intel_cpu, ConvertPrecision, ctx, std::tie(srcPrc, dstPrc), INTEL_CPU_CVT_LIST);
        if (!ctx.converted)
            OPENVINO_THROW("cpu_convert can't convert from: ", srcPrc, " precision to: ", dstPrc);
    }
}

#undef INTEL_CPU_CVT
#undef INTEL_CPU_CVT_LIST

}   // namespace intel_cpu
}   // namespace ov
