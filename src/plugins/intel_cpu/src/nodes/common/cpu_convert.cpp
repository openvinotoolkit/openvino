// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_convert.h"

#include "cpu_memcpy.h"
#include "openvino/core/type/nf4.hpp"
#include "utils/bfloat16.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/jit_avx512_core_fp8cvt.hpp"
#    include "nodes/kernels/x64/jit_kernel.hpp"
#else
#    include "cpu_memory.h"
#    include "openvino/core/type/element_type_traits.hpp"
#    include "selective_build.h"
#    include "utils/general_utils.h"
#endif

namespace ov::intel_cpu {
namespace {

#if defined(OPENVINO_ARCH_X86_64)

using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

enum f8_type : uint8_t { none, f8e4m3, f8e5m2 };

template <typename src_t, typename dst_t>
f8_type get_f8_type() {
    if (std::is_same_v<src_t, ov::float8_e4m3> || std::is_same_v<dst_t, ov::float8_e4m3>) {
        return f8_type::f8e4m3;
    }
    if (std::is_same_v<src_t, ov::float8_e5m2> || std::is_same_v<dst_t, ov::float8_e5m2>) {
        return f8_type::f8e5m2;
    }
    return f8_type::none;
}

template <typename src_t, typename dst_t>
void convert_vec(jit_generator& gen, const RegExp& src, const RegExp& dst);

template <>
void convert_vec<ov::float16, float>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f16vec = gen.xmm3;
    auto const& f32vec = gen.ymm4;

    gen.movdqu(f16vec, gen.xword[src]);
    gen.vcvtph2ps(f32vec, f16vec);
    gen.vmovups(gen.yword[dst], f32vec);
}

template <>
void convert_vec<float, ov::float16>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f16vec = gen.xmm3;
    auto const& f32vec = gen.ymm4;

    gen.vmovups(f32vec, gen.yword[src]);
    gen.vcvtps2ph(f16vec, f32vec, 0);
    gen.movdqu(gen.xword[dst], f16vec);
}

template <typename src_t, typename dst_t>
class jit_convert_array : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_convert_array)

    void generate() override {
        bool is_fp8 = f8_e4m3_emu_ || f8_e5m2_emu_;
        size_t vlen = is_fp8 ? 16u : 8u;
        size_t vlen_log2 = is_fp8 ? 4 : 3;

        preamble();

        // Get arguments addresses
        auto src = arg(&args_t::src);
        auto dst = arg(&args_t::out);
        auto size = arg(&args_t::count);

        size >>= vlen_log2;

        foreach (0, size, [&, this](const Xbyak::Reg64& idx) {
            _convert_vec(*this, src, dst);
            src += _src_size * vlen;
            dst += _dst_size * vlen;
        })
            ;

        mov(size, argPtr(&args_t::count));
        size &= vlen - 1;

        // Tail conversion
        _if(size != 0)._then([&] {
            auto tmp = stack(vlen * sizeof(float));
            tmp.clear();

            auto tail_size = var<size_t>();

            tail_size = size;
            tail_size <<= static_cast<size_t>(std::logb(_src_size));
            copy<uint8_t>(tmp.pointer(), src, tail_size);

            _convert_vec(*this, tmp.pointer(), tmp.pointer());

            tail_size = size;
            tail_size <<= static_cast<size_t>(std::logb(_dst_size));
            copy<uint8_t>(dst, tmp.pointer(), tail_size);
        });

        postamble();

        if (f8_e4m3_emu_) {
            f8_e4m3_emu_->prepare_table();
        }
        if (f8_e5m2_emu_) {
            f8_e5m2_emu_->prepare_table();
        }
        if (uni_vcvtneps2bf16_) {
            uni_vcvtneps2bf16_->emit_data();
        }
    }

public:
    using args_t = struct {
        const void* src;
        void* out;
        const size_t count;
    };

    using fn_t = void (*)(const args_t*);

    using convert_vec_t = void (*)(jit_generator&, const RegExp&, const RegExp&);

    jit_convert_array(convert_vec_t convert_vec)
        : jit_kernel(jit_name()),
          _convert_vec(convert_vec),
          _src_size(sizeof(src_t)),
          _dst_size(sizeof(dst_t)) {
        const auto type = get_f8_type<src_t, dst_t>();
        if (type == f8_type::f8e4m3) {
            f8_e4m3_emu_ = std::make_shared<fp8_emulation_e4m3_t>(this,
                                                                  fp8_emu_reserv_1_,
                                                                  fp8_emu_reserv_2_,
                                                                  fp8_emu_reserv_3_,
                                                                  fp8_emu_reserv_4_,
                                                                  fp8_emu_reserv_5_,
                                                                  fp8_emu_scratch_);
        } else if (type == f8_type::f8e5m2) {
            f8_e5m2_emu_ = std::make_shared<fp8_emulation_e5m2_t>(this,
                                                                  fp8_emu_reserv_1_,
                                                                  fp8_emu_reserv_2_,
                                                                  fp8_emu_reserv_3_,
                                                                  fp8_emu_kmask_aux_,
                                                                  fp8_emu_scratch_);
        }
        const bool is_dst_bf16 = std::is_same<dst_t, ov::intel_cpu::bfloat16_t>::value;
        if (is_dst_bf16 && mayiuse(cpu_isa_t::avx512_core)) {
            uni_vcvtneps2bf16_ = std::make_shared<jit_uni_vcvtneps2bf16>(this, cpu_isa_t::avx512_core);
        }
    }

    static fn_t get() {
        if (mayiuse(cpu_isa_t::avx2) && dnnl::impl::cpu::x64::cpu().has(Xbyak::util::Cpu::tF16C)) {
            static jit_convert_array converter(convert_vec<src_t, dst_t>);
            auto& generator = static_cast<jit_generator&>(converter);
            generator.create_kernel();
            return (fn_t)generator.jit_ker();
        }
        return nullptr;
    }

    std::shared_ptr<fp8_emulation_e4m3_t> get_f8_e4m3_emu() const {
        return f8_e4m3_emu_;
    }

    std::shared_ptr<fp8_emulation_e5m2_t> get_f8_e5m2_emu() const {
        return f8_e5m2_emu_;
    }

    std::shared_ptr<jit_uni_vcvtneps2bf16> get_uni_vcvtneps2bf16() const {
        return uni_vcvtneps2bf16_;
    }

private:
    convert_vec_t _convert_vec;
    size_t _src_size;
    size_t _dst_size;

    std::shared_ptr<fp8_emulation_e4m3_t> f8_e4m3_emu_;
    std::shared_ptr<fp8_emulation_e5m2_t> f8_e5m2_emu_;
    std::shared_ptr<jit_uni_vcvtneps2bf16> uni_vcvtneps2bf16_;

    const Reg64 fp8_emu_scratch_ = rax;
    const Zmm fp8_emu_reserv_1_ = Zmm(9);
    const Zmm fp8_emu_reserv_2_ = Zmm(10);
    const Zmm fp8_emu_reserv_3_ = Zmm(11);
    const Zmm fp8_emu_reserv_4_ = Zmm(12);
    const Zmm fp8_emu_reserv_5_ = Zmm(13);
    const Opmask fp8_emu_kmask_aux_ = Opmask(1);
};

template <>
void convert_vec<float, ov::float8_e4m3>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f8vec = gen.xmm3;
    auto const& f32vec = gen.zmm4;

    auto& cvt = dynamic_cast<jit_convert_array<float, ov::float8_e4m3>&>(gen);

    gen.vmovups(f32vec, gen.zword[src]);
    cvt.get_f8_e4m3_emu()->vcvt_f32_to_f8(f8vec, f32vec);
    gen.vmovdqu(gen.xword[dst], f8vec);
}

template <>
void convert_vec<ov::float8_e4m3, float>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f8vec = gen.xmm3;
    auto const& f32vec = gen.zmm4;

    auto& cvt = dynamic_cast<jit_convert_array<ov::float8_e4m3, float>&>(gen);

    gen.vmovdqu(f8vec, gen.xword[src]);
    cvt.get_f8_e4m3_emu()->vcvt_f8_to_f32(f32vec, f8vec);
    gen.vmovups(gen.zword[dst], f32vec);
}

template <>
void convert_vec<ov::float16, ov::float8_e4m3>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f8vec = gen.xmm3;
    auto const& f16vec = gen.ymm4;

    auto& cvt = dynamic_cast<jit_convert_array<ov::float16, ov::float8_e4m3>&>(gen);

    gen.vmovdqu(f16vec, gen.yword[src]);
    cvt.get_f8_e4m3_emu()->vcvt_f16_to_f8(f8vec, f16vec);
    gen.vmovdqu(gen.xword[dst], f8vec);
}

template <>
void convert_vec<ov::float8_e4m3, ov::float16>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f8vec = gen.xmm3;
    auto const& f16vec = gen.ymm4;

    auto& cvt = dynamic_cast<jit_convert_array<ov::float8_e4m3, ov::float16>&>(gen);

    gen.vmovdqu(f8vec, gen.xword[src]);
    cvt.get_f8_e4m3_emu()->vcvt_f8_to_f16(f16vec, f8vec);
    gen.vmovdqu(gen.yword[dst], f16vec);
}

template <>
void convert_vec<ov::intel_cpu::bfloat16_t, ov::float8_e4m3>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f8vec = gen.xmm3;
    auto const& f16vec = gen.zmm4;

    auto& cvt = dynamic_cast<jit_convert_array<ov::intel_cpu::bfloat16_t, ov::float8_e4m3>&>(gen);

    gen.vpmovzxwd(f16vec, gen.yword[src]);
    gen.vpslld(f16vec, f16vec, 16);
    cvt.get_f8_e4m3_emu()->vcvt_f32_to_f8(f8vec, f16vec);
    gen.vmovdqu(gen.xword[dst], f8vec);
}

template <>
void convert_vec<ov::float8_e4m3, ov::intel_cpu::bfloat16_t>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f8vec = gen.xmm3;
    auto const& f16vec = gen.ymm4;
    auto const& f32vec = gen.zmm4;

    auto& cvt = dynamic_cast<jit_convert_array<ov::float8_e4m3, ov::intel_cpu::bfloat16_t>&>(gen);

    gen.vmovdqu(f8vec, gen.xword[src]);
    cvt.get_f8_e4m3_emu()->vcvt_f8_to_f32(f32vec, f8vec);
    cvt.get_uni_vcvtneps2bf16()->emit_code({static_cast<size_t>(f32vec.getIdx())},
                                           {static_cast<size_t>(f16vec.getIdx())});
    gen.vmovdqu(gen.yword[dst], f16vec);
}

template <>
void convert_vec<float, ov::float8_e5m2>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f8vec = gen.xmm3;
    auto const& f32vec = gen.zmm4;

    auto& cvt = dynamic_cast<jit_convert_array<float, ov::float8_e5m2>&>(gen);

    gen.vmovups(f32vec, gen.zword[src]);
    cvt.get_f8_e5m2_emu()->vcvt_f32_to_f8(f8vec, f32vec);
    gen.vmovdqu(gen.xword[dst], f8vec);
}

template <>
void convert_vec<ov::float8_e5m2, float>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f8vec = gen.xmm3;
    auto const& f32vec = gen.zmm4;

    auto& cvt = dynamic_cast<jit_convert_array<ov::float8_e5m2, float>&>(gen);

    gen.vmovdqu(f8vec, gen.xword[src]);
    cvt.get_f8_e5m2_emu()->vcvt_f8_to_f32(f32vec, f8vec);
    gen.vmovups(gen.zword[dst], f32vec);
}

template <>
void convert_vec<ov::float16, ov::float8_e5m2>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f8vec = gen.xmm3;
    auto const& f16vec = gen.ymm4;

    auto& cvt = dynamic_cast<jit_convert_array<ov::float16, ov::float8_e5m2>&>(gen);

    gen.vmovdqu(f16vec, gen.yword[src]);
    cvt.get_f8_e5m2_emu()->vcvt_f16_to_f8(f8vec, f16vec);
    gen.vmovdqu(gen.xword[dst], f8vec);
}

template <>
void convert_vec<ov::float8_e5m2, ov::float16>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f8vec = gen.xmm3;
    auto const& f16vec = gen.ymm4;

    auto& cvt = dynamic_cast<jit_convert_array<ov::float8_e5m2, ov::float16>&>(gen);

    gen.vmovdqu(f8vec, gen.xword[src]);
    cvt.get_f8_e5m2_emu()->vcvt_f8_to_f16(f16vec, f8vec);
    gen.vmovdqu(gen.yword[dst], f16vec);
}

template <>
void convert_vec<ov::intel_cpu::bfloat16_t, ov::float8_e5m2>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f8vec = gen.xmm3;
    auto const& f16vec = gen.zmm4;

    auto& cvt = dynamic_cast<jit_convert_array<ov::intel_cpu::bfloat16_t, ov::float8_e5m2>&>(gen);

    gen.vpmovzxwd(f16vec, gen.yword[src]);
    gen.vpslld(f16vec, f16vec, 16);
    cvt.get_f8_e5m2_emu()->vcvt_f32_to_f8(f8vec, f16vec);
    gen.vmovdqu(gen.xword[dst], f8vec);
}

template <>
void convert_vec<ov::float8_e5m2, ov::intel_cpu::bfloat16_t>(jit_generator& gen, const RegExp& src, const RegExp& dst) {
    auto const& f8vec = gen.xmm3;
    auto const& f16vec = gen.ymm4;
    auto const& f32vec = gen.zmm4;

    auto& cvt = dynamic_cast<jit_convert_array<ov::float8_e5m2, ov::intel_cpu::bfloat16_t>&>(gen);

    gen.vmovdqu(f8vec, gen.xword[src]);
    cvt.get_f8_e5m2_emu()->vcvt_f8_to_f32(f32vec, f8vec);
    cvt.get_uni_vcvtneps2bf16()->emit_code({static_cast<size_t>(f32vec.getIdx())},
                                           {static_cast<size_t>(f16vec.getIdx())});
    gen.vmovdqu(gen.yword[dst], f16vec);
}

template <typename TI, typename TO>
void jit_convert(const TI* arg, TO* out, size_t count) {
    using jit_impl = jit_convert_array<TI, TO>;
    static auto converter = jit_impl::get();

    if (converter) {
        typename jit_impl::args_t args = {arg, out, count};
        converter(&args);
    } else {
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<TO>(arg[i]);
        }
    }
}

#endif

template <ov::element::Type_t p>
struct PrecisionInfo {
    using value_type = typename element_type_traits<p>::value_type;
};

template <>
struct PrecisionInfo<ov::element::bf16> {
    using value_type = ov::intel_cpu::bfloat16_t;
};

template <>
struct PrecisionInfo<ov::element::f16> {
    using value_type = ov::float16;
};

template <>
struct PrecisionInfo<ov::element::boolean> {
    using value_type = uint8_t;
};

template <
    typename T,
    typename U =
        std::conditional_t<std::is_same_v<ov::float16, T> || std::is_same_v<ov::intel_cpu::bfloat16_t, T>, float, T>>
struct Range {
    const std::tuple<U, U>& fit(const ov::element::Type& prec);

private:
    std::tuple<U, U> _range{std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()};
};

template <typename T, typename U>
const std::tuple<U, U>& Range<T, U>::fit(const ov::element::Type& prec) {
    if (prec.is_real()) {
        double lbound, ubound;
        switch (prec) {
        case ov::element::f8e4m3:
            lbound = static_cast<double>(std::numeric_limits<ov::float8_e4m3>::lowest());
            ubound = static_cast<double>(std::numeric_limits<ov::float8_e4m3>::max());
            break;
        case ov::element::f8e5m2:
            lbound = static_cast<double>(std::numeric_limits<ov::float8_e5m2>::lowest());
            ubound = static_cast<double>(std::numeric_limits<ov::float8_e5m2>::max());
            break;
        case ov::element::bf16:
            lbound = static_cast<double>(std::numeric_limits<ov::intel_cpu::bfloat16_t>::lowest());
            ubound = static_cast<double>(std::numeric_limits<ov::intel_cpu::bfloat16_t>::max());
            break;
        case ov::element::f16:
            lbound = static_cast<double>(std::numeric_limits<ov::float16>::lowest());
            ubound = static_cast<double>(std::numeric_limits<ov::float16>::max());
            break;
        case ov::element::f32:
            lbound = static_cast<double>(std::numeric_limits<float>::lowest());
            ubound = static_cast<double>(std::numeric_limits<float>::max());
            break;
        case ov::element::f64:
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
        case ov::element::boolean:
            lbound = static_cast<int64_t>(std::numeric_limits<bool>::lowest());
            ubound = static_cast<uint64_t>(std::numeric_limits<bool>::max());
            break;
        case ov::element::u8:
            lbound = static_cast<int64_t>(std::numeric_limits<uint8_t>::lowest());
            ubound = static_cast<uint64_t>(std::numeric_limits<uint8_t>::max());
            break;
        case ov::element::i8:
            lbound = static_cast<int64_t>(std::numeric_limits<int8_t>::lowest());
            ubound = static_cast<uint64_t>(std::numeric_limits<int8_t>::max());
            break;
        case ov::element::u16:
            lbound = static_cast<int64_t>(std::numeric_limits<uint16_t>::lowest());
            ubound = static_cast<uint64_t>(std::numeric_limits<uint16_t>::max());
            break;
        case ov::element::i16:
            lbound = static_cast<int64_t>(std::numeric_limits<int16_t>::lowest());
            ubound = static_cast<uint64_t>(std::numeric_limits<int16_t>::max());
            break;
        case ov::element::u32:
            lbound = static_cast<int64_t>(std::numeric_limits<uint32_t>::lowest());
            ubound = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
            break;
        case ov::element::i32:
            lbound = static_cast<int64_t>(std::numeric_limits<int32_t>::lowest());
            ubound = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
            break;
        case ov::element::u64:
            lbound = static_cast<int64_t>(std::numeric_limits<uint64_t>::lowest());
            ubound = static_cast<uint64_t>(std::numeric_limits<uint64_t>::max());
            break;
        case ov::element::i64:
            lbound = static_cast<int64_t>(std::numeric_limits<int64_t>::lowest());
            ubound = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
            break;
        default:
            OPENVINO_THROW("Unsupported precision");
        }
        using ltype = std::conditional_t<std::is_floating_point<U>::value, double, int64_t>;
        using utype = std::conditional_t<std::is_floating_point<U>::value, double, uint64_t>;
        std::get<0>(_range) =
            static_cast<U>(std::max(static_cast<ltype>(std::get<0>(_range)), static_cast<ltype>(lbound)));
        std::get<1>(_range) =
            static_cast<U>(std::min(static_cast<utype>(std::get<1>(_range)), static_cast<utype>(ubound)));
    }
    return _range;
}

struct ConvertContext {
    const void* srcPtr;
    void* dstPtr;
    size_t size;
    ov::element::Type interimPrc;
    ov::element::Type dstPrc;
    bool converted;

    template <typename T>
    [[nodiscard]] std::tuple<T, T> range() const {
        Range<T> r;
        r.fit(interimPrc);
        return r.fit(dstPrc);
    }
};

template <typename T>
struct ConvertPrecision;

template <typename src_t, typename dst_t>
struct ConvertPrecision<std::tuple<src_t, dst_t>> {
    void operator()(ConvertContext& ctx) {
        auto src = static_cast<const src_t*>(ctx.srcPtr);
        auto dst = static_cast<dst_t*>(ctx.dstPtr);
        src_t lbound, ubound;
        std::tie(lbound, ubound) = ctx.range<src_t>();

        // Align with the behavior of ngraph ref and jit implementation. Conversion from f8e4m3-inf
        // to float should output float-inf instead of f8e4m3-max. Proper handling of special values
        // (nan, inf, overflow) has already been assured by the conversion process.
        if (std::is_same<src_t, ov::float8_e4m3>::value || std::is_same<src_t, ov::float8_e5m2>::value ||
            std::is_same<dst_t, ov::float8_e4m3>::value || std::is_same<dst_t, ov::float8_e5m2>::value) {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<dst_t>(src[i]);
            });
            ctx.converted = true;
            return;
        }

        if (std::is_integral<src_t>::value || ctx.interimPrc.is_real() || std::is_integral<dst_t>::value) {
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

template <>
struct ConvertPrecision<std::tuple<float, ov::intel_cpu::bfloat16_t>> {
    void operator()(ConvertContext& ctx) {
        auto src = static_cast<const float*>(ctx.srcPtr);
        auto dst = static_cast<ov::intel_cpu::bfloat16_t*>(ctx.dstPtr);

        if (ctx.interimPrc.is_real()) {
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

template <>
struct ConvertPrecision<std::tuple<ov::intel_cpu::bfloat16_t, float>> {
    void operator()(ConvertContext& ctx) {
        auto src = static_cast<const ov::intel_cpu::bfloat16_t*>(ctx.srcPtr);
        auto dst = static_cast<float*>(ctx.dstPtr);

        if (ctx.interimPrc.is_real()) {
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
template <typename src_t>
struct ConvertPrecision<std::tuple<src_t, ov::float16>> {
    void operator()(ConvertContext& ctx) {
        auto src = static_cast<const src_t*>(ctx.srcPtr);
        auto dst = static_cast<ov::float16*>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);
        typedef float batch_type[batch];

        src_t lbound, ubound;
        std::tie(lbound, ubound) = ctx.range<src_t>();

        if (std::is_integral<src_t>::value) {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                for (size_t j = 0; j < current_batch_size; ++j) {  // src_t -> fp32
                    tmp[j] = static_cast<float>(std::max(std::min(src[offset + j], ubound), lbound));
                }
                jit_convert(tmp, dst + offset, current_batch_size);  // fp32 -> fp16
            });
        } else if (ctx.interimPrc.is_real()) {
            parallel_for(iterations, [&](size_t i) {
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                if (std::is_same<std::remove_cv_t<src_t>, float>::value) {  // fp32 -> fp16
                    jit_convert(reinterpret_cast<const float*>(src) + offset, dst + offset, current_batch_size);
                } else {
                    batch_type tmp;
                    for (size_t j = 0; j < current_batch_size; ++j) {  // src_t -> fp32
                        tmp[j] = static_cast<float>(src[offset + j]);
                    }
                    jit_convert(tmp, dst + offset, current_batch_size);  // fp32 -> fp16
                }
            });
        } else {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                for (size_t j = 0; j < current_batch_size; ++j) {  // src_t -> fp32
                    tmp[j] = static_cast<float>(std::trunc(std::max(std::min(src[offset + j], ubound), lbound)));
                }
                jit_convert(tmp, dst + offset, current_batch_size);  // fp32 -> fp16
            });
        }

        ctx.converted = true;
    }
};

template <typename dst_t>
struct ConvertPrecision<std::tuple<ov::float16, dst_t>> {
    void operator()(ConvertContext& ctx) {
        auto src = static_cast<const ov::float16*>(ctx.srcPtr);
        auto dst = static_cast<dst_t*>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);
        typedef float batch_type[batch];

        float lbound, ubound;
        std::tie(lbound, ubound) = ctx.range<ov::float16>();

        if (std::is_integral<dst_t>::value) {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, tmp, current_batch_size);  // fp16 -> fp32
                for (size_t j = 0; j < current_batch_size; ++j) {    // fp32 -> dst_t
                    dst[offset + j] = static_cast<dst_t>(std::max(std::min(tmp[j], ubound), lbound));
                }
            });
        } else if (ctx.interimPrc.is_real()) {
            parallel_for(iterations, [&](size_t i) {
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                if (std::is_same<std::remove_cv_t<dst_t>, float>::value) {  // fp16 -> fp32
                    jit_convert(src + offset, reinterpret_cast<float*>(dst) + offset, current_batch_size);
                } else {
                    batch_type tmp;
                    jit_convert(src + offset, tmp, current_batch_size);  // fp16 -> fp32
                    for (size_t j = 0; j < current_batch_size; ++j) {    // fp32 -> dst_t
                        dst[offset + j] = static_cast<dst_t>(tmp[j]);
                    }
                }
            });
        } else {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, tmp, current_batch_size);  // fp16 -> fp32
                for (size_t j = 0; j < current_batch_size; ++j) {    // fp32 -> dst_t
                    dst[offset + j] = static_cast<dst_t>(std::trunc(std::max(std::min(tmp[j], ubound), lbound)));
                }
            });
        }

        ctx.converted = true;
    }
};

template <>
struct ConvertPrecision<std::tuple<ov::float16, ov::float16>> {
    void operator()(ConvertContext& ctx) {
        auto src = static_cast<const ov::float16*>(ctx.srcPtr);
        auto dst = static_cast<ov::float16*>(ctx.dstPtr);

        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);
        typedef float batch_type[batch];

        float lbound, ubound;
        std::tie(lbound, ubound) = ctx.range<ov::float16>();

        if (ctx.interimPrc.is_real()) {
            cpu_memcpy(dst, src, ctx.size * sizeof(ov::float16));
        } else {
            parallel_for(iterations, [&](size_t i) {
                batch_type tmp;
                const size_t offset = i * batch;
                const size_t current_batch_size = std::min(ctx.size - offset, batch);
                jit_convert(src + offset, tmp, current_batch_size);  // fp16 -> fp32
                for (size_t j = 0; j < current_batch_size; ++j) {    // truncate fp32
                    tmp[j] = std::trunc(std::max(std::min(tmp[j], ubound), lbound));
                }
                jit_convert(tmp, dst + offset, current_batch_size);  // fp32 -> fp16
            });
        }

        ctx.converted = true;
    }
};
#endif

}  // namespace

#define INTEL_CPU_CVT(ST, DT)                            \
    OV_CASE2(ov::element::ST,                            \
             ov::element::DT,                            \
             PrecisionInfo<ov::element::ST>::value_type, \
             PrecisionInfo<ov::element::DT>::value_type)

#define INTEL_CPU_CVT_FP8_LIST                                                                                       \
    INTEL_CPU_CVT(f32, f8e4m3), INTEL_CPU_CVT(f16, f8e4m3), INTEL_CPU_CVT(bf16, f8e4m3), INTEL_CPU_CVT(f8e4m3, f32), \
        INTEL_CPU_CVT(f8e4m3, f16), INTEL_CPU_CVT(f8e4m3, bf16), INTEL_CPU_CVT(f32, f8e5m2),                         \
        INTEL_CPU_CVT(f16, f8e5m2), INTEL_CPU_CVT(bf16, f8e5m2), INTEL_CPU_CVT(f8e5m2, f32),                         \
        INTEL_CPU_CVT(f8e5m2, f16), INTEL_CPU_CVT(f8e5m2, bf16)

#define INTEL_CPU_CVT_LIST                                                                                             \
    INTEL_CPU_CVT(u8, i8), INTEL_CPU_CVT(u8, u16), INTEL_CPU_CVT(u8, i16), INTEL_CPU_CVT(u8, u32),                     \
        INTEL_CPU_CVT(u8, i32), INTEL_CPU_CVT(u8, u64), INTEL_CPU_CVT(u8, i64), INTEL_CPU_CVT(u8, f32),                \
        INTEL_CPU_CVT(u8, f16), INTEL_CPU_CVT(u8, bf16), INTEL_CPU_CVT(u8, f64), INTEL_CPU_CVT(u8, boolean),           \
        INTEL_CPU_CVT(i8, u8), INTEL_CPU_CVT(i8, u16), INTEL_CPU_CVT(i8, i16), INTEL_CPU_CVT(i8, u32),                 \
        INTEL_CPU_CVT(i8, i32), INTEL_CPU_CVT(i8, u64), INTEL_CPU_CVT(i8, i64), INTEL_CPU_CVT(i8, f32),                \
        INTEL_CPU_CVT(i8, f16), INTEL_CPU_CVT(i8, bf16), INTEL_CPU_CVT(i8, f64), INTEL_CPU_CVT(i8, boolean),           \
        INTEL_CPU_CVT(u16, u8), INTEL_CPU_CVT(u16, i8), INTEL_CPU_CVT(u16, i16), INTEL_CPU_CVT(u16, u32),              \
        INTEL_CPU_CVT(u16, i32), INTEL_CPU_CVT(u16, u64), INTEL_CPU_CVT(u16, i64), INTEL_CPU_CVT(u16, f32),            \
        INTEL_CPU_CVT(u16, f16), INTEL_CPU_CVT(u16, bf16), INTEL_CPU_CVT(u16, f64), INTEL_CPU_CVT(u16, boolean),       \
        INTEL_CPU_CVT(i16, u8), INTEL_CPU_CVT(i16, i8), INTEL_CPU_CVT(i16, u16), INTEL_CPU_CVT(i16, u32),              \
        INTEL_CPU_CVT(i16, i32), INTEL_CPU_CVT(i16, u64), INTEL_CPU_CVT(i16, i64), INTEL_CPU_CVT(i16, f32),            \
        INTEL_CPU_CVT(i16, f16), INTEL_CPU_CVT(i16, bf16), INTEL_CPU_CVT(i16, f64), INTEL_CPU_CVT(i16, boolean),       \
        INTEL_CPU_CVT(u32, u8), INTEL_CPU_CVT(u32, i8), INTEL_CPU_CVT(u32, u16), INTEL_CPU_CVT(u32, i16),              \
        INTEL_CPU_CVT(u32, i32), INTEL_CPU_CVT(u32, u64), INTEL_CPU_CVT(u32, i64), INTEL_CPU_CVT(u32, f32),            \
        INTEL_CPU_CVT(u32, f16), INTEL_CPU_CVT(u32, bf16), INTEL_CPU_CVT(u32, f64), INTEL_CPU_CVT(u32, boolean),       \
        INTEL_CPU_CVT(i32, u8), INTEL_CPU_CVT(i32, i8), INTEL_CPU_CVT(i32, u16), INTEL_CPU_CVT(i32, i16),              \
        INTEL_CPU_CVT(i32, u32), INTEL_CPU_CVT(i32, u64), INTEL_CPU_CVT(i32, i64), INTEL_CPU_CVT(i32, f32),            \
        INTEL_CPU_CVT(i32, f16), INTEL_CPU_CVT(i32, bf16), INTEL_CPU_CVT(i32, f64), INTEL_CPU_CVT(i32, boolean),       \
        INTEL_CPU_CVT(u64, u8), INTEL_CPU_CVT(u64, i8), INTEL_CPU_CVT(u64, u16), INTEL_CPU_CVT(u64, i16),              \
        INTEL_CPU_CVT(u64, u32), INTEL_CPU_CVT(u64, i32), INTEL_CPU_CVT(u64, i64), INTEL_CPU_CVT(u64, f32),            \
        INTEL_CPU_CVT(u64, f16), INTEL_CPU_CVT(u64, bf16), INTEL_CPU_CVT(u64, f64), INTEL_CPU_CVT(u64, boolean),       \
        INTEL_CPU_CVT(i64, u8), INTEL_CPU_CVT(i64, i8), INTEL_CPU_CVT(i64, u16), INTEL_CPU_CVT(i64, i16),              \
        INTEL_CPU_CVT(i64, u32), INTEL_CPU_CVT(i64, i32), INTEL_CPU_CVT(i64, u64), INTEL_CPU_CVT(i64, f32),            \
        INTEL_CPU_CVT(i64, f16), INTEL_CPU_CVT(i64, bf16), INTEL_CPU_CVT(i64, f64), INTEL_CPU_CVT(i64, boolean),       \
        INTEL_CPU_CVT(f32, u8), INTEL_CPU_CVT(f32, i8), INTEL_CPU_CVT(f32, u16), INTEL_CPU_CVT(f32, i16),              \
        INTEL_CPU_CVT(f32, u32), INTEL_CPU_CVT(f32, i32), INTEL_CPU_CVT(f32, u64), INTEL_CPU_CVT(f32, i64),            \
        INTEL_CPU_CVT(f32, f16), INTEL_CPU_CVT(f32, bf16), INTEL_CPU_CVT(f32, f64), INTEL_CPU_CVT(f32, boolean),       \
        INTEL_CPU_CVT(f16, u8), INTEL_CPU_CVT(f16, i8), INTEL_CPU_CVT(f16, u16), INTEL_CPU_CVT(f16, i16),              \
        INTEL_CPU_CVT(f16, u32), INTEL_CPU_CVT(f16, i32), INTEL_CPU_CVT(f16, u64), INTEL_CPU_CVT(f16, i64),            \
        INTEL_CPU_CVT(f16, f32), INTEL_CPU_CVT(f16, bf16), INTEL_CPU_CVT(f16, f64), INTEL_CPU_CVT(f16, boolean),       \
        INTEL_CPU_CVT(bf16, u8), INTEL_CPU_CVT(bf16, i8), INTEL_CPU_CVT(bf16, u16), INTEL_CPU_CVT(bf16, i16),          \
        INTEL_CPU_CVT(bf16, u32), INTEL_CPU_CVT(bf16, i32), INTEL_CPU_CVT(bf16, u64), INTEL_CPU_CVT(bf16, i64),        \
        INTEL_CPU_CVT(bf16, f32), INTEL_CPU_CVT(bf16, f16), INTEL_CPU_CVT(bf16, f64), INTEL_CPU_CVT(bf16, boolean),    \
        INTEL_CPU_CVT(f64, u8), INTEL_CPU_CVT(f64, i8), INTEL_CPU_CVT(f64, u16), INTEL_CPU_CVT(f64, i16),              \
        INTEL_CPU_CVT(f64, u32), INTEL_CPU_CVT(f64, i32), INTEL_CPU_CVT(f64, u64), INTEL_CPU_CVT(f64, i64),            \
        INTEL_CPU_CVT(f64, f32), INTEL_CPU_CVT(f64, f16), INTEL_CPU_CVT(f64, bf16), INTEL_CPU_CVT(f64, boolean),       \
        INTEL_CPU_CVT(boolean, u8), INTEL_CPU_CVT(boolean, i8), INTEL_CPU_CVT(boolean, u16),                           \
        INTEL_CPU_CVT(boolean, i16), INTEL_CPU_CVT(boolean, u32), INTEL_CPU_CVT(boolean, i32),                         \
        INTEL_CPU_CVT(boolean, u64), INTEL_CPU_CVT(boolean, i64), INTEL_CPU_CVT(boolean, f32),                         \
        INTEL_CPU_CVT(boolean, f16), INTEL_CPU_CVT(boolean, bf16), INTEL_CPU_CVT(boolean, f64), INTEL_CPU_CVT(u8, u8), \
        INTEL_CPU_CVT(i8, i8), INTEL_CPU_CVT(u16, u16), INTEL_CPU_CVT(i16, i16), INTEL_CPU_CVT(u32, u32),              \
        INTEL_CPU_CVT(i32, i32), INTEL_CPU_CVT(u64, u64), INTEL_CPU_CVT(i64, i64), INTEL_CPU_CVT(f32, f32),            \
        INTEL_CPU_CVT(f16, f16), INTEL_CPU_CVT(bf16, bf16), INTEL_CPU_CVT(f64, f64), INTEL_CPU_CVT(boolean, boolean),  \
        INTEL_CPU_CVT_FP8_LIST

#define INTEL_CPU_CVT_FROM_BIN_LIST                                                                     \
    INTEL_CPU_CVT(u1, f32), INTEL_CPU_CVT(u1, f16), INTEL_CPU_CVT(u1, bf16), INTEL_CPU_CVT(u1, f64),    \
        INTEL_CPU_CVT(u1, i16), INTEL_CPU_CVT(u1, u8), INTEL_CPU_CVT(u1, i8), INTEL_CPU_CVT(u1, u16),   \
        INTEL_CPU_CVT(u1, i32), INTEL_CPU_CVT(u1, u32), INTEL_CPU_CVT(u1, i64), INTEL_CPU_CVT(u1, u64), \
        INTEL_CPU_CVT(u1, boolean)

struct ConvertFromBinContext {
    const void* srcPtr;
    void* dstPtr;
    size_t size;
    bool converted;
};

template <typename T>
struct ConvertFromBinPrecision;

template <typename src_t, typename dst_t>
struct ConvertFromBinPrecision<std::tuple<src_t, dst_t>> {
    void operator()(ConvertFromBinContext& ctx) {
        auto src = static_cast<const uint8_t*>(ctx.srcPtr);
        auto dst = static_cast<dst_t*>(ctx.dstPtr);
        const size_t nBits = 8;
        const size_t nBytes = rnd_up(ctx.size, nBits);
        parallel_for(nBytes, [&](size_t byteIndex) {
            auto currentBitNum = std::min(nBits, ctx.size - byteIndex * nBits);
            for (size_t bitIndex = 0; bitIndex < currentBitNum; ++bitIndex) {
                dst[byteIndex * nBits + bitIndex] = static_cast<dst_t>((src[byteIndex] & (1 << bitIndex)) >> bitIndex);
            }
        });
        ctx.converted = true;
    }
};

#define INTEL_CPU_CVT_FROM_4BIT_LIST                                                                                 \
    INTEL_CPU_CVT(u4, f32), INTEL_CPU_CVT(u4, i32), INTEL_CPU_CVT(u4, bf16), INTEL_CPU_CVT(u4, f16),                 \
        INTEL_CPU_CVT(u4, i8), INTEL_CPU_CVT(u4, u8), INTEL_CPU_CVT(i4, f32), INTEL_CPU_CVT(i4, i32),                \
        INTEL_CPU_CVT(i4, bf16), INTEL_CPU_CVT(i4, f16), INTEL_CPU_CVT(i4, i8), INTEL_CPU_CVT(i4, u8),               \
        INTEL_CPU_CVT(nf4, f32), INTEL_CPU_CVT(nf4, bf16), INTEL_CPU_CVT(nf4, f16), INTEL_CPU_CVT(nf4, i8),          \
        INTEL_CPU_CVT(nf4, u8), INTEL_CPU_CVT(f4e2m1, f32), INTEL_CPU_CVT(f4e2m1, bf16), INTEL_CPU_CVT(f4e2m1, f16), \
        INTEL_CPU_CVT(f4e2m1, i8), INTEL_CPU_CVT(f4e2m1, u8)

struct ConvertFrom4BitContext {
    ov::element::Type_t inType;
    const void* srcPtr;
    void* dstPtr;
    size_t size;
    bool converted;
};

static int8_t get_i4(const uint8_t& val, bool high) {
    if (high) {
        if (val & 0x80) {
            return static_cast<int8_t>((val >> 4) | 0xf8);
        }
        return static_cast<int8_t>(val >> 4);
    }
    if (val & 0x8) {
        // Just fill in the high 4 bits with 1
        return static_cast<int8_t>(val | 0xf8);
    }
    return static_cast<int8_t>(val & 0xF);
}

static int8_t get_u4(const uint8_t& val, bool high) {
    return high ? (val >> 4) : (val & 0xF);
}

template <typename T>
struct ConvertFrom4BitPrecision;

template <typename src_t, typename dst_t>
struct ConvertFrom4BitPrecision<std::tuple<src_t, dst_t>> {
    void operator()(ConvertFrom4BitContext& ctx) {
        auto src = static_cast<const uint8_t*>(ctx.srcPtr);
        auto dst = static_cast<dst_t*>(ctx.dstPtr);
        if (ctx.inType == ov::element::nf4) {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<dst_t>(ConvertNF4::dequantize(get_u4(src[i / 2], i % 2)));
            });
        } else if (ctx.inType == ov::element::u4) {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<dst_t>(get_u4(src[i / 2], i % 2));
            });
        } else if (ctx.inType == ov::element::i4) {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<dst_t>(get_i4(src[i / 2], i % 2));
            });
        } else if (ctx.inType == ov::element::f4e2m1) {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<dst_t>(float4_e2m1::from_bits(get_u4(src[i / 2], i % 2)));
            });
        } else {
            OPENVINO_THROW("cpu_convert doesn't support input data type: ", ctx.inType, ". Not implemented.");
        }
        ctx.converted = true;
    }
};

#define INTEL_CPU_CVT_TO_4BIT_LIST INTEL_CPU_CVT(f32, nf4), INTEL_CPU_CVT(f16, nf4), INTEL_CPU_CVT(bf16, nf4)

struct ConvertTo4BitContext {
    ov::element::Type_t outType;
    const void* srcPtr;
    void* dstPtr;
    size_t size;
    bool converted;
};

template <typename T>
struct ConvertTo4BitPrecision;

template <typename src_t, typename dst_t>
struct ConvertTo4BitPrecision<std::tuple<src_t, dst_t>> {
    void operator()(ConvertTo4BitContext& ctx) {
        auto insert_half_byte = [](uint8_t dst, uint8_t val, bool high_half) -> uint8_t {
            uint8_t shift = high_half ? 4 : 0;
            return dst | (uint8_t)(val << shift);
        };

        auto src = static_cast<const src_t*>(ctx.srcPtr);
        auto dst = static_cast<uint8_t*>(ctx.dstPtr);
        // each byte must be fully processed within same thread
        auto work_amount = ctx.size / 2;
        auto has_tail = ctx.size % work_amount != 0;
        if (ctx.outType == ov::element::nf4) {
            parallel_for(work_amount, [&](size_t ib) {
                size_t idx = ib * 2;
                const auto val = insert_half_byte(0, ConvertNF4::quantize(static_cast<float>(src[idx])), false);
                dst[ib] = insert_half_byte(val, ConvertNF4::quantize(static_cast<float>(src[idx + 1])), true);
            });

            if (has_tail) {
                dst[work_amount] =
                    insert_half_byte(0, ConvertNF4::quantize(static_cast<float>(src[2 * work_amount])), false);
            }
        } else {
            OPENVINO_THROW("cpu_convert doesn't support output data type: ", ctx.outType, ". Not implemented.");
        }
        ctx.converted = true;
    }
};

#define INTEL_CPU_CVT_FROM_BYTE_FP_LIST \
    INTEL_CPU_CVT(f8e8m0, f32), INTEL_CPU_CVT(f8e8m0, bf16), INTEL_CPU_CVT(f8e8m0, f16)

struct ConvertFromByteFPContext {
    ov::element::Type_t inType;
    const void* srcPtr;
    void* dstPtr;
    size_t size;
    bool converted;
};

template <typename T>
struct ConvertFromByteFPPrecision;

template <typename src_t, typename dst_t>
struct ConvertFromByteFPPrecision<std::tuple<src_t, dst_t>> {
    void operator()(ConvertFromByteFPContext& ctx) {
        auto src = static_cast<const uint8_t*>(ctx.srcPtr);
        auto dst = static_cast<dst_t*>(ctx.dstPtr);
        if (ctx.inType == ov::element::f8e8m0) {
            parallel_for(ctx.size, [&](size_t i) {
                dst[i] = static_cast<dst_t>(float8_e8m0::from_bits(src[i]));
            });
        } else {
            OPENVINO_THROW("cpu_convert doesn't support input data type: ", ctx.inType, ". Not implemented.");
        }
        ctx.converted = true;
    }
};

#if defined(OPENVINO_ARCH_X86_64)
struct ConvertFP8Context {
    const void* srcPtr;
    void* dstPtr;
    size_t size;
    bool converted;
};

template <typename T>
struct ConvertFP8Precision;

template <typename src_t, typename dst_t>
struct ConvertFP8Precision<std::tuple<src_t, dst_t>> {
    void operator()(ConvertFP8Context& ctx) {
        auto src = static_cast<const src_t*>(ctx.srcPtr);
        auto dst = static_cast<dst_t*>(ctx.dstPtr);
        constexpr size_t batch = 64;
        const size_t iterations = ov::intel_cpu::div_up(ctx.size, batch);
        parallel_for(iterations, [&](size_t i) {
            const size_t offset = i * batch;
            const size_t current_batch_size = std::min(ctx.size - offset, batch);
            jit_convert(src + offset, dst + offset, current_batch_size);
        });

        ctx.converted = true;
    }
};
#endif

void cpu_convert(const void* srcPtr,
                 void* dstPtr,
                 ov::element::Type srcPrc,
                 ov::element::Type dstPrc,
                 const size_t size) {
    cpu_convert(srcPtr, dstPtr, srcPrc, dstPrc, dstPrc, size);
}

void cpu_convert(const void* srcPtr,
                 void* dstPtr,
                 ov::element::Type srcPrc,
                 ov::element::Type interimPrc,
                 ov::element::Type dstPrc,
                 const size_t size) {
    if (size == 0) {
        return;
    }
    if (srcPtr == nullptr || dstPtr == nullptr) {
        OPENVINO_THROW("cpu_convert has null data pointer");
    }

    if (srcPrc == dstPrc && srcPrc == interimPrc) {
        const size_t L2_cache_size = dnnl::utils::get_cache_size(2, true);
        const size_t totalSize = size * dstPrc.size();
        if (srcPrc == element::string) {
            auto str_src = reinterpret_cast<const StringMemory::OvString*>(srcPtr);
            auto str_dst = reinterpret_cast<StringMemory::OvString*>(dstPtr);
            std::copy(str_src, str_src + size, str_dst);
        } else if (totalSize >= L2_cache_size) {
            auto src = static_cast<const uint8_t*>(srcPtr);
            auto dst = static_cast<uint8_t*>(dstPtr);
            parallel_nt(0, [&](const size_t ithr, const size_t nthr) {
                size_t start = 0, end = 0;
                splitter(totalSize, nthr, ithr, start, end);
                cpu_memcpy(dst + start, src + start, end - start);
            });
        } else {
            cpu_memcpy(dstPtr, srcPtr, size * dstPrc.size());
        }
    } else if (srcPrc == ov::element::u1) {
        if (srcPrc.bitwidth() != 1) {
            OPENVINO_THROW("cpu_convert can't convert from: ",
                           srcPrc,
                           " <bitsSize == ",
                           srcPrc.bitwidth(),
                           "> precision to: ",
                           dstPrc,
                           ". Not implemented.");
        }
        ConvertFromBinContext ctx{srcPtr, dstPtr, size, false};
        OV_SWITCH(intel_cpu, ConvertFromBinPrecision, ctx, std::tie(srcPrc, dstPrc), INTEL_CPU_CVT_FROM_BIN_LIST);
        if (!ctx.converted) {
            OPENVINO_THROW("cpu_convert can't convert from: ",
                           srcPrc,
                           " <bitsSize == ",
                           srcPrc.bitwidth(),
                           "> precision to: ",
                           dstPrc);
        }
    } else if (srcPrc.bitwidth() == 4u) {
        ConvertFrom4BitContext ctx{srcPrc, srcPtr, dstPtr, size, false};
        OV_SWITCH(intel_cpu, ConvertFrom4BitPrecision, ctx, std::tie(srcPrc, dstPrc), INTEL_CPU_CVT_FROM_4BIT_LIST);
        if (!ctx.converted) {
            OPENVINO_THROW("cpu_convert can't convert from: ", srcPrc, " precision to: ", dstPrc);
        }
    } else if (dstPrc.bitwidth() == 4u) {
        ConvertTo4BitContext ctx{dstPrc, srcPtr, dstPtr, size, false};
        OV_SWITCH(intel_cpu, ConvertTo4BitPrecision, ctx, std::tie(srcPrc, dstPrc), INTEL_CPU_CVT_TO_4BIT_LIST);
        if (!ctx.converted) {
            OPENVINO_THROW("cpu_convert can't convert from: ", srcPrc, " precision to: ", dstPrc);
        }
    } else if (srcPrc == ov::element::f8e8m0) {
        ConvertFromByteFPContext ctx{srcPrc, srcPtr, dstPtr, size, false};
        OV_SWITCH(intel_cpu,
                  ConvertFromByteFPPrecision,
                  ctx,
                  std::tie(srcPrc, dstPrc),
                  INTEL_CPU_CVT_FROM_BYTE_FP_LIST);
        if (!ctx.converted) {
            OPENVINO_THROW("cpu_convert can't convert from: ", srcPrc, " precision to: ", dstPrc);
        }
#if defined(OPENVINO_ARCH_X86_64)
    } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_fp16) &&
               (one_of(srcPrc, ov::element::f8e4m3, ov::element::f8e5m2) ||
                one_of(dstPrc, ov::element::f8e4m3, ov::element::f8e5m2))) {
        ConvertFP8Context ctx{srcPtr, dstPtr, size, false};
        OV_SWITCH(intel_cpu, ConvertFP8Precision, ctx, std::tie(srcPrc, dstPrc), INTEL_CPU_CVT_FP8_LIST);
        if (!ctx.converted) {
            OPENVINO_THROW("cpu_convert can't convert from: ", srcPrc, " precision to: ", dstPrc);
        }
#endif
    } else {
        ConvertContext ctx{srcPtr, dstPtr, size, interimPrc, dstPrc, false};
        OV_SWITCH(intel_cpu, ConvertPrecision, ctx, std::tie(srcPrc, dstPrc), INTEL_CPU_CVT_LIST);
        if (!ctx.converted) {
            OPENVINO_THROW("cpu_convert can't convert from: ", srcPrc, " precision to: ", dstPrc);
        }
    }
}

struct isSupportedContext {
    bool isSupported = false;
};

template <typename DT>
struct isSupported {
    void operator()(isSupportedContext& ctx) {
        ctx.isSupported = true;
    }
};

bool is_supported_convert(ov::element::Type srcPrc, ov::element::Type dstPrc) {
    isSupportedContext ctx;
    OV_SWITCH(intel_cpu, isSupported, ctx, std::tie(srcPrc, dstPrc), INTEL_CPU_CVT_LIST);
    OV_SWITCH(intel_cpu, isSupported, ctx, std::tie(srcPrc, dstPrc), INTEL_CPU_CVT_FROM_BIN_LIST);
    OV_SWITCH(intel_cpu, isSupported, ctx, std::tie(srcPrc, dstPrc), INTEL_CPU_CVT_FROM_4BIT_LIST);
    OV_SWITCH(intel_cpu, isSupported, ctx, std::tie(srcPrc, dstPrc), INTEL_CPU_CVT_FROM_BYTE_FP_LIST);
    OV_SWITCH(intel_cpu, isSupported, ctx, std::tie(srcPrc, dstPrc), INTEL_CPU_CVT_TO_4BIT_LIST);
    return ctx.isSupported;
}

#undef INTEL_CPU_CVT
#undef INTEL_CPU_CVT_LIST

}  // namespace ov::intel_cpu
