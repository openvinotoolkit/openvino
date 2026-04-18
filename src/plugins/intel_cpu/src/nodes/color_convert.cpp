// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "color_convert.h"

#include <algorithm>
#include <cmath>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <openvino/core/type.hpp>
#include <openvino/op/i420_to_bgr.hpp>
#include <openvino/op/i420_to_rgb.hpp>
#include <openvino/op/nv12_to_bgr.hpp>
#include <openvino/op/nv12_to_rgb.hpp>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "cpu_parallel.hpp"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shape_inference/custom/color_convert.hpp"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <xbyak/xbyak.h>

#    include <array>
#    include <common/c_types_map.hpp>
#    include <cpu/x64/jit_generator.hpp>

#    include "kernels/x64/jit_kernel.hpp"
#endif

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov::intel_cpu::node {
namespace {

std::tuple<Algorithm, std::string> getAlgorithmFor(const std::shared_ptr<const ov::Node>& op) {
    if (ov::is_type<ov::op::v8::NV12toRGB>(op)) {
        return std::make_tuple(Algorithm::ColorConvertNV12toRGB, std::string());
    }
    if (ov::is_type<ov::op::v8::NV12toBGR>(op)) {
        return std::make_tuple(Algorithm::ColorConvertNV12toBGR, std::string());
    }
    if (ov::is_type<ov::op::v8::I420toRGB>(op)) {
        return std::make_tuple(Algorithm::ColorConvertI420toRGB, std::string());
    }
    if (ov::is_type<ov::op::v8::I420toBGR>(op)) {
        return std::make_tuple(Algorithm::ColorConvertI420toBGR, std::string());
    }
    return std::make_tuple(Algorithm::Default, std::string("Type ") + op->get_type_name() + " is not supported.");
}

class Converter : public ColorConvert::Converter {
    using Base = ColorConvert::Converter;

public:
    explicit Converter(Node* node);

    [[nodiscard]] bool singlePlane() const;

    template <typename T>
    std::tuple<T, T, T> yuv_to_rgb(float y, float u, float v);
};

Converter::Converter(Node* node)
    : Base(node,
           node->getAlgorithm() == Algorithm::ColorConvertNV12toRGB ||
                   node->getAlgorithm() == Algorithm::ColorConvertI420toRGB
               ? ColorFormat{{0, 1, 2}}
               : ColorFormat{{2, 1, 0}}) {}

bool Converter::singlePlane() const {
    return _node->getOriginalInputsNumber() == 1;
}

template <typename T>
std::tuple<T, T, T> Converter::yuv_to_rgb(float y, float u, float v) {
    auto c = y - 16.F;
    auto d = u - 128.F;
    auto e = v - 128.F;
    auto clip = [](float a) -> T {
        if (std::is_integral<T>()) {
            return static_cast<T>(std::min(std::max(std::round(a), 0.F), 255.F));
        }
        return static_cast<T>(std::min(std::max(a, 0.F), 255.F));
    };
    auto r = clip(1.164F * c + 1.596F * e);
    auto g = clip(1.164F * c - 0.391F * d - 0.813F * e);
    auto b = clip(1.164F * c + 2.018F * d);
    return std::make_tuple(r, g, b);
}

#if defined(OPENVINO_ARCH_X86_64)
struct jit_uni_converter : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_converter)

    struct Params {
        const void* y;
        const void* u;
        const void* v;
        void* dst;
        size_t width;
        uint8_t colorFormat;  // RGB: 0, BGR: !=0
    };

    using function_t = void (*)(const Params*);

    void init();

    void operator()(const Params& args) const {
        _fn(&args);
    }

protected:
    jit_uni_converter();

    // Aggregate output of `yuv_to_rgb`. Kept as a plain struct of three
    // `variable` members so the function can return by value without the
    // tuple/`std::tie`/`std::move` ceremony — each call site ends up with a
    // single named local (`auto rgb = yuv_to_rgb(...)`) that captures cleanly
    // in nested lambdas and reads as `rgb.r`, `rgb.g`, `rgb.b`. Aggregate
    // init from three prvalue FMA expressions materializes the members
    // directly via C++17 guaranteed copy elision — no move ctor calls, no
    // vmovups emitted into the kernel stream.
    //
    // Planar YUV triple returned by `load_yuv` and consumed by `yuv_to_rgb`.
    // Passed by value into `yuv_to_rgb` so the function owns its own copies
    // and can rebind them in place via the move-assign operator at
    // jit_kernel.hpp:549 (`y = (y - y_off) * y_scale`). The old register's
    // refcount drops immediately on rebind, recovering the slot for later
    // work. This keeps the per-iteration vector-register working peak down
    // to ~8 — well within the DSL's 16-register vector pool.
    template <size_t N>
    struct yuv_vec {
        variable<float[N]> y;
        variable<float[N]> u;
        variable<float[N]> v;
    };

    template <size_t N>
    struct rgb_vec {
        variable<float[N]> r;
        variable<float[N]> g;
        variable<float[N]> b;
    };

    template <size_t N>
    rgb_vec<N> yuv_to_rgb(yuv_vec<N> yuv, bool is_integral);

    function_t _fn = nullptr;
    variable<const float*> _consts;
};

jit_uni_converter::jit_uni_converter() : jit_kernel(jit_name()), _consts(*this) {}

void jit_uni_converter::init() {
    OPENVINO_ASSERT(create_kernel() == status::success, "Can't generate jit color converter kernel");
    _fn = reinterpret_cast<function_t>(const_cast<uint8_t*>(jit_ker()));
}

template <size_t N>
jit_uni_converter::rgb_vec<N> jit_uni_converter::yuv_to_rgb(yuv_vec<N> yuv, bool is_integral) {
    // BT.601 slot layout in the static data[] array populated via
    // `_consts = data` at the two create() sites below.
    //   0: Y_OFFSET  = 16
    //   1: UV_OFFSET = 128
    //   2: Y_SCALE   = 1.164
    //   3: V_TO_R    = 1.596
    //   4: U_TO_G    = 0.391
    //   5: U_TO_B    = 2.018
    //   6: V_TO_G    = 0.813
    //   7: CLAMP_HI  = 255
    auto bc = [&](int slot) {
        auto t = var<float[N]>();
        uni_vbroadcastss(t, ptr[_consts + slot * sizeof(float)]);
        return t;
    };

    // BT.601:  y' = 1.164 * (y - 16),  u' = u - 128,  v' = v - 128
    //          R  = y' + 1.596 * v'
    //          G  = y' - 0.391 * u' - 0.813 * v'
    //          B  = y' + 2.018 * u'
    //
    // In-place mutation via move-assign-rebind (jit_kernel.hpp:549): each
    // assignment drops the old register (raw Y/U/V) and rebinds to the
    // prvalue result's fresh register. Old slots return to the pool
    // immediately, keeping the working peak low.
    yuv.y = (yuv.y - bc(0)) * bc(2);
    auto uv_off = bc(1);
    yuv.u = yuv.u - uv_off;
    yuv.v = yuv.v - uv_off;

    auto r = fma(bc(3), yuv.v, yuv.y);                       // y + 1.596 * v
    auto g = fnma(bc(6), yuv.v, fnma(bc(4), yuv.u, yuv.y));  // y - 0.391 * u - 0.813 * v
    auto b = fma(bc(5), yuv.u, yuv.y);                       // y + 2.018 * u

    // Float-output path only: preserve the [0, 255] output contract.
    // Integer-output path skips this entirely — the narrowing store emitter
    // (jit_store_emitter, default arithmetic_mode::saturation) rounds via
    // vcvtps2dq and saturates via the pack chain when narrowing float to u8.
    if (!is_integral) {
        auto lo = var<float[N]>();
        uni_vxorps(lo, lo, lo);
        auto hi = bc(7);
        r = r.clamp(lo, hi);
        g = g.clamp(lo, hi);
        b = b.clamp(lo, hi);
    }

    return rgb_vec<N>{std::move(r), std::move(g), std::move(b)};
}
#endif

namespace nv12 {

ColorConvert::Converter::PrimitiveDescs supportedPrimitiveDescs(Node* node) {
    const LayoutType layout = LayoutType::ncsp;  // 0,1,2,3

    const ov::element::Type precision =
        node->getOriginalInputPrecisionAtPort(0) == ov::element::u8 ? ov::element::u8 : ov::element::f32;

    ColorConvert::Converter::PrimitiveDescs descs;

    descs.emplace_back(std::vector<PortConfigurator>{node->getOriginalInputsNumber(), {layout, precision}},
                       std::vector<PortConfigurator>{{layout, precision}},
                       mayiuse(cpu_isa_t::sse41) ? impl_desc_type::jit_uni : impl_desc_type::ref,
                       true);

    return descs;
}

template <typename T, impl_desc_type I>
class SinglePlaneConvert;
template <typename T, impl_desc_type I>
class TwoPlaneConvert;

class RefConverter : public Converter {
public:
    explicit RefConverter(Node* node);

protected:
    template <typename T>
    void convert(const T* y,
                 const T* uv,
                 T* dst,
                 size_t batch_size,
                 size_t height,
                 size_t width,
                 size_t stride_y,
                 size_t stride_uv,
                 const CpuParallelPtr& cpu_parallel);
};

RefConverter::RefConverter(Node* node) : Converter(node) {
    OPENVINO_ASSERT(node->getOriginalInputsNumber() == (singlePlane() ? 1 : 2),
                    "NV12Converter node has incorrect number of inputs");
    OPENVINO_ASSERT(node->getOriginalOutputsNumber(), "NV12Converter node has incorrect number of outputs");
}

template <typename T>
void RefConverter::convert(const T* y,
                           const T* uv,
                           T* dst,
                           size_t batch_size,
                           size_t height,
                           size_t width,
                           size_t stride_y,
                           size_t stride_uv,
                           const CpuParallelPtr& cpu_parallel) {
    cpu_parallel->parallel_for2d(batch_size, height, [&](int batch, int h) {
        T* out = dst + batch * width * height * 3;
        auto y_ptr = y + batch * stride_y;
        auto uv_ptr = uv + batch * stride_uv;

        for (size_t w = 0; w < width; w++) {
            auto y_index = h * width + w;
            auto y_val = static_cast<float>(y_ptr[y_index]);
            auto uv_index = (h / 2) * width + (w / 2) * 2;
            auto u_val = static_cast<float>(uv_ptr[uv_index]);
            auto v_val = static_cast<float>(uv_ptr[uv_index + 1]);
            auto [r, g, b] = yuv_to_rgb<T>(y_val, u_val, v_val);
            out[y_index * 3 + _colorFormat[0]] = r;
            out[y_index * 3 + _colorFormat[1]] = g;
            out[y_index * 3 + _colorFormat[2]] = b;
        }
    });
}

template <typename T>
class SinglePlaneConvert<T, impl_desc_type::ref> : public RefConverter {
public:
    using RefConverter::RefConverter;

    void execute(const CpuParallelPtr& cpu_parallel, [[maybe_unused]] const dnnl::stream& strm) override {
        const auto& dims = inputDims(0);

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM] * 2 / 3;
        const size_t width = dims[W_DIM];

        const T* y = static_cast<const T*>(input(0));
        const T* uv = y + width * height;
        T* dst = static_cast<T*>(output(0));

        convert<T>(y, uv, dst, batch_size, height, width, height * width * 3 / 2, height * width * 3 / 2, cpu_parallel);
    }
};

template <typename T>
class TwoPlaneConvert<T, impl_desc_type::ref> : public RefConverter {
public:
    using RefConverter::RefConverter;

    void execute(const CpuParallelPtr& cpu_parallel, [[maybe_unused]] const dnnl::stream& strm) override {
        const auto& dims = inputDims(0);

        const T* y = static_cast<const T*>(input(0));
        const T* uv = static_cast<const T*>(input(1));
        T* dst = static_cast<T*>(output(0));

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM];
        const size_t width = dims[W_DIM];

        convert<T>(y, uv, dst, batch_size, height, width, height * width, height * width / 2, cpu_parallel);
    }
};

#if defined(OPENVINO_ARCH_X86_64)
template <typename T>
class JitConverter;

template <typename T, size_t N>
class JitConverter<T[N]> : public jit_uni_converter {
private:
    void generate() override;
    yuv_vec<N> load_yuv(const variable<const T*>& src_y, const variable<const T*>& src_uv);
    std::tuple<variable<float[N]>, variable<float[N]>> unpack_uv(const variable<float[N]>& uv);
};

template <typename T, size_t N>
void JitConverter<T[N]>::generate() {
    using namespace Xbyak;
    using reg_type = typename reg_traits<float[N]>::type;

    preamble();

    static const float data[8] = {16.F, 128.F, 1.164F, 1.596F, 0.391F, 2.018F, 0.813F, 255.F};
    _consts = data;

    const size_t step = N * sizeof(T);

    // LLVM-style: begin_ir before arg — args are IR GPR values.
    // Required for ir_load_partial in the tail path (ir_memcpy needs
    // pointer vids as IR reads).
    begin_ir();

    auto src_y = arg<const T*>(&Params::y);
    auto src_uv = arg<const T*>(&Params::u);
    auto dst = arg<T*>(&Params::dst);
    auto width = arg(&Params::width);
    auto colorFormat = arg(&Params::colorFormat);

    const auto reg_capacity_log = static_cast<int>(std::logb(N));

    auto main_count = variable<size_t>(*this, ir_def_gpr({width.vid()},
        [this, reg_capacity_log](const jit_kernel_ir::EmitContext& ctx) {
            mov(Reg64(ctx.def->idx), Reg64(ctx.reads[0].idx));
            shr(Reg64(ctx.def->idx), reg_capacity_log);
        }, "main_count"));

    auto tail_count = variable<size_t>(*this, ir_def_gpr({width.vid()},
        [this](const jit_kernel_ir::EmitContext& ctx) {
            mov(Reg64(ctx.def->idx), Reg64(ctx.reads[0].idx));
            and_(Reg64(ctx.def->idx), static_cast<size_t>(N - 1));
        }, "tail_count"));

    auto consts_idx = static_cast<std::uint32_t>(_consts.reg().getIdx());

    auto bc = [&](int slot) {
        return ir_broadcast<N>(
            address_frame(sizeof(float))[Xbyak::Reg64(consts_idx) + slot * sizeof(float)]);
    };

    auto y_off    = bc(0);
    auto uv_off   = bc(1);
    auto y_scale  = bc(2);
    auto v_to_r   = bc(3);
    auto u_to_g   = bc(4);
    auto u_to_b   = bc(5);
    auto v_to_g   = bc(6);
    auto clamp_hi = bc(7);

    auto clamp_lo = ir_def<N>({}, [this](const jit_kernel_ir::EmitContext& ctx) {
        uni_vxorps(reg_type(ctx.def->idx), reg_type(ctx.def->idx), reg_type(ctx.def->idx));
    }, "vxorps");

    constexpr uint8_t even_mask = 0xA0;
    constexpr uint8_t odd_mask  = 0xF5;

    // Shared BT.601 conversion: load Y + interleaved UV, math, clamp.
    auto convert = [&]() {
        auto y_raw = ir_load<N>(src_y);
        auto uv    = ir_load<N>(src_uv);
        auto u = vsubps(uv.shuffle(even_mask), uv_off);
        auto v = vsubps(uv.shuffle(odd_mask), uv_off);
        auto y = vmulps(vsubps(y_raw, y_off), y_scale);
        auto r = fma(v_to_r, v, y).clamp(clamp_lo, clamp_hi);
        auto g = fnma(v_to_g, v, fnma(u_to_g, u, y)).clamp(clamp_lo, clamp_hi);
        auto b = fma(u_to_b, u, y).clamp(clamp_lo, clamp_hi);
        return std::make_tuple(std::move(r), std::move(g), std::move(b));
    };

    // ── Main loop ─────────────────────────────────────────────
    foreach(size_t{0}, main_count, [&](const variable<size_t>&) {
        auto rgb = convert();
        auto& r = std::get<0>(rgb);
        auto& g = std::get<1>(rgb);
        auto& b = std::get<2>(rgb);
        ir_cmp(colorFormat, size_t{0});
        ir_if(&Xbyak::CodeGenerator::jne,
            [&]() { store_interleaved3(dst, r, g, b); },
            [&]() { store_interleaved3(dst, b, g, r); });
        ir_use({src_y.vid(), src_uv.vid(), dst.vid()},
            [this](const jit_kernel_ir::EmitContext& ctx) {
                add(Xbyak::Reg64(ctx.reads[0].idx), step);
                add(Xbyak::Reg64(ctx.reads[1].idx), step);
                add(Xbyak::Reg64(ctx.reads[2].idx), 3 * step);
            }, "ptr_advance");
    });

    // ── Tail ──────────────────────────────────────────────────
    ir_cmp(tail_count, size_t{0});
    ir_if(&Xbyak::CodeGenerator::je, [&]() {
        auto y_raw = ir_load_partial<N>(src_y, tail_count);
        auto uv    = ir_load_partial<N>(src_uv, tail_count);
        auto u = vsubps(uv.shuffle(even_mask), uv_off);
        auto v = vsubps(uv.shuffle(odd_mask), uv_off);
        auto y = vmulps(vsubps(y_raw, y_off), y_scale);
        auto r = fma(v_to_r, v, y).clamp(clamp_lo, clamp_hi);
        auto g = fnma(v_to_g, v, fnma(u_to_g, u, y)).clamp(clamp_lo, clamp_hi);
        auto b = fma(u_to_b, u, y).clamp(clamp_lo, clamp_hi);
        ir_cmp(colorFormat, size_t{0});
        ir_if(&Xbyak::CodeGenerator::jne,
            [&]() { store_interleaved3(dst, r, g, b, tail_count); },
            [&]() { store_interleaved3(dst, b, g, r, tail_count); });
    });

    end_ir();

    postamble();
}

template <typename T, size_t N>
jit_uni_converter::yuv_vec<N>
JitConverter<T[N]>::load_yuv(const variable<const T*>& src_y, const variable<const T*>& src_uv) {
    auto y = var<float[N]>(src_y);
    auto uv = var<float[N]>(src_uv);

    auto [u, v] = unpack_uv(uv);

    src_y += N * sizeof(T);
    src_uv += N * sizeof(T);

    return {std::move(y), std::move(u), std::move(v)};
}

template <typename T, size_t N>
std::tuple<jit_kernel::variable<float[N]>, jit_kernel::variable<float[N]>> JitConverter<T[N]>::unpack_uv(
    const variable<float[N]>& uv) {
    constexpr uint8_t even_mask = 0xA0;  // 0b10100000 → [0,0,2,2] per 128-bit lane
    constexpr uint8_t odd_mask = 0xF5;   // 0b11110101 → [1,1,3,3] per 128-bit lane

    auto u = uv.shuffle(even_mask);  // u = uv[0,0,2,2,4,4,6,6,...]
    auto v = uv.shuffle(odd_mask);   // v = uv[1,1,3,3,5,5,7,7,...]

    return std::make_tuple(std::move(u), std::move(v));
}

template <typename T>
const jit_uni_converter& jit_converter_create() {
    auto createKernel = []() {
        std::unique_ptr<jit_uni_converter> kernel;

        if (mayiuse(cpu_isa_t::avx512_core)) {
            auto converter = new JitConverter<T[16]>;
            kernel.reset(converter);
            converter->init();
        } else if (mayiuse(cpu_isa_t::avx2)) {
            auto converter = new JitConverter<T[8]>;
            kernel.reset(converter);
            converter->init();
        } else if (mayiuse(cpu_isa_t::sse41)) {
            auto converter = new JitConverter<T[4]>;
            kernel.reset(converter);
            converter->init();
        } else {
            OPENVINO_THROW("Can't create jit color converter kernel");
        }

        return kernel;
    };

    static auto kernel = createKernel();

    return *kernel;
}

template <typename T>
const jit_uni_converter& jit_converter_get() {
    return jit_converter_create<T>();
}

template <typename T>
class SinglePlaneConvert<T, impl_desc_type::jit_uni> : public Converter {
public:
    explicit SinglePlaneConvert(Node* node) : Converter(node) {
        jit_converter_create<T>();
    }

    void execute(const CpuParallelPtr& cpu_parallel, [[maybe_unused]] const dnnl::stream& strm) override {
        const auto& kernel = jit_converter_get<T>();
        const auto& dims = inputDims(0);

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM] * 2 / 3;
        const size_t width = dims[W_DIM];

        const T* y = static_cast<const T*>(input(0));
        const T* uv = y + width * height;
        T* dst = static_cast<T*>(output(0));

        const size_t stride_y = height * width * 3 / 2;
        const size_t stride_uv = height * width * 3 / 2;

        cpu_parallel->parallel_for2d(batch_size, height, [&](int batch, int h) {
            auto u_v = uv + batch * stride_uv + (h / 2) * width;
            typename jit_uni_converter::Params args{
                y + batch * stride_y + h * width,
                u_v,
                u_v,
                dst + (batch * width * height + h * width) * 3,
                width,
                _colorFormat[0]};  // The first byte is enough to determine the RGB or BGR format.
            kernel(args);
        });
    }
};

template <typename T>
class TwoPlaneConvert<T, impl_desc_type::jit_uni> : public Converter {
public:
    explicit TwoPlaneConvert(Node* node) : Converter(node) {
        jit_converter_create<T>();
    }

    void execute(const CpuParallelPtr& cpu_parallel, [[maybe_unused]] const dnnl::stream& strm) override {
        const auto& kernel = jit_converter_get<T>();
        const auto& dims = inputDims(0);

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM];
        const size_t width = dims[W_DIM];

        const T* y = static_cast<const T*>(input(0));
        const T* uv = static_cast<const T*>(input(1));
        T* dst = static_cast<T*>(output(0));

        const size_t stride_y = height * width;
        const size_t stride_uv = height * width / 2;

        cpu_parallel->parallel_for2d(batch_size, height, [&](int batch, int h) {
            auto u_v = uv + batch * stride_uv + (h / 2) * width;
            typename jit_uni_converter::Params args{
                y + batch * stride_y + h * width,
                u_v,
                u_v,
                dst + (batch * width * height + h * width) * 3,
                width,
                _colorFormat[0]  // The first byte is enough to determine the RGB or BGR format.
            };
            kernel(args);
        });
    }
};
#endif
}  // namespace nv12

namespace i420 {

ColorConvert::Converter::PrimitiveDescs supportedPrimitiveDescs(Node* node) {
    const LayoutType layout = LayoutType::ncsp;  // 0,1,2,3

    const ov::element::Type precision =
        node->getOriginalInputPrecisionAtPort(0) == ov::element::u8 ? ov::element::u8 : ov::element::f32;

    ColorConvert::Converter::PrimitiveDescs descs;

    descs.emplace_back(std::vector<PortConfigurator>{node->getOriginalInputsNumber(), {layout, precision}},
                       std::vector<PortConfigurator>{{layout, precision}},
                       mayiuse(cpu_isa_t::sse41) ? impl_desc_type::jit_uni : impl_desc_type::ref,
                       true);

    return descs;
}

template <typename T, impl_desc_type I>
class SinglePlaneConvert;
template <typename T, impl_desc_type I>
class ThreePlaneConvert;

class RefConverter : public Converter {
public:
    explicit RefConverter(Node* node);

protected:
    template <typename T>
    void convert(const T* y,
                 const T* u,
                 const T* v,
                 T* dst,
                 size_t batch_size,
                 size_t height,
                 size_t width,
                 size_t stride_y,
                 size_t stride_uv,
                 const CpuParallelPtr& cpu_parallel);
};

RefConverter::RefConverter(Node* node) : Converter(node) {
    OPENVINO_ASSERT(node->getOriginalInputsNumber() == (singlePlane() ? 1 : 3),
                    "I420Converter node has incorrect number of inputs");
    OPENVINO_ASSERT(node->getOriginalOutputsNumber(), "I420Converter node has incorrect number of outputs");
}

template <typename T>
void RefConverter::convert(const T* y,
                           const T* u,
                           const T* v,
                           T* dst,
                           size_t batch_size,
                           size_t height,
                           size_t width,
                           size_t stride_y,
                           size_t stride_uv,
                           const CpuParallelPtr& cpu_parallel) {
    cpu_parallel->parallel_for2d(batch_size, height, [&](int batch, int h) {
        T* out = dst + batch * width * height * 3;
        auto y_ptr = y + batch * stride_y;
        auto u_ptr = u + batch * stride_uv;
        auto v_ptr = v + batch * stride_uv;

        for (size_t w = 0; w < width; w++) {
            auto y_index = h * width + w;
            auto y_val = static_cast<float>(y_ptr[y_index]);
            auto uv_index = (h / 2) * (width / 2) + w / 2;
            auto u_val = static_cast<float>(u_ptr[uv_index]);
            auto v_val = static_cast<float>(v_ptr[uv_index]);
            auto [r, g, b] = yuv_to_rgb<T>(y_val, u_val, v_val);
            out[y_index * 3 + _colorFormat[0]] = r;
            out[y_index * 3 + _colorFormat[1]] = g;
            out[y_index * 3 + _colorFormat[2]] = b;
        }
    });
}

template <typename T>
class SinglePlaneConvert<T, impl_desc_type::ref> : public RefConverter {
public:
    using RefConverter::RefConverter;

    void execute(const CpuParallelPtr& cpu_parallel, [[maybe_unused]] const dnnl::stream& strm) override {
        const auto& dims = inputDims(0);

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM] * 2 / 3;
        const size_t width = dims[W_DIM];

        const T* y = static_cast<const T*>(input(0));
        const T* u = y + width * height;
        const T* v = y + 5 * width * height / 4;
        T* dst = static_cast<T*>(output(0));

        convert<
            T>(y, u, v, dst, batch_size, height, width, height * width * 3 / 2, height * width * 3 / 2, cpu_parallel);
    }
};

template <typename T>
class ThreePlaneConvert<T, impl_desc_type::ref> : public RefConverter {
public:
    using RefConverter::RefConverter;

    void execute(const CpuParallelPtr& cpu_parallel, [[maybe_unused]] const dnnl::stream& strm) override {
        const auto& dims = inputDims(0);

        const T* y = static_cast<const T*>(input(0));
        const T* u = static_cast<const T*>(input(1));
        const T* v = static_cast<const T*>(input(2));
        T* dst = static_cast<T*>(output(0));

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM];
        const size_t width = dims[W_DIM];

        convert<T>(y, u, v, dst, batch_size, height, width, height * width, height * width / 4, cpu_parallel);
    }
};

#if defined(OPENVINO_ARCH_X86_64)
template <typename T>
class JitConverter;

template <typename T, size_t N>
class JitConverter<T[N]> : public jit_uni_converter {
private:
    void generate() override;
};

template <typename T, size_t N>
void JitConverter<T[N]>::generate() {
    using namespace Xbyak;
    using reg_type = typename reg_traits<float[N]>::type;

    preamble();

    static const float data[8] = {16.F, 128.F, 1.164F, 1.596F, 0.391F, 2.018F, 0.813F, 255.F};
    _consts = data;

    const auto reg_capacity_log = static_cast<int>(std::logb(N));
    const size_t step = N * sizeof(T);
    constexpr size_t uv_step = N * sizeof(T) / 2;

    // UV duplication permute order: [0,0,1,1,2,2,...] — each chroma
    // sample covers 2 luma samples in 4:2:0 subsampling.
    static const uint8_t uv_unpack_order[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};

    begin_ir();

    // LLVM-style: args are IR GPR values, loaded at lowering time.
    auto src_y = arg<const T*>(&Params::y);
    auto src_u = arg<const T*>(&Params::u);
    auto src_v = arg<const T*>(&Params::v);
    auto dst = arg<T*>(&Params::dst);
    auto width = arg(&Params::width);
    auto colorFormat = arg(&Params::colorFormat);

    // Pre-compute loop counts as IR GPR ops.
    auto main_count = variable<size_t>(*this, ir_def_gpr({width.vid()},
        [this, reg_capacity_log](const jit_kernel_ir::EmitContext& ctx) {
            mov(Reg64(ctx.def->idx), Reg64(ctx.reads[0].idx));
            shr(Reg64(ctx.def->idx), reg_capacity_log);
        }, "main_count"));

    auto tail_count = variable<size_t>(*this, ir_def_gpr({width.vid()},
        [this](const jit_kernel_ir::EmitContext& ctx) {
            mov(Reg64(ctx.def->idx), Reg64(ctx.reads[0].idx));
            and_(Reg64(ctx.def->idx), static_cast<size_t>(N - 1));
        }, "tail_count"));

    auto uv_tail = variable<size_t>(*this, ir_def_gpr({tail_count.vid()},
        [this](const jit_kernel_ir::EmitContext& ctx) {
            mov(Reg64(ctx.def->idx), Reg64(ctx.reads[0].idx));
            shr(Reg64(ctx.def->idx), 1);
        }, "uv_tail"));

    auto consts_idx = static_cast<std::uint32_t>(_consts.reg().getIdx());

    // ── Hoist BT.601 constants (same as NV12 kernel) ───────────
    auto bc = [&](int slot) {
        return ir_broadcast<N>(
            address_frame(sizeof(float))[Xbyak::Reg64(consts_idx) + slot * sizeof(float)]);
    };
    auto y_off    = bc(0);
    auto uv_off   = bc(1);
    auto y_scale  = bc(2);
    auto v_to_r   = bc(3);
    auto u_to_g   = bc(4);
    auto u_to_b   = bc(5);
    auto v_to_g   = bc(6);
    auto clamp_hi = bc(7);

    auto clamp_lo = ir_def<N>({}, [this](const jit_kernel_ir::EmitContext& ctx) {
        uni_vxorps(reg_type(ctx.def->idx), reg_type(ctx.def->idx), reg_type(ctx.def->idx));
    }, "vxorps");

    // ── Main loop: full-width iterations ───────────────────────
    foreach(size_t{0}, main_count, [&](const variable<size_t>&) {
        // Load Y (full width) — ir_load sees src_y.vid(), uses it as GPR read
        auto y_raw = ir_load<N>(src_y);

        // Load U/V (half width N/2 into lower half of N-wide register)
        // Pointer is an IR GPR read — resolved at lowering time.
        auto u_raw = ir_def<N>({src_u.vid()}, [this](const jit_kernel_ir::EmitContext& ctx) {
            auto ptr = Xbyak::Reg64(ctx.reads[0].idx);
            if constexpr (std::is_same_v<T, uint8_t>) {
                uni_vpmovzxbd(reg_type(ctx.def->idx), address_frame(N / 2)[ptr]);
                uni_vcvtdq2ps(reg_type(ctx.def->idx), reg_type(ctx.def->idx));
            } else {
                uni_vmovups(reg_type(ctx.def->idx),
                            address_frame(N / 2 * sizeof(T))[ptr]);
            }
        }, "load_u_half");

        auto v_raw = ir_def<N>({src_v.vid()}, [this](const jit_kernel_ir::EmitContext& ctx) {
            auto ptr = Xbyak::Reg64(ctx.reads[0].idx);
            if constexpr (std::is_same_v<T, uint8_t>) {
                uni_vpmovzxbd(reg_type(ctx.def->idx), address_frame(N / 2)[ptr]);
                uni_vcvtdq2ps(reg_type(ctx.def->idx), reg_type(ctx.def->idx));
            } else {
                uni_vmovups(reg_type(ctx.def->idx),
                            address_frame(N / 2 * sizeof(T))[ptr]);
            }
        }, "load_v_half");

        // Unpack UV: duplicate each chroma sample [u0,u0,u1,u1,...]
        auto u = u_raw.permute(uv_unpack_order);
        auto v = v_raw.permute(uv_unpack_order);

        // BT.601 conversion (inlined, same math as yuv_to_rgb)
        auto y = vmulps(vsubps(y_raw, y_off), y_scale);
        auto uu = vsubps(u, uv_off);
        auto vv = vsubps(v, uv_off);

        auto r = fma(v_to_r, vv, y);
        auto g = fnma(v_to_g, vv, fnma(u_to_g, uu, y));
        auto b = fma(u_to_b, uu, y);

        // Clamp required for all types: vpmovusdb saturates unsigned
        // (negative → 255, not 0), so clamp before conversion.
        r = r.clamp(clamp_lo, clamp_hi);
        g = g.clamp(clamp_lo, clamp_hi);
        b = b.clamp(clamp_lo, clamp_hi);

        // Color format dispatch
        ir_cmp(colorFormat, size_t{0});
        ir_if(&Xbyak::CodeGenerator::jne,
            [&]() { store_interleaved3(dst, r, g, b); },
            [&]() { store_interleaved3(dst, b, g, r); });

        // Advance pointers — reads all four pointer vids, emits add for each
        ir_use({src_y.vid(), src_u.vid(), src_v.vid(), dst.vid()},
            [this](const jit_kernel_ir::EmitContext& ctx) {
                add(Xbyak::Reg64(ctx.reads[0].idx), step);
                add(Xbyak::Reg64(ctx.reads[1].idx), uv_step);
                add(Xbyak::Reg64(ctx.reads[2].idx), uv_step);
                add(Xbyak::Reg64(ctx.reads[3].idx), 3 * step);
            }, "ptr_advance");
    });

    // ── Tail: partial loads + same IR math ──────────────────
    // ir_load_partial handles runtime-count loads safely (stack+copy).
    // Same BT.601 math as the main loop, reusing the hoisted constants.
    ir_cmp(tail_count, size_t{0});
    ir_if(&Xbyak::CodeGenerator::je, [&]() {
        auto y_raw = ir_load_partial<N>(src_y, tail_count);
        auto u_raw = ir_load_partial<N>(src_u, uv_tail);
        auto v_raw = ir_load_partial<N>(src_v, uv_tail);

        auto u = u_raw.permute(uv_unpack_order);
        auto v = v_raw.permute(uv_unpack_order);

        auto y = vmulps(vsubps(y_raw, y_off), y_scale);
        auto uu = vsubps(u, uv_off);
        auto vv = vsubps(v, uv_off);

        auto r = fma(v_to_r, vv, y);
        auto g = fnma(v_to_g, vv, fnma(u_to_g, uu, y));
        auto b = fma(u_to_b, uu, y);

        // Clamp required for all types: vpmovusdb saturates unsigned
        // (negative → 255, not 0), so clamp before conversion.
        r = r.clamp(clamp_lo, clamp_hi);
        g = g.clamp(clamp_lo, clamp_hi);
        b = b.clamp(clamp_lo, clamp_hi);

        ir_cmp(colorFormat, size_t{0});
        ir_if(&Xbyak::CodeGenerator::jne,
            [&]() { store_interleaved3(dst, r, g, b, tail_count); },
            [&]() { store_interleaved3(dst, b, g, r, tail_count); });
    });

    end_ir();

    postamble();
}

template <typename T>
const jit_uni_converter& jit_converter_create() {
    auto createKernel = []() {
        std::unique_ptr<jit_uni_converter> kernel;

        if (mayiuse(cpu_isa_t::avx512_core)) {
            auto converter = new JitConverter<T[16]>;
            kernel.reset(converter);
            converter->init();
        } else if (mayiuse(cpu_isa_t::avx2)) {
            auto converter = new JitConverter<T[8]>;
            kernel.reset(converter);
            converter->init();
        } else if (mayiuse(cpu_isa_t::sse41)) {
            auto converter = new JitConverter<T[4]>;
            kernel.reset(converter);
            converter->init();
        } else {
            OPENVINO_THROW("Can't create jit color converter kernel");
        }

        return kernel;
    };

    static auto kernel = createKernel();

    return *kernel;
}

template <typename T>
const jit_uni_converter& jit_converter_get() {
    return jit_converter_create<T>();
}

template <typename T>
class SinglePlaneConvert<T, impl_desc_type::jit_uni> : public Converter {
public:
    explicit SinglePlaneConvert(Node* node) : Converter(node) {
        jit_converter_create<T>();
    }

    void execute(const CpuParallelPtr& cpu_parallel, [[maybe_unused]] const dnnl::stream& strm) override {
        const auto& kernel = jit_converter_get<T>();
        const auto& dims = inputDims(0);

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM] * 2 / 3;
        const size_t width = dims[W_DIM];

        const T* y = static_cast<const T*>(input(0));
        const T* u = y + width * height;
        const T* v = y + 5 * width * height / 4;
        T* dst = static_cast<T*>(output(0));

        const size_t stride_y = height * width * 3 / 2;
        const size_t stride_uv = height * width * 3 / 2;

        cpu_parallel->parallel_for2d(batch_size, height, [&](int batch, int h) {
            typename jit_uni_converter::Params args{
                y + batch * stride_y + h * width,                // y
                u + batch * stride_uv + (h / 2) * (width / 2),   // u
                v + batch * stride_uv + (h / 2) * (width / 2),   // v
                dst + (batch * width * height + h * width) * 3,  // dst
                width,                                           // width
                _colorFormat[0]                                  // colorFormat - RGB or BGR format
            };
            kernel(args);
        });
    }
};

template <typename T>
class ThreePlaneConvert<T, impl_desc_type::jit_uni> : public Converter {
public:
    explicit ThreePlaneConvert(Node* node) : Converter(node) {
        jit_converter_create<T>();
    }

    void execute(const CpuParallelPtr& cpu_parallel, [[maybe_unused]] const dnnl::stream& strm) override {
        const auto& kernel = jit_converter_get<T>();
        const auto& dims = inputDims(0);

        const T* y = static_cast<const T*>(input(0));
        const T* u = static_cast<const T*>(input(1));
        const T* v = static_cast<const T*>(input(2));
        T* dst = static_cast<T*>(output(0));

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM];
        const size_t width = dims[W_DIM];

        const size_t stride_y = height * width;
        const size_t stride_uv = height * width / 4;

        cpu_parallel->parallel_for2d(batch_size, height, [&](int batch, int h) {
            typename jit_uni_converter::Params args{
                y + batch * stride_y + h * width,                // y
                u + batch * stride_uv + (h / 2) * (width / 2),   // u
                v + batch * stride_uv + (h / 2) * (width / 2),   // v
                dst + (batch * width * height + h * width) * 3,  // dst
                width,                                           // width
                _colorFormat[0]                                  // colorFormat - RGB or BGR format
            };
            kernel(args);
        });
    }
};
#endif
}  // namespace i420

}  // namespace

ColorConvert::Converter::Converter(Node* node, const ColorFormat& colorFormat)
    : _node(node),
      _colorFormat(colorFormat) {}

ov::element::Type ColorConvert::Converter::inputPrecision(size_t idx) const {
    return _node->getParentEdgeAt(idx)->getMemory().getDesc().getPrecision();
}

ov::element::Type ColorConvert::Converter::outputPrecision(size_t idx) const {
    return _node->getChildEdgeAt(idx)->getMemory().getDesc().getPrecision();
}

const void* ColorConvert::Converter::input(size_t idx) const {
    return _node->getSrcDataAtPort(idx);
}

void* ColorConvert::Converter::output(size_t idx) const {
    return _node->getDstDataAtPort(idx);
}

const VectorDims& ColorConvert::Converter::inputDims(size_t idx) const {
    return _node->getParentEdgeAt(idx)->getMemory().getStaticDims();
}

bool ColorConvert::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    Algorithm alg{};
    std::tie(alg, errorMessage) = getAlgorithmFor(op);
    return alg != Algorithm::Default;
}

ColorConvert::ColorConvert(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, ColorConvertShapeInferFactory(op)) {
    std::string errorMessage;
    std::tie(algorithm, errorMessage) = getAlgorithmFor(op);
    if (algorithm == Algorithm::Default) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void ColorConvert::getSupportedDescriptors() {}

void ColorConvert::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    switch (algorithm) {
    case Algorithm::ColorConvertNV12toRGB:
    case Algorithm::ColorConvertNV12toBGR: {
        for (const auto& desc : nv12::supportedPrimitiveDescs(this)) {
            const auto& inPortConfigs = std::get<0>(desc);
            const auto& outPortConfigs = std::get<1>(desc);
            const auto implType = std::get<2>(desc);
            addSupportedPrimDesc(inPortConfigs, outPortConfigs, implType);
        }
        initSupportedNV12Impls();
        break;
    }
    case Algorithm::ColorConvertI420toRGB:
    case Algorithm::ColorConvertI420toBGR: {
        for (const auto& desc : i420::supportedPrimitiveDescs(this)) {
            const auto& inPortConfigs = std::get<0>(desc);
            const auto& outPortConfigs = std::get<1>(desc);
            const auto implType = std::get<2>(desc);
            addSupportedPrimDesc(inPortConfigs, outPortConfigs, implType);
        }
        initSupportedI420Impls();
        break;
    }
    default:
        break;
    }
}

void ColorConvert::initSupportedNV12Impls() {
#define SUPPORTED_IMPL(Impl, type, desc_type)                         \
    [](Node* node) {                                                  \
        return new nv12::Impl<type, impl_desc_type::desc_type>(node); \
    };

    // ref
    {
        auto& impls = _supportedImpls[impl_desc_type::ref][algorithm];
        impls[ov::element::Type_t::u8][true] = SUPPORTED_IMPL(SinglePlaneConvert, uint8_t, ref);
        impls[ov::element::Type_t::u8][false] = SUPPORTED_IMPL(TwoPlaneConvert, uint8_t, ref);
        impls[ov::element::Type_t::f32][true] = SUPPORTED_IMPL(SinglePlaneConvert, float, ref);
        impls[ov::element::Type_t::f32][false] = SUPPORTED_IMPL(TwoPlaneConvert, float, ref);
    }

#if defined(OPENVINO_ARCH_X86_64)
    // jit_uni
    {
        auto& impls = _supportedImpls[impl_desc_type::jit_uni][algorithm];
        impls[ov::element::Type_t::u8][true] = SUPPORTED_IMPL(SinglePlaneConvert, uint8_t, jit_uni);
        impls[ov::element::Type_t::u8][false] = SUPPORTED_IMPL(TwoPlaneConvert, uint8_t, jit_uni);
        impls[ov::element::Type_t::f32][true] = SUPPORTED_IMPL(SinglePlaneConvert, float, jit_uni);
        impls[ov::element::Type_t::f32][false] = SUPPORTED_IMPL(TwoPlaneConvert, float, jit_uni);
    }
#endif
#undef SUPPORTED_IMPL
}

void ColorConvert::initSupportedI420Impls() {
#define SUPPORTED_IMPL(Impl, type, desc_type)                         \
    [](Node* node) {                                                  \
        return new i420::Impl<type, impl_desc_type::desc_type>(node); \
    };

    // ref
    {
        auto& impls = _supportedImpls[impl_desc_type::ref][algorithm];
        impls[ov::element::Type_t::u8][true] = SUPPORTED_IMPL(SinglePlaneConvert, uint8_t, ref);
        impls[ov::element::Type_t::u8][false] = SUPPORTED_IMPL(ThreePlaneConvert, uint8_t, ref);
        impls[ov::element::Type_t::f32][true] = SUPPORTED_IMPL(SinglePlaneConvert, float, ref);
        impls[ov::element::Type_t::f32][false] = SUPPORTED_IMPL(ThreePlaneConvert, float, ref);
    }

#if defined(OPENVINO_ARCH_X86_64)
    // jit_uni
    {
        auto& impls = _supportedImpls[impl_desc_type::jit_uni][algorithm];
        impls[ov::element::Type_t::u8][true] = SUPPORTED_IMPL(SinglePlaneConvert, uint8_t, jit_uni);
        impls[ov::element::Type_t::u8][false] = SUPPORTED_IMPL(ThreePlaneConvert, uint8_t, jit_uni);
        impls[ov::element::Type_t::f32][true] = SUPPORTED_IMPL(SinglePlaneConvert, float, jit_uni);
        impls[ov::element::Type_t::f32][false] = SUPPORTED_IMPL(ThreePlaneConvert, float, jit_uni);
    }
#endif
#undef SUPPORTED_IMPL
}

void ColorConvert::createPrimitive() {
    const NodeDesc* desc = getSelectedPrimitiveDescriptor();
    CPU_NODE_ASSERT(desc, "has no optimal primitive descriptor selected");

    if (!_impl) {
        const auto& cfg = desc->getConfig();
        const auto precision = cfg.inConfs[0].getMemDesc()->getPrecision();
        const bool isSinglePlane = cfg.inConfs.size() == 1;

        _impl = std::unique_ptr<Converter>(
            _supportedImpls.at(desc->getImplementationType()).at(algorithm).at(precision).at(isSinglePlane)(this));
    }
}

void ColorConvert::execute(const dnnl::stream& strm) {
    CPU_NODE_ASSERT(_impl, "has no any implemented converter");
    _impl->execute(context->getCpuParallel(), strm);
}

bool ColorConvert::created() const {
    return getType() == Type::ColorConvert;
}

bool ColorConvert::needPrepareParams() const {
    return false;
}

void ColorConvert::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

}  // namespace ov::intel_cpu::node
