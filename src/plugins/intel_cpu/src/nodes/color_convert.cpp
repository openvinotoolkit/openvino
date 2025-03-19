// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "color_convert.h"

#include <memory_desc/dnnl_blocked_memory_desc.h>

#include <openvino/core/type.hpp>
#include <openvino/op/i420_to_bgr.hpp>
#include <openvino/op/i420_to_rgb.hpp>
#include <openvino/op/nv12_to_bgr.hpp>
#include <openvino/op/nv12_to_rgb.hpp>

#include "kernels/x64/jit_kernel.hpp"
#include "openvino/core/parallel.hpp"
#include "shape_inference/custom/color_convert.hpp"

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
    Converter(Node* node);

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
    auto c = y - 16.f;
    auto d = u - 128.f;
    auto e = v - 128.f;
    auto clip = [](float a) -> T {
        if (std::is_integral<T>()) {
            return static_cast<T>(std::min(std::max(std::round(a), 0.f), 255.f));
        }
        return static_cast<T>(std::min(std::max(a, 0.f), 255.f));
    };
    auto r = clip(1.164f * c + 1.596f * e);
    auto g = clip(1.164f * c - 0.391f * d - 0.813f * e);
    auto b = clip(1.164f * c + 2.018f * d);
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

    template <size_t N>
    void yuv_to_rgb(const variable<float[N]>& y,
                    const variable<float[N]>& u,
                    const variable<float[N]>& v,
                    const variable<uint8_t>& color_format,
                    bool round);
    template <typename T, size_t N>
    void store_tail(const variable<T*>& dst,
                    const variable<float[N]>& a,
                    const variable<float[N]>& b,
                    const variable<float[N]>& c,
                    const variable<size_t>& size);

    function_t _fn;
    variable<const float*> _consts;
};

jit_uni_converter::jit_uni_converter() : jit_kernel(jit_name()), _consts(*this) {}

void jit_uni_converter::init() {
    if (create_kernel() != status::success) {
        OPENVINO_THROW("Can't generate jit color converter kernel");
    }
    _fn = (function_t)jit_ker();
}

template <size_t N>
void jit_uni_converter::yuv_to_rgb(const variable<float[N]>& y,
                                   const variable<float[N]>& u,
                                   const variable<float[N]>& v,
                                   const variable<uint8_t>& color_format,
                                   bool round) {
    auto clip = [&](const variable<float[N]>& op, const variable<float[N]>& a, const variable<float[N]>& b) {
        if (round) {
            uni_vroundps(op, op, 0);
        }
        uni_vmaxps(op, op, a);
        uni_vminps(op, op, b);
    };

    // blend r,g,b and put to r0,r1,r2
    auto blend = [&](const variable<float[N]>& r,
                     const variable<float[N]>& g,
                     const variable<float[N]>& b,
                     const variable<float[N]>& r0,
                     const variable<float[N]>& r1,
                     const variable<float[N]>& r2) {
        /*
            Input:
            r0,r1,r2,r3,r4,r5,r6,r7
            g0,g1,g2,g3,g4,g5,g6,g7
            b0,b1,b2,b3,b4,b5,b6,b7

            Permutation:
            r0,r3,r6,r1,r4,r7,r2,r5
            g5,g0,g3,g6,g1,g4,g7,g2
            b2,b5,b0,b3,b6,b1,b4,b7

            Blend
            r0,g0,xx,r1,g1,xx,r2,g2     blend 1+2 by mask 10210210
            r0,g0,b0,r1,g1,b1,r2,g2     blend +3  by mask 00100100

            xx,r3,g3,xx,r4,g4,xx,r5     blend 1+2 by mask 02102102
            b2,r3,g3,b3,r4,g4,b4,r5     blend +3  by mask 01001001

            g5,xx,r6,g6,xx,r7,g7,xx     blend 1+2 by mask 21021021
            g5,b5,r6,g6,b6,r7,g7,b7     blend +3  by mask 10010010

            Result
            a = r0,g0,b0,r1,g1,b1,r2,g2
            b = b2,r3,g3,b3,r4,g4,b4,r5
            c = g5,b5,r6,g6,b6,r7,g7,b7
        */

        auto genPermutationMask = [&](int offset) {
            std::array<uint8_t, N> mask{};
            for (size_t i = 0; i < mask.size(); ++i) {
                mask[(i * 3 + offset) % mask.size()] = i;
            }
            return mask;
        };

        r.permute(genPermutationMask(0));
        g.permute(genPermutationMask(1));
        b.permute(genPermutationMask(2));

        auto blendWithMask = [&](int offset, const variable<float[N]>& result) {
            static const uint32_t blendMasks[2] = {0x92492492, 0x24924924};
            const auto mask0 = static_cast<const uint16_t>(blendMasks[0] >> ((offset * N) % 3));
            const auto mask1 = static_cast<const uint16_t>(blendMasks[1] >> ((offset * N) % 3));

            result = r;
            result.blend(g, mask0);
            result.blend(b, mask1);
        };

        blendWithMask(0, r0);
        blendWithMask(1, r1);
        blendWithMask(2, r2);
    };  // blend

    // Reserve registers
    auto r = var<float[N]>();
    auto g = var<float[N]>();
    auto b = var<float[N]>();
    auto tmp = var<float[N]>();

    uni_vbroadcastss(tmp, ptr[_consts + 0 * sizeof(float)]);  // tmp = [16.0f,16.0f,...]
    uni_vsubps(y, y, tmp);                                    // y = y - tmp
    uni_vbroadcastss(tmp, ptr[_consts + 1 * sizeof(float)]);  // tmp = [128.f,128.f,...]
    uni_vsubps(u, u, tmp);                                    // u = u - tmp
    uni_vsubps(v, v, tmp);                                    // v = v - tmp

    uni_vbroadcastss(tmp, ptr[_consts + 2 * sizeof(float)]);  // tmp = [1.164f,1.164f,...]
    uni_vmulps(y, y, tmp);                                    // y = y * tmp

    uni_vbroadcastss(r, ptr[_consts + 3 * sizeof(float)]);  // r = [1.596f,1.596f,...]
    uni_vmulps(r, r, v);                                    // r = r * v
    uni_vaddps(r, r, y);                                    // r = r + y

    uni_vbroadcastss(g, ptr[_consts + 4 * sizeof(float)]);    // g = [0.391f,0.391f,...]
    uni_vmulps(g, g, u);                                      // g = g * u
    uni_vsubps(g, y, g);                                      // g = y - g
    uni_vbroadcastss(tmp, ptr[_consts + 6 * sizeof(float)]);  // tmp = [0.813f,0.813f,...]
    uni_vmulps(tmp, tmp, v);                                  // tmp = tmp * v
    uni_vsubps(g, g, tmp);                                    // g = g - tmp

    uni_vbroadcastss(b, ptr[_consts + 5 * sizeof(float)]);  // b = [2.018f,2.018f,...]
    uni_vmulps(b, b, u);                                    // b = b * u
    uni_vaddps(b, b, y);                                    // b = b + y

    // clip
    uni_vxorps(y, y, y);
    uni_vbroadcastss(u, ptr[_consts + 7 * sizeof(float)]);

    clip(r, y, u);
    clip(g, y, u);
    clip(b, y, u);

    _if(color_format == 0)
        ._then([&] {
            blend(r, g, b, y, u, v);
        })
        ._else([&] {
            blend(b, g, r, y, u, v);
        });
}

template <typename T, size_t N>
void jit_uni_converter::store_tail(const variable<T*>& dst,
                                   const variable<float[N]>& a,
                                   const variable<float[N]>& b,
                                   const variable<float[N]>& c,
                                   const variable<size_t>& size) {
    const size_t step = N * sizeof(T);
    auto s = stack(3 * step);

    auto sptr = var<T*>();
    sptr = s.pointer();

    store(sptr, a);
    sptr += step;
    store(sptr, b);
    sptr += step;
    store(sptr, c);

    auto copy_size = size * static_cast<size_t>(3u);

    copy<T>(ptr[dst], s.pointer(), copy_size);
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
    RefConverter(Node* node);

protected:
    template <typename T>
    void convert(const T* y,
                 const T* uv,
                 T* dst,
                 size_t batch_size,
                 size_t height,
                 size_t width,
                 size_t stride_y,
                 size_t stride_uv);
};

RefConverter::RefConverter(Node* node) : Converter(node) {
    if (node->getOriginalInputsNumber() != (singlePlane() ? 1 : 2)) {
        OPENVINO_THROW("NV12Converter node has incorrect number of inputs");
    }
    if (!node->getOriginalOutputsNumber()) {
        OPENVINO_THROW("NV12Converter node has incorrect number of outputs");
    }
}

template <typename T>
void RefConverter::convert(const T* y,
                           const T* uv,
                           T* dst,
                           size_t batch_size,
                           size_t height,
                           size_t width,
                           size_t stride_y,
                           size_t stride_uv) {
    ov::parallel_for2d(batch_size, height, [&](int batch, int h) {
        T* out = dst + batch * width * height * 3;
        auto y_ptr = y + batch * stride_y;
        auto uv_ptr = uv + batch * stride_uv;

        for (size_t w = 0; w < width; w++) {
            auto y_index = h * width + w;
            auto y_val = static_cast<float>(y_ptr[y_index]);
            auto uv_index = (h / 2) * width + (w / 2) * 2;
            auto u_val = static_cast<float>(uv_ptr[uv_index]);
            auto v_val = static_cast<float>(uv_ptr[uv_index + 1]);
            T r, g, b;
            std::tie(r, g, b) = yuv_to_rgb<T>(y_val, u_val, v_val);
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

    void execute(const dnnl::stream& strm) override {
        const auto& dims = inputDims(0);

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM] * 2 / 3;
        const size_t width = dims[W_DIM];

        const T* y = static_cast<const T*>(input(0));
        const T* uv = y + width * height;
        T* dst = static_cast<T*>(output(0));

        convert<T>(y, uv, dst, batch_size, height, width, height * width * 3 / 2, height * width * 3 / 2);
    }
};

template <typename T>
class TwoPlaneConvert<T, impl_desc_type::ref> : public RefConverter {
public:
    using RefConverter::RefConverter;

    void execute(const dnnl::stream& strm) override {
        const auto& dims = inputDims(0);

        const T* y = static_cast<const T*>(input(0));
        const T* uv = static_cast<const T*>(input(1));
        T* dst = static_cast<T*>(output(0));

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM];
        const size_t width = dims[W_DIM];

        convert<T>(y, uv, dst, batch_size, height, width, height * width, height * width / 2);
    }
};

#if defined(OPENVINO_ARCH_X86_64)
template <typename T>
class JitConverter;

template <typename T, size_t N>
class JitConverter<T[N]> : public jit_uni_converter {
private:
    void generate() override;
    std::tuple<variable<float[N]>, variable<float[N]>, variable<float[N]>> load_yuv(const variable<const T*>& src_y,
                                                                                    const variable<const T*>& src_uv);
    std::tuple<variable<float[N]>, variable<float[N]>> unpack_uv(const variable<float[N]>& uv);
};

template <typename T, size_t N>
void JitConverter<T[N]>::generate() {
    preamble();

    // Get arguments addresses
    auto src_y = arg<const T*>(&Params::y);
    auto src_uv = arg<const T*>(&Params::u);
    auto dst = arg<T*>(&Params::dst);
    auto width = arg(&Params::width);
    auto colorFormat = arg(&Params::colorFormat);

    static const float data[8] = {16.f, 128.f, 1.164f, 1.596f, 0.391f, 2.018f, 0.813f, 255.f};
    _consts = data;

    const auto reg_capacity_log = static_cast<size_t>(std::logb(N));
    const size_t step = N * sizeof(T);

    width >>= reg_capacity_log;

    foreach (0, width, [&](const Reg64& idx) {
        auto yuv = load_yuv(src_y, src_uv);

        // Aliases
        const auto& y = std::get<0>(yuv);
        const auto& u = std::get<1>(yuv);
        const auto& v = std::get<2>(yuv);

        yuv_to_rgb(y, u, v, colorFormat, std::is_integral<T>::value);

        store(dst, y);
        dst += step;
        store(dst, u);
        dst += step;
        store(dst, v);
        dst += step;
    })
        ;

    mov(width, argPtr(&Params::width));
    width &= N - 1;

    _if(width != 0)._then([&] {
        auto y = var<float[N]>();
        auto uv = var<float[N]>();

        load(y, src_y, width);
        load(uv, src_uv, width);

        auto uv_pair = unpack_uv(uv);

        // Aliases
        const auto& u = std::get<0>(uv_pair);
        const auto& v = std::get<1>(uv_pair);

        yuv_to_rgb(y, u, v, colorFormat, std::is_integral<T>::value);

        store_tail(dst, y, u, v, width);
    });

    postamble();
}

template <typename T, size_t N>
std::tuple<jit_kernel::variable<float[N]>, jit_kernel::variable<float[N]>, jit_kernel::variable<float[N]>>
JitConverter<T[N]>::load_yuv(const variable<const T*>& src_y, const variable<const T*>& src_uv) {
    auto y = var<float[N]>();
    auto uv = var<float[N]>();

    load(y, src_y);
    load(uv, src_uv);

    auto uv_pair = unpack_uv(uv);

    src_y += N * sizeof(T);
    src_uv += N * sizeof(T);

    return std::make_tuple(std::move(y), std::move(std::get<0>(uv_pair)), std::move(std::get<1>(uv_pair)));
}

template <typename T, size_t N>
std::tuple<jit_kernel::variable<float[N]>, jit_kernel::variable<float[N]>> JitConverter<T[N]>::unpack_uv(
    const variable<float[N]>& uv) {
    auto u = var<float[N]>();
    auto v = var<float[N]>();

    const uint8_t even_mask = 0xA0;  // 0b10100000
    const uint8_t odd_mask = 0xF5;   // 0b11110101

    uni_vshufps(u, uv, uv, even_mask);  // u = uv[0,0,2,2,4,4,6,6]
    uni_vshufps(v, uv, uv, odd_mask);   // v = uv[1,1,3,3,5,5,7,7]

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
    SinglePlaneConvert(Node* node) : Converter(node) {
        jit_converter_create<T>();
    }

    void execute(const dnnl::stream& strm) override {
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

        ov::parallel_for2d(batch_size, height, [&](int batch, int h) {
            typename jit_uni_converter::Params args;
            args.y = y + batch * stride_y + h * width;
            args.u = args.v = uv + batch * stride_uv + (h / 2) * width;
            args.dst = dst + (batch * width * height + h * width) * 3;
            args.width = width;
            args.colorFormat = _colorFormat[0];  // The first byte is enough to determine the RGB or BGR format.
            kernel(args);
        });
    }
};

template <typename T>
class TwoPlaneConvert<T, impl_desc_type::jit_uni> : public Converter {
public:
    TwoPlaneConvert(Node* node) : Converter(node) {
        jit_converter_create<T>();
    }

    void execute(const dnnl::stream& strm) override {
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

        ov::parallel_for2d(batch_size, height, [&](int batch, int h) {
            typename jit_uni_converter::Params args;
            args.y = y + batch * stride_y + h * width;
            args.u = args.v = uv + batch * stride_uv + (h / 2) * width;
            args.dst = dst + (batch * width * height + h * width) * 3;
            args.width = width;
            args.colorFormat = _colorFormat[0];  // The first byte is enough to determine the RGB or BGR format.
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
    RefConverter(Node* node);

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
                 size_t stride_uv);
};

RefConverter::RefConverter(Node* node) : Converter(node) {
    if (node->getOriginalInputsNumber() != (singlePlane() ? 1 : 3)) {
        OPENVINO_THROW("I420Converter node has incorrect number of inputs");
    }
    if (!node->getOriginalOutputsNumber()) {
        OPENVINO_THROW("I420Converter node has incorrect number of outputs");
    }
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
                           size_t stride_uv) {
    ov::parallel_for2d(batch_size, height, [&](int batch, int h) {
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
            T r, g, b;
            std::tie(r, g, b) = yuv_to_rgb<T>(y_val, u_val, v_val);
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

    void execute(const dnnl::stream& strm) override {
        const auto& dims = inputDims(0);

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM] * 2 / 3;
        const size_t width = dims[W_DIM];

        const T* y = static_cast<const T*>(input(0));
        const T* u = y + width * height;
        const T* v = y + 5 * width * height / 4;
        T* dst = static_cast<T*>(output(0));

        convert<T>(y, u, v, dst, batch_size, height, width, height * width * 3 / 2, height * width * 3 / 2);
    }
};

template <typename T>
class ThreePlaneConvert<T, impl_desc_type::ref> : public RefConverter {
public:
    using RefConverter::RefConverter;

    void execute(const dnnl::stream& strm) override {
        const auto& dims = inputDims(0);

        const T* y = static_cast<const T*>(input(0));
        const T* u = static_cast<const T*>(input(1));
        const T* v = static_cast<const T*>(input(2));
        T* dst = static_cast<T*>(output(0));

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM];
        const size_t width = dims[W_DIM];

        convert<T>(y, u, v, dst, batch_size, height, width, height * width, height * width / 4);
    }
};

#if defined(OPENVINO_ARCH_X86_64)
template <typename T>
class JitConverter;

template <typename T, size_t N>
class JitConverter<T[N]> : public jit_uni_converter {
private:
    void generate() override;
    std::tuple<variable<float[N]>, variable<float[N]>, variable<float[N]>> load_yuv(const variable<const T*>& src_y,
                                                                                    const variable<const T*>& src_u,
                                                                                    const variable<const T*>& src_v);
    void unpack_uv(const variable<float[N]>& u, const variable<float[N]>& v);
};

template <typename T, size_t N>
void JitConverter<T[N]>::generate() {
    preamble();

    // Get arguments addresses
    auto src_y = arg<const T*>(&Params::y);
    auto src_u = arg<const T*>(&Params::u);
    auto src_v = arg<const T*>(&Params::v);
    auto dst = arg<T*>(&Params::dst);
    auto width = arg(&Params::width);
    auto colorFormat = arg(&Params::colorFormat);

    static const float data[8] = {16.f, 128.f, 1.164f, 1.596f, 0.391f, 2.018f, 0.813f, 255.f};
    _consts = data;

    const auto reg_capacity_log = static_cast<size_t>(std::logb(N));
    const size_t step = N * sizeof(T);

    width >>= reg_capacity_log;

    foreach (0, width, [&](const Reg64& idx) {
        auto yuv = load_yuv(src_y, src_u, src_v);

        // Aliases
        const auto& y = std::get<0>(yuv);
        const auto& u = std::get<1>(yuv);
        const auto& v = std::get<2>(yuv);

        yuv_to_rgb(y, u, v, colorFormat, std::is_integral<T>::value);

        store(dst, y);
        dst += step;
        store(dst, u);
        dst += step;
        store(dst, v);
        dst += step;
    })
        ;

    mov(width, argPtr(&Params::width));
    width &= N - 1;

    _if(width != 0)._then([&] {
        auto y = var<float[N]>();
        auto u = var<float[N]>();
        auto v = var<float[N]>();

        auto uv_width = width >> 1;

        load(y, src_y, width);
        load(u, src_u, uv_width);
        load(v, src_v, uv_width);

        unpack_uv(u, v);

        yuv_to_rgb(y, u, v, colorFormat, std::is_integral<T>::value);

        store_tail(dst, y, u, v, width);
    });

    postamble();
}

template <typename T, size_t N>
std::tuple<jit_kernel::variable<float[N]>, jit_kernel::variable<float[N]>, jit_kernel::variable<float[N]>>
JitConverter<T[N]>::load_yuv(const variable<const T*>& src_y,
                             const variable<const T*>& src_u,
                             const variable<const T*>& src_v) {
    auto y = var<float[N]>();
    auto u = var<float[N]>();
    auto v = var<float[N]>();

    load(y, src_y);
    load(u, src_u, N / 2);
    load(v, src_v, N / 2);

    unpack_uv(u, v);

    src_y += N * sizeof(T);
    src_u += N * sizeof(T) / 2;
    src_v += N * sizeof(T) / 2;

    return std::make_tuple(std::move(y), std::move(u), std::move(v));
}

template <typename T, size_t N>
void JitConverter<T[N]>::unpack_uv(const variable<float[N]>& u, const variable<float[N]>& v) {
    static const uint8_t order[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};
    u.permute(order);
    v.permute(order);
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
    SinglePlaneConvert(Node* node) : Converter(node) {
        jit_converter_create<T>();
    }

    void execute(const dnnl::stream& strm) override {
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

        ov::parallel_for2d(batch_size, height, [&](int batch, int h) {
            typename jit_uni_converter::Params args;
            args.y = y + batch * stride_y + h * width;
            args.u = u + batch * stride_uv + (h / 2) * (width / 2);
            args.v = v + batch * stride_uv + (h / 2) * (width / 2);
            args.dst = dst + (batch * width * height + h * width) * 3;
            args.width = width;
            args.colorFormat = _colorFormat[0];  // The first byte is enough to determine the RGB or BGR format.
            kernel(args);
        });
    }
};

template <typename T>
class ThreePlaneConvert<T, impl_desc_type::jit_uni> : public Converter {
public:
    ThreePlaneConvert(Node* node) : Converter(node) {
        jit_converter_create<T>();
    }

    void execute(const dnnl::stream& strm) override {
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

        ov::parallel_for2d(batch_size, height, [&](int batch, int h) {
            typename jit_uni_converter::Params args;
            args.y = y + batch * stride_y + h * width;
            args.u = u + batch * stride_uv + (h / 2) * (width / 2);
            args.v = v + batch * stride_uv + (h / 2) * (width / 2);
            args.dst = dst + (batch * width * height + h * width) * 3;
            args.width = width;
            args.colorFormat = _colorFormat[0];  // The first byte is enough to determine the RGB or BGR format.
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
    Algorithm alg;
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
    if (!desc) {
        THROW_CPU_NODE_ERR("has no optimal primitive descriptor selected");
    }

    if (!_impl) {
        const auto& cfg = desc->getConfig();
        const auto precision = cfg.inConfs[0].getMemDesc()->getPrecision();
        const bool isSinglePlane = cfg.inConfs.size() == 1;

        _impl = std::unique_ptr<Converter>(
            _supportedImpls.at(desc->getImplementationType()).at(algorithm).at(precision).at(isSinglePlane)(this));
    }
}

void ColorConvert::execute(const dnnl::stream& strm) {
    if (!_impl) {
        THROW_CPU_NODE_ERR("has no any implemented converter");
    }
    _impl->execute(strm);
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
