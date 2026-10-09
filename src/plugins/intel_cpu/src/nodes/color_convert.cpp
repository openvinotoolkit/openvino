// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "color_convert.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <openvino/core/type.hpp>
#include <openvino/op/bgr_to_nv12.hpp>
#include <openvino/op/i420_to_bgr.hpp>
#include <openvino/op/i420_to_rgb.hpp>
#include <openvino/op/nv12_to_bgr.hpp>
#include <openvino/op/nv12_to_rgb.hpp>
#include <openvino/op/rgb_to_nv12.hpp>
#include <string>
#include <tuple>
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
#include "openvino/reference/utils/convert_color_util.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "shape_inference/custom/color_convert.hpp"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <xbyak/xbyak.h>

#    include <array>
#    include <common/c_types_map.hpp>
#    include <cpu/x64/jit_generator.hpp>
#    include <cpu/x64/cpu_isa_traits.hpp>

#    include "kernels/x64/jit_kernel.hpp"
#endif

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;
#endif

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
    if (ov::is_type<ov::op::v17::RGBtoNV12>(op)) {
        return std::make_tuple(Algorithm::ColorConvertRGBtoNV12, std::string());
    }
    if (ov::is_type<ov::op::v17::BGRtoNV12>(op)) {
        return std::make_tuple(Algorithm::ColorConvertBGRtoNV12, std::string());
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
                   node->getAlgorithm() == Algorithm::ColorConvertI420toRGB ||
                   node->getAlgorithm() == Algorithm::ColorConvertRGBtoNV12
               ? ColorFormat{{0, 1, 2}}
               : ColorFormat{{2, 1, 0}}) {}

bool Converter::singlePlane() const {
    return _node->getOriginalInputsNumber() == 1;
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

    function_t _fn = nullptr;
    variable<const float*> _consts;
};

jit_uni_converter::jit_uni_converter() : jit_kernel(jit_name()), _consts(*this) {}

void jit_uni_converter::init() {
    OPENVINO_ASSERT(create_kernel() == status::success, "Can't generate jit color converter kernel");
    _fn = reinterpret_cast<function_t>(const_cast<uint8_t*>(jit_ker()));
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

        std::ignore = r.permute(genPermutationMask(0));
        std::ignore = g.permute(genPermutationMask(1));
        std::ignore = b.permute(genPermutationMask(2));

        auto blendWithMask = [&](int offset, const variable<float[N]>& result) {
            static const uint32_t blendMasks[2] = {0x92492492, 0x24924924};
            const auto mask0 = static_cast<uint16_t>(blendMasks[0] >> ((offset * N) % 3));
            const auto mask1 = static_cast<uint16_t>(blendMasks[1] >> ((offset * N) % 3));

            result = r;
            std::ignore = result.blend(g, mask0);
            std::ignore = result.blend(b, mask1);
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
    uni_vbroadcastss(tmp, ptr[_consts + 1 * sizeof(float)]);  // tmp = [128.F,128.F,...]
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

    auto copy_size = size * static_cast<size_t>(3U);

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
                       ov::with_cpu_x86_sse42() ? impl_desc_type::jit_uni : impl_desc_type::ref,
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
            auto [r, g, b] = ov::reference::yuv_pixel_to_rgb<T>(y_val, u_val, v_val);
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

    static const float data[8] = {16.F, 128.F, 1.164F, 1.596F, 0.391F, 2.018F, 0.813F, 255.F};
    _consts = data;

    const auto reg_capacity_log = static_cast<size_t>(std::logb(N));
    const size_t step = N * sizeof(T);

    width >>= reg_capacity_log;

    foreach (0, width, [&]([[maybe_unused]] const Reg64& idx) {
        auto yuv = load_yuv(src_y, src_uv);

        // Aliases
        const auto& y = std::get<0>(yuv);
        const auto& u = std::get<1>(yuv);
        const auto& v = std::get<2>(yuv);

        yuv_to_rgb(y, u, v, colorFormat, std::is_integral_v<T>);

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

        yuv_to_rgb(y, u, v, colorFormat, std::is_integral_v<T>);

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

        if (ov::with_cpu_x86_avx512_core()) {
            auto converter = new JitConverter<T[16]>;
            kernel.reset(converter);
            converter->init();
        } else if (ov::with_cpu_x86_avx2()) {
            auto converter = new JitConverter<T[8]>;
            kernel.reset(converter);
            converter->init();
        } else if (ov::with_cpu_x86_sse42()) {
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
                       ov::with_cpu_x86_sse42() ? impl_desc_type::jit_uni : impl_desc_type::ref,
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
            auto [r, g, b] = ov::reference::yuv_pixel_to_rgb<T>(y_val, u_val, v_val);
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

    static const float data[8] = {16.F, 128.F, 1.164F, 1.596F, 0.391F, 2.018F, 0.813F, 255.F};
    _consts = data;

    const auto reg_capacity_log = static_cast<size_t>(std::logb(N));
    const size_t step = N * sizeof(T);

    width >>= reg_capacity_log;

    foreach (0, width, [&]([[maybe_unused]] const Reg64& idx) {
        auto yuv = load_yuv(src_y, src_u, src_v);

        // Aliases
        const auto& y = std::get<0>(yuv);
        const auto& u = std::get<1>(yuv);
        const auto& v = std::get<2>(yuv);

        yuv_to_rgb(y, u, v, colorFormat, std::is_integral_v<T>);

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

        yuv_to_rgb(y, u, v, colorFormat, std::is_integral_v<T>);

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
    std::ignore = u.permute(order);
    std::ignore = v.permute(order);
}

template <typename T>
const jit_uni_converter& jit_converter_create() {
    auto createKernel = []() {
        std::unique_ptr<jit_uni_converter> kernel;

        if (ov::with_cpu_x86_avx512_core()) {
            auto converter = new JitConverter<T[16]>;
            kernel.reset(converter);
            converter->init();
        } else if (ov::with_cpu_x86_avx2()) {
            auto converter = new JitConverter<T[8]>;
            kernel.reset(converter);
            converter->init();
        } else if (ov::with_cpu_x86_sse42()) {
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

namespace to_nv12 {

ColorConvert::Converter::PrimitiveDescs supportedPrimitiveDescs(Node* node) {
    const LayoutType layout = LayoutType::ncsp;

    const ov::element::Type precision =
        node->getOriginalInputPrecisionAtPort(0) == ov::element::u8 ? ov::element::u8 : ov::element::f32;

    ColorConvert::Converter::PrimitiveDescs descs;

    std::vector<PortConfigurator> outConfigs(node->getOriginalOutputsNumber(), PortConfigurator{layout, precision});

    descs.emplace_back(std::vector<PortConfigurator>{{layout, precision}},
                       outConfigs,
                       ov::with_cpu_x86_sse42() ? impl_desc_type::jit_uni : impl_desc_type::ref,
                       true);

    return descs;
}

template <typename T, impl_desc_type I>
class SinglePlaneConvert;
template <typename T, impl_desc_type I>
class TwoPlaneConvert;

class RefConverter : public ColorConvert::Converter {
    using Base = ColorConvert::Converter;

public:
    explicit RefConverter(Node* node)
        : Base(node,
               // RGBtoNV12: R is channel 0;  BGRtoNV12: R is channel 2.
               node->getAlgorithm() == Algorithm::ColorConvertRGBtoNV12 ? ColorFormat{{0, 1, 2}}
                                                                        : ColorFormat{{2, 1, 0}}) {
        OPENVINO_ASSERT(node->getOriginalInputsNumber() == 1, "RGBtoNV12/BGRtoNV12 node must have exactly 1 input");
        const auto nout = node->getOriginalOutputsNumber();
        OPENVINO_ASSERT(nout == 1 || nout == 2, "RGBtoNV12/BGRtoNV12 node must have 1 or 2 outputs");
    }

protected:
    template <typename T>
    void convert(const T* src,
                 T* dst_y,
                 T* dst_uv,
                 size_t batch_size,
                 size_t height,
                 size_t width,
                 size_t stride_in,
                 size_t stride_y,
                 size_t stride_uv,
                 const CpuParallelPtr& cpu_parallel) {
        const size_t r_idx = _colorFormat[0];  // RGB: 0, BGR: 2
        const size_t g_idx = _colorFormat[1];  // always 1
        const size_t b_idx = _colorFormat[2];  // RGB: 2, BGR: 0

        // Process pairs of rows so UV (2x2 subsampled) can be averaged over all 4 pixels.
        cpu_parallel->parallel_for2d(batch_size, height / 2, [&](int batch, int half_h) {
            const size_t h0 = static_cast<size_t>(half_h) * 2;
            const size_t h1 = h0 + 1;

            const T* src0 = src + static_cast<size_t>(batch) * stride_in + h0 * width * 3;
            const T* src1 = src + static_cast<size_t>(batch) * stride_in + h1 * width * 3;
            T* y_out0 = dst_y + static_cast<size_t>(batch) * stride_y + h0 * width;
            T* y_out1 = dst_y + static_cast<size_t>(batch) * stride_y + h1 * width;

            T* uv_out = dst_uv + static_cast<size_t>(batch) * stride_uv + static_cast<size_t>(half_h) * width;

            for (size_t w = 0; w < width; w += 2) {
                double u_sum = 0.0, v_sum = 0.0;

                auto process_pixel = [&](const T* row, T* y_row, size_t col) {
                    T y_val, u_val, v_val;
                    std::tie(y_val, u_val, v_val) = ov::reference::rgb_pixel_to_yuv<T>(row[(col * 3) + r_idx],
                                                                                       row[(col * 3) + g_idx],
                                                                                       row[(col * 3) + b_idx]);
                    y_row[col] = y_val;
                    u_sum += static_cast<double>(u_val);
                    v_sum += static_cast<double>(v_val);
                };

                process_pixel(src0, y_out0, w);
                process_pixel(src0, y_out0, w + 1);
                process_pixel(src1, y_out1, w);
                process_pixel(src1, y_out1, w + 1);

                uv_out[w] = ov::reference::round_cast<T>(u_sum / 4.0);      // U
                uv_out[w + 1] = ov::reference::round_cast<T>(v_sum / 4.0);  // V
            }
        });
    }
};

template <typename T>
class SinglePlaneConvert<T, impl_desc_type::ref> : public RefConverter {
public:
    using RefConverter::RefConverter;

    void execute(const CpuParallelPtr& cpu_parallel, [[maybe_unused]] const dnnl::stream& strm) override {
        const auto& in_dims = inputDims(0);
        const size_t batch_size = in_dims[N_DIM];
        const size_t height = in_dims[H_DIM];
        const size_t width = in_dims[W_DIM];

        const T* src = static_cast<const T*>(input(0));
        T* dst = static_cast<T*>(output(0));

        const size_t out_stride = height * width * 3 / 2;

        convert<T>(src,
                   dst,
                   dst + height * width,
                   batch_size,
                   height,
                   width,
                   height * width * 3,
                   out_stride,
                   out_stride,
                   cpu_parallel);
    }
};

template <typename T>
class TwoPlaneConvert<T, impl_desc_type::ref> : public RefConverter {
public:
    using RefConverter::RefConverter;

    void execute(const CpuParallelPtr& cpu_parallel, [[maybe_unused]] const dnnl::stream& strm) override {
        const auto& in_dims = inputDims(0);
        const size_t batch_size = in_dims[N_DIM];
        const size_t height = in_dims[H_DIM];
        const size_t width = in_dims[W_DIM];

        const T* src = static_cast<const T*>(input(0));
        T* dst_y = static_cast<T*>(output(0));
        T* dst_uv = static_cast<T*>(output(1));

        convert<T>(src,
                   dst_y,
                   dst_uv,
                   batch_size,
                   height,
                   width,
                   height * width * 3,
                   height * width,
                   height * width / 2,
                   cpu_parallel);
    }
};

#if defined(OPENVINO_ARCH_X86_64)

struct jit_to_nv12_converter : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_to_nv12_converter)
    struct Params {
        const void* src0;  // interleaved RGB/BGR row 0
        const void* src1;  // interleaved RGB/BGR row 1
        void* dst_y0;      // Y output for row 0
        void* dst_y1;      // Y output for row 1
        void* dst_uv;      // interleaved UV output (width/2 pairs)
        size_t width;      // pixels per row
        uint8_t colorFmt;  // 0 = RGB input, else BGR input
    };

    using function_t = void (*)(const Params*);

    void init();

    void operator()(const Params& args) const {
        _fn(&args);
    }

protected:
    jit_to_nv12_converter();

    template <size_t N>
    void deinterleave(const variable<float[N]>& v0,
                      const variable<float[N]>& v1,
                      const variable<float[N]>& v2,
                      const variable<float[N]>& ch0,
                      const variable<float[N]>& ch1,
                      const variable<float[N]>& ch2);

    template <size_t N>
    void rgb_to_yuv(const variable<float[N]>& r,
                    const variable<float[N]>& g,
                    const variable<float[N]>& b,
                    const variable<float[N]>& y_out,
                    const variable<float[N]>& u_out,
                    const variable<float[N]>& v_out,
                    bool do_round);

    function_t _fn = nullptr;
    variable<const float*> _consts;
};

jit_to_nv12_converter::jit_to_nv12_converter() : jit_kernel(jit_name()), _consts(*this) {}

void jit_to_nv12_converter::init() {
    OPENVINO_ASSERT(create_kernel() == status::success, "Can't generate jit RGB-to-NV12 converter kernel");
    _fn = reinterpret_cast<function_t>(const_cast<uint8_t*>(jit_ker()));
}

template <size_t N>
void jit_to_nv12_converter::deinterleave(const variable<float[N]>& v0,
                                         const variable<float[N]>& v1,
                                         const variable<float[N]>& v2,
                                         const variable<float[N]>& ch0,
                                         const variable<float[N]>& ch1,
                                         const variable<float[N]>& ch2) {
    auto extract_ch = [&](int ch, const variable<float[N]>& out) {
        std::array<uint8_t, N> perm0{};
        std::array<uint8_t, N> perm1{};
        std::array<uint8_t, N> perm2{};
        uint16_t mask_b = 0;
        uint16_t mask_c = 0;

        for (size_t p = 0; p < N; ++p) {
            size_t idx = (3 * p) + static_cast<size_t>(ch);
            size_t reg_id = idx / N;
            size_t lane = idx % N;

            if (reg_id == 0) {
                perm0[p] = static_cast<uint8_t>(lane);
            } else if (reg_id == 1) {
                perm1[p] = static_cast<uint8_t>(lane);
                mask_b |= static_cast<uint16_t>(1U << p);
            } else {
                perm2[p] = static_cast<uint8_t>(lane);
                mask_c |= static_cast<uint16_t>(1U << p);
            }
        }

        auto qa = var<float[N]>();
        auto qb = var<float[N]>();
        auto qc = var<float[N]>();
        qa = v0;
        qb = v1;
        qc = v2;

        std::ignore = qa.permute(perm0);
        std::ignore = qb.permute(perm1);
        std::ignore = qc.permute(perm2);

        out = qa;
        std::ignore = out.blend(qb, mask_b);
        std::ignore = out.blend(qc, mask_c);
    };

    extract_ch(0, ch0);
    extract_ch(1, ch1);
    extract_ch(2, ch2);
}

// BT.601 limited-range RGB->YUV constants:
//   Y =  0.257R + 0.504G + 0.098B + 16
//   U = -0.148R - 0.291G + 0.439B + 128
//   V =  0.439R - 0.368G - 0.071B + 128
// Stored in _consts in order:
//   [0] 0.257f  [1] 0.504f  [2] 0.098f  [3] 16.f
//   [4] -0.148f [5] 0.291f  [6] 0.439f  [7] 128.f
//   [8] 0.368f  [9] 0.071f  [10] 255.f  (clip ceil)

template <size_t N>
void jit_to_nv12_converter::rgb_to_yuv(const variable<float[N]>& r,
                                       const variable<float[N]>& g,
                                       const variable<float[N]>& b,
                                       const variable<float[N]>& y_out,
                                       const variable<float[N]>& u_out,
                                       const variable<float[N]>& v_out,
                                       bool do_round) {
    auto tmp = var<float[N]>();
    auto zero = var<float[N]>();
    auto ceil_val = var<float[N]>();
    auto half_val = var<float[N]>();

    uni_vxorps(zero, zero, zero);
    uni_vbroadcastss(ceil_val, ptr[_consts + 10 * sizeof(float)]);
    if (do_round) {
        uni_vbroadcastss(half_val, ptr[_consts + 12 * sizeof(float)]);
    }

    auto clip = [&](const variable<float[N]>& x) {
        if (do_round) {
            uni_vaddps(x, x, half_val);
            uni_vroundps(x, x, 1);  // floor(x + 0.5f) to match std::round
        }
        uni_vmaxps(x, x, zero);
        uni_vminps(x, x, ceil_val);
    };

    uni_vbroadcastss(y_out, ptr[_consts + 0 * sizeof(float)]);  // y = 0.257
    uni_vmulps(y_out, y_out, r);                                // y = 0.257R
    uni_vbroadcastss(tmp, ptr[_consts + 1 * sizeof(float)]);    // tmp = 0.504
    uni_vmulps(tmp, tmp, g);                                    // tmp = 0.504G
    uni_vaddps(y_out, y_out, tmp);                              // y += 0.504G
    uni_vbroadcastss(tmp, ptr[_consts + 2 * sizeof(float)]);    // tmp = 0.098
    uni_vmulps(tmp, tmp, b);                                    // tmp = 0.098B
    uni_vaddps(y_out, y_out, tmp);                              // y += 0.098B
    uni_vbroadcastss(tmp, ptr[_consts + 3 * sizeof(float)]);    // tmp = 16
    uni_vaddps(y_out, y_out, tmp);                              // y += 16
    clip(y_out);

    uni_vbroadcastss(u_out, ptr[_consts + 4 * sizeof(float)]);  // u = -0.148
    uni_vmulps(u_out, u_out, r);                                // u = -0.148R
    uni_vbroadcastss(tmp, ptr[_consts + 5 * sizeof(float)]);    // tmp = 0.291
    uni_vmulps(tmp, tmp, g);                                    // tmp = 0.291G
    uni_vsubps(u_out, u_out, tmp);                              // u = -0.148R - 0.291G
    uni_vbroadcastss(tmp, ptr[_consts + 6 * sizeof(float)]);    // tmp = 0.439
    uni_vmulps(tmp, tmp, b);                                    // tmp = 0.439B
    uni_vaddps(u_out, u_out, tmp);                              // u = -0.148R - 0.291G + 0.439B
    uni_vbroadcastss(tmp, ptr[_consts + 7 * sizeof(float)]);    // tmp = 128
    uni_vaddps(u_out, u_out, tmp);                              // u += 128
    clip(u_out);

    uni_vbroadcastss(v_out, ptr[_consts + 6 * sizeof(float)]);  // v = 0.439
    uni_vmulps(v_out, v_out, r);                                // v = 0.439R
    uni_vbroadcastss(tmp, ptr[_consts + 8 * sizeof(float)]);    // tmp = 0.368
    uni_vmulps(tmp, tmp, g);                                    // tmp = 0.368G
    uni_vsubps(v_out, v_out, tmp);                              // v -= 0.368G
    uni_vbroadcastss(tmp, ptr[_consts + 9 * sizeof(float)]);    // tmp = 0.071
    uni_vmulps(tmp, tmp, b);                                    // tmp = 0.071B
    uni_vsubps(v_out, v_out, tmp);                              // v -= 0.071B
    uni_vbroadcastss(tmp, ptr[_consts + 7 * sizeof(float)]);    // tmp = 128
    uni_vaddps(v_out, v_out, tmp);                              // v += 128
    clip(v_out);
}

template <typename T>
class JitConverter;

template <typename T, size_t N>
class JitConverter<T[N]> : public jit_to_nv12_converter {
private:
    void generate() override;

    void load_rgb(const variable<const T*>& src,
                  const variable<float[N]>& r,
                  const variable<float[N]>& g,
                  const variable<float[N]>& b,
                  const variable<uint8_t>& color_fmt);

    void interleave_uv(const variable<float[N]>& u_sum,
                       const variable<float[N]>& v_sum,
                       const variable<float[N]>& uv_out,
                       bool is_integral);
};

template <typename T, size_t N>
void JitConverter<T[N]>::load_rgb(const variable<const T*>& src,
                                  const variable<float[N]>& r,
                                  const variable<float[N]>& g,
                                  const variable<float[N]>& b,
                                  const variable<uint8_t>& color_fmt) {
    // Load 3 packed float registers covering N pixels of interleaved data.
    auto v0 = var<float[N]>();
    auto v1 = var<float[N]>();
    auto v2 = var<float[N]>();

    const size_t step = N * sizeof(T);
    load(v0, src);
    src += step;
    load(v1, src);
    src += step;
    load(v2, src);
    src += step;

    // Deinterleave into channel registers.
    // ch0 = first channel (R for RGB, B for BGR)
    // ch2 = last  channel (B for RGB, R for BGR)
    auto ch0 = var<float[N]>();
    auto ch1 = var<float[N]>();
    auto ch2 = var<float[N]>();

    deinterleave(v0, v1, v2, ch0, ch1, ch2);

    // Assign R/G/B based on color format.
    _if(color_fmt == 0)
        ._then([&] {
            r = ch0;  // ch0 = R
            g = ch1;
            b = ch2;
        })
        ._else([&] {
            b = ch0;  // ch0 = B (BGR input)
            g = ch1;
            r = ch2;
        });
}

template <typename T, size_t N>
void JitConverter<T[N]>::interleave_uv(const variable<float[N]>& u_sum,
                                       const variable<float[N]>& v_sum,
                                       const variable<float[N]>& uv_out,
                                       bool is_integral) {
    auto gen_even_perm = []() {
        std::array<uint8_t, N> mask{};
        for (size_t i = 0; i < N; ++i) {
            mask[i] = static_cast<uint8_t>((i / 2) * 2);
        }
        return mask;
    };
    auto gen_odd_perm = []() {
        std::array<uint8_t, N> mask{};
        for (size_t i = 0; i < N; ++i) {
            mask[i] = static_cast<uint8_t>(((i / 2) * 2) + 1);
        }
        return mask;
    };

    const auto blend_mask = static_cast<uint16_t>(0xAAAAAAAAU & ((1U << N) - 1));

    auto even_sum = var<float[N]>();
    {
        auto u_even = var<float[N]>();
        auto v_even = var<float[N]>();
        u_even = u_sum;
        v_even = v_sum;
        std::ignore = u_even.permute(gen_even_perm());
        std::ignore = v_even.permute(gen_even_perm());
        even_sum = u_even;
        std::ignore = even_sum.blend(v_even, blend_mask);
    }

    auto odd_sum = var<float[N]>();
    {
        auto u_odd = var<float[N]>();
        auto v_odd = var<float[N]>();
        u_odd = u_sum;
        v_odd = v_sum;
        std::ignore = u_odd.permute(gen_odd_perm());
        std::ignore = v_odd.permute(gen_odd_perm());
        odd_sum = u_odd;
        std::ignore = odd_sum.blend(v_odd, blend_mask);
    }

    uni_vaddps(even_sum, even_sum, odd_sum);
    uv_out = even_sum;

    auto quarter = var<float[N]>();
    uni_vbroadcastss(quarter, ptr[_consts + 11 * sizeof(float)]);  // 0.25f
    uni_vmulps(uv_out, uv_out, quarter);

    if (is_integral) {
        auto zero = var<float[N]>();
        auto ceil_val = var<float[N]>();
        auto half_val = var<float[N]>();
        uni_vxorps(zero, zero, zero);
        uni_vbroadcastss(ceil_val, ptr[_consts + 10 * sizeof(float)]);
        uni_vbroadcastss(half_val, ptr[_consts + 12 * sizeof(float)]);
        uni_vaddps(uv_out, uv_out, half_val);
        uni_vroundps(uv_out, uv_out, 1);  // floor(x + 0.5f) to match std::round
        uni_vmaxps(uv_out, uv_out, zero);
        uni_vminps(uv_out, uv_out, ceil_val);
    }
}

template <typename T, size_t N>
void JitConverter<T[N]>::generate() {
    preamble();

    auto src0 = arg<const T*>(&Params::src0);
    auto src1 = arg<const T*>(&Params::src1);
    auto dst_y0 = arg<T*>(&Params::dst_y0);
    auto dst_y1 = arg<T*>(&Params::dst_y1);
    auto dst_uv = arg<T*>(&Params::dst_uv);
    auto width = arg(&Params::width);
    auto color_fmt = arg(&Params::colorFmt);

    static const float data[13] =
        {0.257F, 0.504F, 0.098F, 16.F, -0.148F, 0.291F, 0.439F, 128.F, 0.368F, 0.071F, 255.F, 0.25F, 0.5F};
    _consts = data;

    const auto reg_capacity_log = static_cast<size_t>(std::logb(N));
    const size_t y_step = N * sizeof(T);
    const size_t uv_step = N * sizeof(T);

    width >>= reg_capacity_log;

    foreach (0, width, [&]([[maybe_unused]] const variable<size_t>& /*idx*/) {
        auto u_sum = var<float[N]>();
        auto v_sum = var<float[N]>();

        // Row 0
        {
            auto r0 = var<float[N]>();
            auto g0 = var<float[N]>();
            auto b0 = var<float[N]>();
            load_rgb(src0, r0, g0, b0, color_fmt);

            auto y0 = var<float[N]>();
            auto v0 = var<float[N]>();
            rgb_to_yuv(r0, g0, b0, y0, u_sum, v0, std::is_integral_v<T>);

            store(dst_y0, y0);
            dst_y0 += y_step;

            v_sum = v0;
        }

        // Row 1
        {
            auto r1 = var<float[N]>();
            auto g1 = var<float[N]>();
            auto b1 = var<float[N]>();
            load_rgb(src1, r1, g1, b1, color_fmt);

            auto y1 = var<float[N]>();
            auto u1 = var<float[N]>();
            auto v1 = var<float[N]>();
            rgb_to_yuv(r1, g1, b1, y1, u1, v1, std::is_integral_v<T>);

            store(dst_y1, y1);
            dst_y1 += y_step;

            uni_vaddps(u_sum, u_sum, u1);
            uni_vaddps(v_sum, v_sum, v1);
        }

        auto uv_out = var<float[N]>();
        interleave_uv(u_sum, v_sum, uv_out, std::is_integral_v<T>);

        store(dst_uv, uv_out);
        dst_uv += uv_step;
    })
        ;

    // Tail: remaining pixels (width % N). NV12 requires even width, so tail is even.
    mov(width, argPtr(&Params::width));
    width &= N - 1;

    _if(width != 0)._then([&] {
        auto u_sumt = var<float[N]>();
        auto v_sumt = var<float[N]>();

        auto len = var<size_t>();
        mov(len, static_cast<const Xbyak::Reg64&>(width));
        const auto& len_reg = static_cast<const Xbyak::Reg64&>(len);
        lea(len_reg, ptr[len_reg + len_reg * 2]);  // len = width * 3

        // Row 0 tail
        {
            auto s0 = stack(3 * N * sizeof(T));
            s0.clear();
            copy<T>(s0.pointer(), static_cast<const Xbyak::Reg64&>(src0), len);

            auto buf0 = var<const T*>();
            buf0 = s0.pointer();

            auto r0t = var<float[N]>();
            auto g0t = var<float[N]>();
            auto b0t = var<float[N]>();
            load_rgb(buf0, r0t, g0t, b0t, color_fmt);

            auto y0t = var<float[N]>();
            auto v0 = var<float[N]>();
            rgb_to_yuv(r0t, g0t, b0t, y0t, u_sumt, v0, std::is_integral_v<T>);

            store(dst_y0, y0t, width);

            v_sumt = v0;
        }

        // Row 1 tail
        {
            auto s1 = stack(3 * N * sizeof(T));
            s1.clear();
            copy<T>(s1.pointer(), static_cast<const Xbyak::Reg64&>(src1), len);

            auto buf1 = var<const T*>();
            buf1 = s1.pointer();

            auto r1t = var<float[N]>();
            auto g1t = var<float[N]>();
            auto b1t = var<float[N]>();
            load_rgb(buf1, r1t, g1t, b1t, color_fmt);

            auto y1t = var<float[N]>();
            auto u1t = var<float[N]>();
            auto fv1t = var<float[N]>();
            rgb_to_yuv(r1t, g1t, b1t, y1t, u1t, fv1t, std::is_integral_v<T>);

            store(dst_y1, y1t, width);

            uni_vaddps(u_sumt, u_sumt, u1t);
            uni_vaddps(v_sumt, v_sumt, fv1t);
        }

        auto uv_outt = var<float[N]>();
        interleave_uv(u_sumt, v_sumt, uv_outt, std::is_integral_v<T>);

        store(dst_uv, uv_outt, width);
    });

    postamble();
}

template <typename T>
const jit_to_nv12_converter& jit_to_nv12_create() {
    auto createKernel = []() {
        std::unique_ptr<jit_to_nv12_converter> kernel;
        if (mayiuse(cpu_isa_t::avx512_core)) {
            auto c = new JitConverter<T[16]>;
            kernel.reset(c);
            c->init();
        } else if (mayiuse(cpu_isa_t::avx2)) {
            auto c = new JitConverter<T[8]>;
            kernel.reset(c);
            c->init();
        } else if (mayiuse(cpu_isa_t::sse41)) {
            auto c = new JitConverter<T[4]>;
            kernel.reset(c);
            c->init();
        } else {
            OPENVINO_THROW("Can't create jit RGB-to-NV12 converter kernel");
        }
        return kernel;
    };
    static auto kernel = createKernel();
    return *kernel;
}

template <typename T>
const jit_to_nv12_converter& jit_to_nv12_get() {
    return jit_to_nv12_create<T>();
}

template <typename T>
class SinglePlaneConvert<T, impl_desc_type::jit_uni> : public RefConverter {
public:
    explicit SinglePlaneConvert(Node* node) : RefConverter(node) {
        jit_to_nv12_create<T>();
    }

    void execute(const CpuParallelPtr& cpu_parallel, [[maybe_unused]] const dnnl::stream& strm) override {
        const auto& kernel = jit_to_nv12_get<T>();
        const auto& in_dims = inputDims(0);
        const size_t batch = in_dims[N_DIM];
        const size_t height = in_dims[H_DIM];
        const size_t width = in_dims[W_DIM];

        const T* src = static_cast<const T*>(input(0));
        T* dst = static_cast<T*>(output(0));
        T* dst_y = dst;
        T* dst_uv = dst + height * width;

        const size_t stride_in = height * width * 3;
        const size_t stride_y = height * width * 3 / 2;  // packed NV12 output stride
        const size_t stride_uv = height * width * 3 / 2;

        cpu_parallel->parallel_for2d(batch, height / 2, [&](int b, int half_h) {
            const size_t h0 = static_cast<size_t>(half_h) * 2;
            const size_t h1 = h0 + 1;

            typename jit_to_nv12_converter::Params args{
                src + static_cast<size_t>(b) * stride_in + h0 * width * 3,
                src + static_cast<size_t>(b) * stride_in + h1 * width * 3,
                dst_y + static_cast<size_t>(b) * stride_y + h0 * width,
                dst_y + static_cast<size_t>(b) * stride_y + h1 * width,
                dst_uv + static_cast<size_t>(b) * stride_uv + static_cast<size_t>(half_h) * width,
                width,
                _colorFormat[0]};
            kernel(args);
        });
    }
};

template <typename T>
class TwoPlaneConvert<T, impl_desc_type::jit_uni> : public RefConverter {
public:
    explicit TwoPlaneConvert(Node* node) : RefConverter(node) {
        jit_to_nv12_create<T>();
    }

    void execute(const CpuParallelPtr& cpu_parallel, [[maybe_unused]] const dnnl::stream& strm) override {
        const auto& kernel = jit_to_nv12_get<T>();
        const auto& in_dims = inputDims(0);
        const size_t batch = in_dims[N_DIM];
        const size_t height = in_dims[H_DIM];
        const size_t width = in_dims[W_DIM];

        const T* src = static_cast<const T*>(input(0));
        T* dst_y = static_cast<T*>(output(0));
        T* dst_uv = static_cast<T*>(output(1));

        const size_t stride_in = height * width * 3;
        const size_t stride_y = height * width;
        const size_t stride_uv = height * width / 2;

        cpu_parallel->parallel_for2d(batch, height / 2, [&](int b, int half_h) {
            const size_t h0 = static_cast<size_t>(half_h) * 2;
            const size_t h1 = h0 + 1;

            typename jit_to_nv12_converter::Params args{
                src + static_cast<size_t>(b) * stride_in + h0 * width * 3,
                src + static_cast<size_t>(b) * stride_in + h1 * width * 3,
                dst_y + static_cast<size_t>(b) * stride_y + h0 * width,
                dst_y + static_cast<size_t>(b) * stride_y + h1 * width,
                dst_uv + static_cast<size_t>(b) * stride_uv + static_cast<size_t>(half_h) * width,
                width,
                _colorFormat[0]};
            kernel(args);
        });
    }
};

#endif  // OPENVINO_ARCH_X86_64

}  // namespace to_nv12
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
    case Algorithm::ColorConvertRGBtoNV12:
    case Algorithm::ColorConvertBGRtoNV12: {
        for (const auto& desc : to_nv12::supportedPrimitiveDescs(this)) {
            const auto& inPortConfigs = std::get<0>(desc);
            const auto& outPortConfigs = std::get<1>(desc);
            const auto implType = std::get<2>(desc);
            addSupportedPrimDesc(inPortConfigs, outPortConfigs, implType);
        }
        initSupportedtoNV12Impls();
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

void ColorConvert::initSupportedtoNV12Impls() {
#define SUPPORTED_IMPL(Impl, type, desc_type)                            \
    [](Node* node) {                                                     \
        return new to_nv12::Impl<type, impl_desc_type::desc_type>(node); \
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

void ColorConvert::createPrimitive() {
    const NodeDesc* desc = getSelectedPrimitiveDescriptor();
    CPU_NODE_ASSERT(desc, "has no optimal primitive descriptor selected");

    if (!_impl) {
        const auto& cfg = desc->getConfig();
        const auto precision = cfg.inConfs[0].getMemDesc()->getPrecision();

        bool isSinglePlane = false;
        if (algorithm == Algorithm::ColorConvertRGBtoNV12 || algorithm == Algorithm::ColorConvertBGRtoNV12) {
            isSinglePlane = cfg.outConfs.size() == 1;
        } else {
            isSinglePlane = cfg.inConfs.size() == 1;
        }

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
