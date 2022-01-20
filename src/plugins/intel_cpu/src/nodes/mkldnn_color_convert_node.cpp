// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_color_convert_node.h"
#include <memory_desc/dnnl_blocked_memory_desc.h>
#include <openvino/op/nv12_to_bgr.hpp>
#include <openvino/op/nv12_to_rgb.hpp>
#include <openvino/core/type.hpp>
#include <ie/ie_parallel.hpp>
#include <utils/jit_kernel.hpp>

using namespace InferenceEngine;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::cpu::x64;
using namespace Xbyak;

namespace MKLDNNPlugin {

namespace {

std::tuple<Algorithm, std::string> getAlgorithmFor(const std::shared_ptr<const ngraph::Node>& op) {
    if (ov::is_type<ov::op::v8::NV12toRGB>(op))
        return std::make_tuple(Algorithm::ColorConvertNV12toRGB, std::string());
    if (ov::is_type<ov::op::v8::NV12toBGR>(op))
        return std::make_tuple(Algorithm::ColorConvertNV12toBGR, std::string());
    return std::make_tuple(Algorithm::Default, "Only v8::NV12toRGB or v8::NV12toBGR operation is supported");
}

namespace nv12 {

MKLDNNColorConvertNode::Converter::PrimitiveDescs supportedPrimitiveDescs(MKLDNNNode *node) {
    const LayoutType layout = LayoutType::ncsp; // 0,1,2,3

    const Precision precision = node->getOriginalInputPrecisionAtPort(0) == Precision::U8
                                    ? Precision::U8
                                    : Precision::FP32;

    MKLDNNColorConvertNode::Converter::PrimitiveDescs descs;

    descs.emplace_back(std::vector<PortConfigurator> { node->getOriginalInputsNumber(), { layout, precision } },
                        std::vector<PortConfigurator> { { layout, precision } },
                        mayiuse(cpu_isa_t::sse41)
                            ? impl_desc_type::jit_uni
                            : impl_desc_type::ref,
                        true);

    return std::move(descs);
}

class Converter : public MKLDNNColorConvertNode::Converter {
    using Base = MKLDNNColorConvertNode::Converter;

public:
    Converter(MKLDNNNode *node);

protected:
    Shapes shapeInfer() const override;
    bool singlePlane() const;

    template<typename T>
    void convert(const T* y,
                 const T* uv,
                 T* dst,
                 size_t batch_size,
                 size_t height,
                 size_t width,
                 size_t stride_y,
                 size_t stride_uv);
};

Converter::Converter(MKLDNNNode *node)
    : Base(node, node->getAlgorithm() == Algorithm::ColorConvertNV12toRGB
                        ? ColorFormat { { 0, 1, 2 } }
                        : ColorFormat { { 2, 1, 0 } }) {
    if (node->getOriginalInputsNumber() != (singlePlane() ? 1: 2))
        IE_THROW() <<"NV12Converter node has incorrect number of inputs";
    if (!node->getOriginalOutputsNumber())
        IE_THROW() <<"NV12Converter node has incorrect number of outputs";
}

MKLDNNColorConvertNode::Converter::Shapes
Converter::shapeInfer() const {
    const auto & dims = inputDims(0);
    if (dims.size() != 4)
        IE_THROW() <<"NV12Converter node has incorrect input dimensions";
    return singlePlane()
                ? Shapes { { dims[N_DIM], dims[H_DIM] * 2 / 3, dims[W_DIM], 3 } }
                : Shapes { { dims[N_DIM], dims[H_DIM], dims[W_DIM], 3 } };
}

bool Converter::singlePlane() const {
    return _node->getOriginalInputsNumber() == 1;
}

template<typename T>
void Converter::convert(const T* y,
                        const T* uv,
                        T* dst,
                        size_t batch_size,
                        size_t height,
                        size_t width,
                        size_t stride_y,
                        size_t stride_uv) {
    InferenceEngine::parallel_for2d(batch_size, height, [&](int batch, int h) {
        T* out = dst + batch * width * height * 3;
        auto y_ptr = y + batch * stride_y;
        auto uv_ptr = uv + batch * stride_uv;

        for (int w = 0; w < width; w++) {
            auto y_index = h * width + w;
            auto y_val = static_cast<float>(y_ptr[y_index]);
            auto uv_index = (h / 2) * width + (w / 2) * 2;
            auto u_val = static_cast<float>(uv_ptr[uv_index]);
            auto v_val = static_cast<float>(uv_ptr[uv_index + 1]);
            auto c = y_val - 16.f;
            auto d = u_val - 128.f;
            auto e = v_val - 128.f;
            auto clip = [](float a) -> T {
                if (std::is_integral<T>()) {
                    return static_cast<T>(std::min(std::max(std::round(a), 0.f), 255.f));
                } else {
                    return static_cast<T>(std::min(std::max(a, 0.f), 255.f));
                }
            };

            auto r = clip(1.164f * c + 1.596f * e);
            auto g = clip(1.164f * c - 0.391f * d - 0.813f * e);
            auto b = clip(1.164f * c + 2.018f * d);

            out[y_index * 3 + _colorFormat[0]] = r;
            out[y_index * 3 + _colorFormat[1]] = g;
            out[y_index * 3 + _colorFormat[2]] = b;
        }
    });
}

template<typename T, impl_desc_type I>
class SinglePlaneConvert;
template<typename T, impl_desc_type I>
class TwoPlaneConvert;

template<typename T>
class SinglePlaneConvert<T, impl_desc_type::ref> : public Converter {
public:
    using Converter::Converter;

    void execute(mkldnn::stream strm) override {
        const auto & dims = inputDims(0);

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM] * 2 / 3;
        const size_t width = dims[W_DIM];

        const T* y = static_cast<const T*>(input(0));
        const T* uv = y + width * height;
        T* dst = static_cast<T*>(output(0));

        convert<T>(y, uv, dst,
                   batch_size,
                   height,
                   width,
                   height * width * 3 / 2,
                   height * width * 3 / 2);
    }
};

template<typename T>
class TwoPlaneConvert<T, impl_desc_type::ref> : public Converter {
public:
    using Converter::Converter;

    void execute(mkldnn::stream strm) override {
        const auto & dims = inputDims(0);

        const T* y = static_cast<const T*>(input(0));
        const T* uv = static_cast<const T*>(input(1));
        T* dst = static_cast<T*>(output(0));

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM];
        const size_t width = dims[W_DIM];

        convert<T>(y, uv, dst,
                   batch_size,
                   height,
                   width,
                   height * width,
                   height * width / 2);
    }
};

struct jit_uni_converter : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_converter)

    struct Params {
        const void * y;
        const void * uv;
        void * dst;
        size_t width;
        uint8_t colorFormat;    // RGB: 0, BGR: !=0
    };

    typedef void (*function_t)(const Params *);

    void operator()(const Params & args) const {
        _fn(&args);
    }

    template<typename T>
    static const jit_uni_converter & get();

protected:
    jit_uni_converter() = default;

    function_t _fn;
};

template<typename T, cpu_isa_t isa>
class jit_uni_converter_impl : public jit_uni_converter {
    void generate() override;
};

template<typename T, cpu_isa_t isa>
void jit_uni_converter_impl<T, isa>::generate() {
    using reg_type = typename isa_traits<isa>::reg::type;
    using var_type = variable<float[isa_traits<isa>::reg::length]>;
    constexpr auto reg_capacity = isa_traits<isa>::reg::length;

    preamble();

    // Get arguments addresses
    auto y = arg<const T *>(&Params::y);
    auto uv = arg<const T *>(&Params::uv);
    auto dst = arg<T *>(&Params::dst);
    auto width = arg(&Params::width);
    auto colorFormat = arg(&Params::colorFormat);

    // Reserve registars
    auto consts = reserve<Reg64>();
    auto tmp = var<float[reg_capacity]>();
    auto y_val = var<float[reg_capacity]>();
    auto u_val = var<float[reg_capacity]>();
    auto v_val = var<float[reg_capacity]>();
    auto r = var<float[reg_capacity]>();
    auto g = var<float[reg_capacity]>();
    auto b = var<float[reg_capacity]>();

    // Aliases
    const auto & c = y_val;
    const auto & d = u_val;
    const auto & e = v_val;
    const auto & uv_val = tmp;

    const uint8_t even_mask = 0xA0;         // 0b10100000
    const uint8_t odd_mask = 0xF5;          // 0b11110101
    static const float data[8] = { 16.f, 128.f, 1.164f, 1.596f, 0.391f, 2.018f, 0.813f, 255.f };

    mov(consts, (size_t)data);

    auto clip = [this](const reg_type & op, const reg_type & a, const reg_type & b) {
        if (std::is_integral<T>())
            uni_vroundps(op, op, 0);
        uni_vmaxps(op, op, a);
        uni_vminps(op, op, b);
    };

    auto blend = [&, this](const reg_type & r, const reg_type & g, const reg_type & b) {
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
            c = r0,g0,b0,r1,g1,b1,r2,g2
            d = b2,r3,g3,b3,r4,g4,b4,r5
            e = g5,b5,r6,g6,b6,r7,g7,b7
        */

        auto genPermutationMask = [](int offset) -> std::array<int, isa_traits<isa>::reg::length> {
            std::array<int, isa_traits<isa>::reg::length> mask {};

            if (!!(isa & cpu_isa_t::avx)) {
                for (int i = 0; i < mask.size(); ++i)
                    mask[(i * 3 + offset) % mask.size()] = i;
            } else {
                int & m0 = mask.front();
                for (int i = 0; i < 4; ++i)
                    m0 |= i << ((i * 3 + offset) % 4) * 2;
            }

            return std::move(mask);
        };

        static const auto permutationMask4r = genPermutationMask(0);
        static const auto permutationMask4g = genPermutationMask(1);
        static const auto permutationMask4b = genPermutationMask(2);

        uni_vpermps(r, permutationMask4r.data(), r);
        uni_vpermps(g, permutationMask4g.data(), g);
        uni_vpermps(b, permutationMask4b.data(), b);

        auto blendWithMask = [&](int offset, const var_type & result) {
            static const uint32_t blendMasks[2] = {
                0x92492492,
                0x24924924
            };
            const uint16_t mask0 = static_cast<const uint16_t>(blendMasks[0] >> ((offset * reg_capacity) % 3));
            const uint16_t mask1 = static_cast<const uint16_t>(blendMasks[1] >> ((offset * reg_capacity) % 3));

            result = r;
            result.blend(g, mask0);
            result.blend(b, mask1);
        };

        blendWithMask(0, c);
        blendWithMask(1, d);
        blendWithMask(2, e);
    };  // blend

    auto colorConvert = [&](const reg_type & y_val, const reg_type & uv_val) {
        uni_vshufps(u_val, uv_val, uv_val, even_mask);              // u_val = tmp[0,0,2,2,4,4,6,6]
        uni_vshufps(v_val, uv_val, uv_val, odd_mask);               // v_val = tmp[1,1,3,3,5,5,7,7]

        uni_vbroadcastss(tmp, ptr[consts + 0 * sizeof(float)]);     // tmp = [16.0f,16.0f,...]
        uni_vsubps(c, y_val, tmp);                                  // c = y_val - tmp
        uni_vbroadcastss(tmp, ptr[consts + 1 * sizeof(float)]);     // tmp = [128.f,128.f,...]
        uni_vsubps(d, u_val, tmp);                                  // d = u_val - tmp
        uni_vsubps(e, v_val, tmp);                                  // e = v_val - tmp

        uni_vbroadcastss(tmp, ptr[consts + 2 * sizeof(float)]);     // tmp = [1.164f,1.164f,...]
        uni_vmulps(c, c, tmp);                                      // c = c * tmp

        uni_vbroadcastss(r, ptr[consts + 3 * sizeof(float)]);       // r = [1.596f,1.596f,...]
        uni_vmulps(r, r, e);                                        // r = r * e
        uni_vaddps(r, r, c);                                        // r = r + c

        uni_vbroadcastss(g, ptr[consts + 4 * sizeof(float)]);       // g = [0.391f,0.391f,...]
        uni_vmulps(g, g, d);                                        // g = g * d
        uni_vsubps(g, c, g);                                        // g = c - g
        uni_vbroadcastss(tmp, ptr[consts + 6 * sizeof(float)]);     // tmp = [0.813f,0.813f,...]
        uni_vmulps(tmp, tmp, e);                                    // tmp = tmp * e
        uni_vsubps(g, g, tmp);                                      // g = g - tmp

        uni_vbroadcastss(b, ptr[consts + 5 * sizeof(float)]);       // b = [2.018f,2.018f,...]
        uni_vmulps(b, b, d);                                        // b = b * d
        uni_vaddps(b, b, c);                                        // b = b + c

        // clip
        uni_vxorps(c, c, c);
        uni_vbroadcastss(d, ptr[consts + 7 * sizeof(float)]);

        clip(r, c, d);
        clip(g, c, d);
        clip(b, c, d);

        _if(colorFormat == 0)
        ._then([&]{ blend(r, g, b); })
        ._else([&]{ blend(b, g, r); });
    };

    const size_t reg_capacity_log = static_cast<size_t>(std::logb(reg_capacity));
    const size_t step = reg_capacity * sizeof(T);

    width >>= reg_capacity_log;

    foreach(0, width, [&](const Reg64 & idx) {
        load(y_val, y);
        load(uv_val, uv);

        colorConvert(y_val, uv_val);

        store(dst, c);  dst += step;
        store(dst, d);  dst += step;
        store(dst, e);  dst += step;

        y += step;
        uv += step;
    });

    mov(width, argPtr(&Params::width));
    width &= reg_capacity - 1;

    _if(width != 0)
    ._then([&] {
        auto s = stack(3 * step);
        s.clear();

        copy<T>(s.pointer(), y, width);
        copy<T>(ptr[s.pointer() + step], uv, width);

        y = s.pointer();
        lea(uv, ptr[s.pointer() + step]);

        load(y_val, y);
        load(uv_val, uv);

        colorConvert(y_val, uv_val);

        store(y, c);    y += step;
        store(y, d);    y += step;
        store(y, e);

        lea(width, ptr[width + width * 2]);
        copy<T>(ptr[dst], s.pointer(), width);
    });

    postamble();
}

template<typename T>
const jit_uni_converter & jit_uni_converter::get() {
    auto createKernel = []() {
        std::unique_ptr<jit_uni_converter> kernel;

        if (mayiuse(cpu_isa_t::avx512_common)) {
            kernel.reset(new jit_uni_converter_impl<T, cpu_isa_t::avx512_common>);
        } else if (mayiuse(cpu_isa_t::avx2)) {
            kernel.reset(new jit_uni_converter_impl<T, cpu_isa_t::avx2>);
        } else if (mayiuse(cpu_isa_t::sse41)) {
            kernel.reset(new jit_uni_converter_impl<T, cpu_isa_t::sse41>);
        } else {
            IE_THROW() << "Can't create jit color converter kernel";
        }

        if (kernel->create_kernel() != status::success)
            IE_THROW() << "Can't generate jit color converter kernel";
        kernel->_fn = (function_t)kernel->jit_ker();

        return std::move(kernel);
    };

    static auto kernel = createKernel();

    return *kernel;
}

template<typename T>
class SinglePlaneConvert<T, impl_desc_type::jit_uni> : public Converter {
public:
    using Converter::Converter;

    void execute(mkldnn::stream strm) override {
        const auto & kernel = jit_uni_converter::get<T>();
        const auto & dims = inputDims(0);

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM] * 2 / 3;
        const size_t width = dims[W_DIM];

        const T* y = static_cast<const T*>(input(0));
        const T* uv = y + width * height;
        T* dst = static_cast<T*>(output(0));

        const size_t stride_y = height * width * 3 / 2;
        const size_t stride_uv = height * width * 3 / 2;

        InferenceEngine::parallel_for2d(batch_size, height, [&](int batch, int h) {
            typename jit_uni_converter::Params args;
            args.y = y + batch * stride_y + h * width;
            args.uv = uv + batch * stride_uv + (h / 2) * width;
            args.dst = dst + (batch * width * height + h * width) * 3;
            args.width = width;
            args.colorFormat = _colorFormat[0]; // The first byte is enough to determine the RGB or BGR format.
            kernel(args);
        });
    }
};

template<typename T>
class TwoPlaneConvert<T, impl_desc_type::jit_uni> : public Converter {
public:
    using Converter::Converter;

    void execute(mkldnn::stream strm) override {
        const auto & kernel = jit_uni_converter::get<T>();
        const auto & dims = inputDims(0);

        const size_t batch_size = dims[N_DIM];
        const size_t height = dims[H_DIM];
        const size_t width = dims[W_DIM];

        const T* y = static_cast<const T*>(input(0));
        const T* uv = static_cast<const T*>(input(1));
        T* dst = static_cast<T*>(output(0));

        const size_t stride_y = height * width;
        const size_t stride_uv = height * width / 2;

        InferenceEngine::parallel_for2d(batch_size, height, [&](int batch, int h) {
            typename jit_uni_converter::Params args;
            args.y = y + batch * stride_y + h * width;
            args.uv = uv + batch * stride_uv + (h / 2) * width;
            args.dst = dst + (batch * width * height + h * width) * 3;
            args.width = width;
            args.colorFormat = _colorFormat[0]; // The first byte is enough to determine the RGB or BGR format.
            kernel(args);
        });
    }
};

}   // namespace nv12
}   // namespace

MKLDNNColorConvertNode::Converter::Converter(MKLDNNNode *node, const ColorFormat & colorFormat)
    : _node(node)
    , _colorFormat(colorFormat) {
}

InferenceEngine::Precision MKLDNNColorConvertNode::Converter::inputPrecision(size_t idx) const {
    return _node->getParentEdgeAt(idx)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getPrecision();
}

InferenceEngine::Precision MKLDNNColorConvertNode::Converter::outputPrecision(size_t idx) const {
    return _node->getChildEdgeAt(idx)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getPrecision();
}

const void * MKLDNNColorConvertNode::Converter::input(size_t idx) const {
    return _node->getParentEdgeAt(idx)->getMemoryPtr()->GetPtr();
}

void * MKLDNNColorConvertNode::Converter::output(size_t idx) const {
    return _node->getChildEdgeAt(idx)->getMemoryPtr()->GetPtr();
}

const VectorDims & MKLDNNColorConvertNode::Converter::inputDims(size_t idx) const {
    return _node->getParentEdgesAtPort(idx)[0]->getMemory().getStaticDims();
}

bool MKLDNNColorConvertNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    Algorithm alg;
    std::tie(alg, errorMessage) = getAlgorithmFor(op);
    return alg != Algorithm::Default;
}

MKLDNNColorConvertNode::MKLDNNColorConvertNode(const std::shared_ptr<ngraph::Node>& op,
                                               const mkldnn::engine& eng,
                                               MKLDNNWeightsSharing::Ptr &cache)
    : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    std::tie(algorithm, errorMessage) = getAlgorithmFor(op);
    if (algorithm == Algorithm::Default)
        IE_THROW(NotImplemented) << errorMessage;
}

void MKLDNNColorConvertNode::getSupportedDescriptors() {}

void MKLDNNColorConvertNode::initSupportedPrimitiveDescriptors() {
    if (supportedPrimitiveDescriptors.empty()) {
        switch (algorithm) {
            case Algorithm::ColorConvertNV12toRGB:
            case Algorithm::ColorConvertNV12toBGR: {
                for (const auto &desc : nv12::supportedPrimitiveDescs(this)) {
                    const auto & inPortConfigs = std::get<0>(desc);
                    const auto & outPortConfigs = std::get<1>(desc);
                    const auto implType = std::get<2>(desc);
                    const auto dynBatchSupport = std::get<3>(desc);
                    addSupportedPrimDesc(inPortConfigs, outPortConfigs, implType, dynBatchSupport);
                }
                initSupportedNV12Impls();
                break;
            default:
                break;
            }
        }
    }
}

void MKLDNNColorConvertNode::initSupportedNV12Impls() {
    #define SUPPORTED_IMPL(Impl, type, desc_type)                           \
        [](MKLDNNNode *node) {                                              \
            return new nv12::Impl<type, impl_desc_type::desc_type>(node);   \
        };

    // ref
    {
        auto &impls = _supportedImpls[impl_desc_type::ref][algorithm];
        impls[Precision::U8][true] = SUPPORTED_IMPL(SinglePlaneConvert, uint8_t, ref);
        impls[Precision::U8][false] = SUPPORTED_IMPL(TwoPlaneConvert, uint8_t, ref);
        impls[Precision::FP32][true] = SUPPORTED_IMPL(SinglePlaneConvert, float, ref);
        impls[Precision::FP32][false] = SUPPORTED_IMPL(TwoPlaneConvert, float, ref);
    }

    // jit_uni
    {
        auto &impls = _supportedImpls[impl_desc_type::jit_uni][algorithm];
        impls[Precision::U8][true] = SUPPORTED_IMPL(SinglePlaneConvert, uint8_t, jit_uni);
        impls[Precision::U8][false] = SUPPORTED_IMPL(TwoPlaneConvert, uint8_t, jit_uni);
        impls[Precision::FP32][true] = SUPPORTED_IMPL(SinglePlaneConvert, float, jit_uni);
        impls[Precision::FP32][false] = SUPPORTED_IMPL(TwoPlaneConvert, float, jit_uni);
    }

    #undef SUPPORTED_IMPL
}

void MKLDNNColorConvertNode::createPrimitive() {
    const NodeDesc *desc = getSelectedPrimitiveDescriptor();
    if (!desc)
        IE_THROW() << getTypeStr() + " node with name '" + getName() + "' "
                   << "no optimal primitive descriptor selected";

    if (!_impl) {
        const auto & cfg = desc->getConfig();
        const auto precision = cfg.inConfs[0].desc->getPrecision();
        const bool isSinglePlane = cfg.inConfs.size() == 1;

        _impl = std::unique_ptr<Converter>(_supportedImpls
                                            .at(desc->getImplementationType())
                                            .at(algorithm)
                                            .at(precision)
                                            .at(isSinglePlane)(this));
    }
}

void MKLDNNColorConvertNode::execute(mkldnn::stream strm) {
    if (!_impl)
        IE_THROW() << getTypeStr() + " node with name '" + getName() + "' "
                   << "has no any implemented converter";
    _impl->execute(strm);
}

bool MKLDNNColorConvertNode::created() const {
    return getType() == ColorConvert;
}

std::vector<VectorDims> MKLDNNColorConvertNode::shapeInfer() const {
    if (!_impl)
        IE_THROW() << getTypeStr() + " node with name '" + getName() + "' "
                   << "has no any implemented converter";
    return _impl->shapeInfer();
}

bool MKLDNNColorConvertNode::needPrepareParams() const {
    return false;
}

void MKLDNNColorConvertNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

REG_MKLDNN_PRIM_FOR(MKLDNNColorConvertNode, ColorConvert);

}  // namespace MKLDNNPlugin
