// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_norm.h"

#include "common/arbitrary_order_desc_creator.h"
#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "openvino/util/common_util.hpp"
#include "shape_inference/custom/rms_norm.hpp"
#include "openvino/op/rms_norm.hpp"
#include "openvino/opsets/opset6.hpp"
#include "kernels/x64/rms_kernel.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace ov::intel_cpu::kernel;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
namespace node {

struct RMSNormKey {
    ov::element::Type precision;
    size_t data_size;
    size_t scale_size;
    size_t eps;
    size_t hash() const;
    bool operator==(const RMSNormKey& rhs) const;
};

size_t RMSNormKey::hash() const {
    size_t seed = 0;
    seed = hash_combine(seed, precision.hash());
    seed = hash_combine(seed, data_size);
    seed = hash_combine(seed, scale_size);
    seed = hash_combine(seed, eps);

    return seed;
}

bool RMSNormKey::operator==(const RMSNormKey& rhs) const {
    auto retVal = precision == rhs.precision &&
                  data_size == rhs.data_size &&
                  scale_size == rhs.scale_size &&
                  eps == rhs.eps;

    return retVal;
}

static std::shared_ptr<kernel::JitKernelBase> createJitKernel(const jit_rms_compile_params& param) {
    std::shared_ptr<kernel::JitKernelBase> res;

    MAYBE_UNUSED(param);

#if defined(OPENVINO_ARCH_X86_64)

    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        res = std::make_shared<jit_rms_kernel<dnnl::impl::cpu::x64::avx512_core>>(param);
    } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        res = std::make_shared<jit_rms_kernel<dnnl::impl::cpu::x64::avx2>>(param);
    }

    if (res)
        res->create_kernel();

#endif // OPENVINO_ARCH_X86_64

    return res;
}

static void execJitKernel(const std::shared_ptr<kernel::JitKernelBase>& ker, const uint8_t* src, uint8_t* dst, const float* scale) {
    MAYBE_UNUSED(ker);
    MAYBE_UNUSED(src);
    MAYBE_UNUSED(dst);
    MAYBE_UNUSED(scale);

#if defined(OPENVINO_ARCH_X86_64)

    jit_rms_call_args call_args;
    call_args.src = src;
    call_args.dst = dst;
    call_args.scale = scale;
    (*ker)(&call_args);

#endif // OPENVINO_ARCH_X86_64
}

struct RMSNorm::RMSNormExecutor : public RMSNorm::Executor {
    RMSNormExecutor(ov::element::Type precision, size_t data_size, size_t scale_size, float eps, bool has_scale) : m_precision(precision) {
        jit_rms_compile_params jcp;
        jcp.src_prc = precision;
        jcp.dst_prc = precision;
        jcp.data_size = data_size;
        jcp.scale_size = scale_size;
        jcp.eps = eps;
        jcp.has_scale = has_scale;
        m_kernel = createJitKernel(jcp);
    }
    void execute(const std::vector<MemoryPtr>& inputs, const MemoryPtr output) override {
        auto src = inputs[0]->getDataAs<uint8_t>();
        auto dst = output->getDataAs<uint8_t>();
        float* scale = nullptr;
        if (inputs.size() > 2)
            scale = inputs[2]->getDataAs<float>();

        const auto& src_strides = inputs[0]->getDescWithType<BlockedMemoryDesc>()->getStrides();
        const auto& dst_strides = output->getDescWithType<BlockedMemoryDesc>()->getStrides();
        const auto& shape = inputs[0]->getStaticDims();
        const auto src_stride = src_strides[src_strides.size() - 2] * m_precision.size();
        const auto dst_stride = dst_strides[dst_strides.size() - 2] * m_precision.size();
        auto n = shape_size(shape) / shape[shape.size() - 1];
        parallel_for(n, [&] (size_t i) {
            execJitKernel(m_kernel, src + i * src_stride, dst + i * dst_stride, scale);
        });
    }

private:
    ov::element::Type m_precision;
    std::shared_ptr<kernel::JitKernelBase> m_kernel;
};

RMSNorm::RMSNorm(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, RMSNormShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }
    const auto rms = std::dynamic_pointer_cast<const ov::op::internal::RMSNorm>(op);
    m_eps = static_cast<float>(rms->get_epsilon());
    m_has_scale = op->get_input_size() > 2;
}

void RMSNorm::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto precision = getOriginalInputPrecisionAtPort(0);

    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_core)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    if (m_has_scale) {
        addSupportedPrimDesc({{LayoutType::ncsp, precision}, {LayoutType::ncsp, ov::element::i32}, {LayoutType::ncsp, ov::element::f32}},
                            {{LayoutType::ncsp, precision}},
                            impl_type);
    } else {
        addSupportedPrimDesc({{LayoutType::ncsp, precision}, {LayoutType::ncsp, ov::element::i32}},
                            {{LayoutType::ncsp, precision}},
                            impl_type);
    }
}

void RMSNorm::createPrimitive() {
    auto precision = getOriginalInputPrecisionAtPort(0);
    auto data_dims = getSrcMemoryAtPort(0)->getDescWithType<BlockedMemoryDesc>()->getBlockDims();
    auto has_scale = getOriginalInputsNumber() > 2;
    size_t data_size = data_dims[data_dims.size() - 1];
    size_t scale_size = 0;
    if (has_scale) {
        scale_size = getSrcMemoryAtPort(2)->getDescWithType<BlockedMemoryDesc>()->getBlockDims()[0];
    }

    RMSNormKey key = {precision, data_size, scale_size, static_cast<size_t>(dnnl::impl::float2int(m_eps))};

    auto builder = [&](const RMSNormKey& key) -> std::shared_ptr<RMSNormExecutor> {
#ifdef OPENVINO_ARCH_X86_64
        return std::make_shared<RMSNormExecutor>(precision, data_size, scale_size, m_eps, has_scale);
#else
        return nullptr;
#endif
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    if (!result.first) {
        OPENVINO_THROW("RMSNorm Executor creation fails with precision " + precision.to_string());
    }
    m_executor = result.first;
}

void RMSNorm::execute(dnnl::stream strm) {
    auto orginInputNumber = getOriginalInputsNumber();
    std::vector<MemoryPtr> inputs(orginInputNumber);

    for (size_t i = 0; i < orginInputNumber; i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }

    m_executor->execute(inputs, getDstMemoryAtPort(0));
}

bool RMSNorm::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto rms = std::dynamic_pointer_cast<const ov::op::internal::RMSNorm>(op);
        if (rms) {
            // check the last dimension of data
            auto data_pshape = op->input_value(0).get_partial_shape();
            if (data_pshape.rank().is_dynamic()) {
                errorMessage = "RMSNorm data rank is not static.";
                return false;
            }
            const auto& data_rank = op->get_input_partial_shape(0).rank().get_length();
            if (data_pshape[data_rank - 1].is_dynamic()) {
                errorMessage = "RMSNorm last dimension of data is not static.";
                return false;
            }
            if (data_rank == 1) {
                errorMessage = "RMSNorm data rank must be greater than 1.";
                return false;
            }
            // check axes
            auto axes_op = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
            if (!axes_op) {
                errorMessage = "RMSNorm axes is expected as Constant.";
                return false;
            }
            // axes should be 1d or scalar in spec
            auto axes_vals = axes_op->cast_vector<int>();
            if (axes_vals[0] != -1 && axes_vals[0] != data_rank - 1) {
                errorMessage = "RMSNorm axes must be the last dimension.";
                return false;
            }

            // check scale
            if (op->get_input_size() > 2) {
                if (op->get_input_partial_shape(2).rank().get_length() > 1) {
                    errorMessage = "RMSNorm scale must be 1D or scalar.";
                    return false;
                }
                if (op->get_input_partial_shape(2).is_dynamic()) {
                    errorMessage = "RMSNorm scale shape is not static.";
                    return false;
                }
            }
            if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
                errorMessage = "RMSNorm needs avx2+.";
                return false;
            }
        } else {
            errorMessage = "Only RMSNorm operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
