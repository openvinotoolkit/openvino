// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope.h"

#include <chrono>
#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "kernels/x64/rope_kernel.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

using namespace ov::intel_cpu::kernel;

namespace ov::intel_cpu::node {

RoPE::RoPE(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto node = ov::as_type_ptr<const op::internal::RoPE>(op);
    m_config = node->get_config();
}

static std::shared_ptr<kernel::JitKernelBase> createJitKernel(const jit_rotary_compile_params& param,
                                                              bool check_vec_size2 = false) {
    std::shared_ptr<kernel::JitKernelBase> res;

    MAYBE_UNUSED(param);
    MAYBE_UNUSED(check_vec_size2);

#if defined(OPENVINO_ARCH_X86_64)

    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        bool flag = true;
        if (check_vec_size2) {
            auto vec_size = jit_rotary_kernel<dnnl::impl::cpu::x64::avx512_core>::vec_size;
            if (param.rotary_ndims % (vec_size * 2) != 0) {
                flag = false;
            }
        }
        if (flag) {
            res = std::make_shared<jit_rotary_kernel<dnnl::impl::cpu::x64::avx512_core>>(param);
        }
    } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        bool flag = true;
        if (check_vec_size2) {
            auto vec_size = jit_rotary_kernel<dnnl::impl::cpu::x64::avx2>::vec_size;
            if (param.rotary_ndims % (vec_size * 2) != 0) {
                flag = false;
            }
        }
        if (flag) {
            res = std::make_shared<jit_rotary_kernel<dnnl::impl::cpu::x64::avx2>>(param);
        }
    }

    if (res) {
        res->create_kernel();
    }

#endif  // OPENVINO_ARCH_X86_64

    return res;
}

static void execJitKernel(const std::shared_ptr<kernel::JitKernelBase>& ker,
                          const void* src,
                          void* dst,
                          const float* cos,
                          const float* sin) {
    MAYBE_UNUSED(ker);
    MAYBE_UNUSED(src);
    MAYBE_UNUSED(dst);
    MAYBE_UNUSED(cos);
    MAYBE_UNUSED(sin);

#if defined(OPENVINO_ARCH_X86_64)

    jit_rotary_call_args call_args;
    call_args.src = src;
    call_args.cos = cos;
    call_args.sin = sin;
    call_args.dst = dst;
    (*ker)(&call_args);

#endif  // OPENVINO_ARCH_X86_64
}

template <typename T>
struct RoPE::RoPEExecutorRotateHalf : public RoPE::Executor {
    const op::internal::RoPE::Config& m_config;
    std::shared_ptr<kernel::JitKernelBase> m_rotaryKernel;

    RoPEExecutorRotateHalf(const op::internal::RoPE::Config& config) : m_config(config) {
        jit_rotary_compile_params jcp;
        jcp.src_prc = precision_of<T>::value;
        jcp.dst_prc = precision_of<T>::value;
        jcp.rotary_ndims = config.rotary_ndims;
        jcp.interleave = false;
        m_rotaryKernel = createJitKernel(jcp);
    }

    void execute(const dnnl::stream& strm,
                 const std::vector<MemoryPtr>& inputs,
                 const std::vector<MemoryPtr>& outputs) override {
        ov::intel_cpu::PlainTensor t_src(inputs[0]);
        ov::intel_cpu::PlainTensor t_cos(inputs[1]);
        ov::intel_cpu::PlainTensor t_sin(inputs[2]);
        ov::intel_cpu::PlainTensor t_dst(outputs[0]);
        ov::intel_cpu::PlainTensor gather;
        auto rotary_dims = m_config.rotary_ndims;

        bool can_inplace = true;
        if (m_config.slice_stop - m_config.slice_start > 0) {
            t_src = t_src.slice(3, m_config.slice_start, m_config.slice_stop);
            can_inplace = false;
        }
        if (m_config.input_trans0213) {
            t_src = t_src.permute({0, 2, 1, 3});
            can_inplace = false;
        }
        if (m_config.gather_position_arg_id > 0) {
            gather.reset(inputs[m_config.gather_position_arg_id]);
        }

        if (t_cos.m_rank == 2) {
            t_cos = t_cos.reshape({1, 1, t_cos.size(0), t_cos.size(1)});
        }
        if (t_sin.m_rank == 2) {
            t_sin = t_sin.reshape({1, 1, t_sin.size(0), t_sin.size(1)});
        }

        auto batch_size = t_src.size(0);
        auto head_cnt = t_src.size(1);
        auto seq_len = t_src.size(2);
        auto feature_size = t_src.size(3);

        parallel_for3d(batch_size, head_cnt, seq_len, [&](size_t b, size_t h, size_t p) {
            auto cos_pos = p;
            if (gather) {
                if (gather.m_rank == 4) {
                    cos_pos = gather.at<int32_t>({b, h, p, 0}, true);
                } else {
                    cos_pos = gather.at<int32_t>({b, p}, true);
                }
            }
            auto* src = t_src.ptr<T>(b, h, p);
            auto* cos = &t_cos.at<float>({b, h, cos_pos, 0}, true);
            auto* sin = &t_sin.at<float>({b, h, cos_pos, 0}, true);
            auto* dst = t_dst.ptr<T>(b, h, p, 0);

            if (m_rotaryKernel) {
                execJitKernel(m_rotaryKernel, src, dst, cos, sin);
            } else {
                auto half_rotary_dims = rotary_dims / 2;
                size_t i = 0;
                for (; i < half_rotary_dims; i++) {
                    auto src0 = src[i];
                    auto src1 = src[i + half_rotary_dims];
                    dst[i] = cos[i] * src0 - sin[i] * src1;
                    dst[i + half_rotary_dims] = cos[i + half_rotary_dims] * src1 + sin[i + half_rotary_dims] * src0;
                }
            }
            if (!can_inplace) {
                memcpy(dst + rotary_dims, src + rotary_dims, (feature_size - rotary_dims) * sizeof(T));
            }
        });
    }
};

template <typename T>
struct RoPE::RoPEExecutorInterleaved : public RoPE::Executor {
    const op::internal::RoPE::Config& m_config;
    std::shared_ptr<kernel::JitKernelBase> m_rotaryKernel;

    RoPEExecutorInterleaved(const op::internal::RoPE::Config& config) : m_config(config) {
        jit_rotary_compile_params jcp;
        jcp.src_prc = precision_of<T>::value;
        jcp.dst_prc = precision_of<T>::value;
        jcp.rotary_ndims = config.rotary_ndims;
        jcp.interleave = true;
        jcp.mix_cos_sin = false;
        m_rotaryKernel = createJitKernel(jcp, true);
    }

    void execute(const dnnl::stream& strm,
                 const std::vector<MemoryPtr>& inputs,
                 const std::vector<MemoryPtr>& outputs) override {
        ov::intel_cpu::PlainTensor t_src(inputs[0]);
        ov::intel_cpu::PlainTensor t_sin_cos(inputs[1]);
        ov::intel_cpu::PlainTensor t_dst(outputs[0]);

        auto batch_size = t_src.size(0);
        auto seq_len = t_src.size(1);
        auto head_cnt = t_src.size(2);
        auto head_dims = t_src.size(3);

        auto rotary_dims = m_config.rotary_ndims;
        auto half_rotary_dims = rotary_dims / 2;

        parallel_for3d(batch_size, seq_len, head_cnt, [&](size_t b, size_t p, size_t h) {
            auto* x = t_src.ptr<T>(b, p, h);
            float* sin = &t_sin_cos.at<float>({b, p, 0}, true);
            float* cos = &t_sin_cos.at<float>({b, p, half_rotary_dims}, true);
            auto* dst = m_config.output_trans0213 ? t_dst.ptr<T>(b, h, p) : t_dst.ptr<T>(b, p, h);

            if (m_rotaryKernel) {
                execJitKernel(m_rotaryKernel, x, dst, cos, sin);
            } else {
                size_t i = 0;
                for (size_t j = 0; i < rotary_dims; i += 2, j++) {
                    dst[i] = cos[j] * x[i] - sin[j] * x[i + 1];
                    dst[i + 1] = cos[j] * x[i + 1] + sin[j] * x[i];
                }
            }
            memcpy(dst + rotary_dims, x + rotary_dims, (head_dims - rotary_dims) * sizeof(T));
        });
    }
};

template <typename T>
struct RoPE::RoPEExecutorChatGLM : public RoPE::Executor {
    const op::internal::RoPE::Config& m_config;
    std::shared_ptr<kernel::JitKernelBase> m_rotaryKernel;

    RoPEExecutorChatGLM(const op::internal::RoPE::Config& config) : m_config(config) {
        jit_rotary_compile_params jcp;
        jcp.src_prc = precision_of<T>::value;
        jcp.dst_prc = precision_of<T>::value;
        jcp.rotary_ndims = config.rotary_ndims;
        jcp.interleave = true;
        jcp.mix_cos_sin = true;
        m_rotaryKernel = createJitKernel(jcp, true);
    }

    void execute(const dnnl::stream& strm,
                 const std::vector<MemoryPtr>& inputs,
                 const std::vector<MemoryPtr>& outputs) override {
        ov::intel_cpu::PlainTensor t_src(inputs[0]);
        ov::intel_cpu::PlainTensor t_cos_sin(inputs[1]);
        ov::intel_cpu::PlainTensor t_dst(outputs[0]);

        // [seq_len, batch_size, (hidden_states_q + hidden_states_k + hidden_states_v)]
        if (m_config.slice_stop - m_config.slice_start > 0) {
            t_src = t_src.slice(2, m_config.slice_start, m_config.slice_stop);
        }
        if (m_config.support_2d_rope) {
            // src [batch, length, H x S]
            auto seq_len = t_src.size(1);
            auto batch_size = t_src.size(0);

            auto head_cnt = m_config.head_cnt;
            auto head_size = m_config.head_size;

            auto rotary_dims = m_config.rotary_ndims;

            parallel_for3d(batch_size, head_cnt, seq_len, [&](size_t b, size_t h, size_t p) {
                // src [batch, length, H x S]
                auto* src = t_src.ptr<T>(b, p, h * head_size);
                // [batch_size, length, ndims//2, 2]
                auto* cos_sin = &t_cos_sin.at<float>({b, p, 0, 0}, true);
                auto* dst = t_dst.ptr<T>(b, h, p, 0);

                if (m_rotaryKernel) {
                    execJitKernel(m_rotaryKernel, src, dst, cos_sin, nullptr);
                } else {
                    size_t i = 0;
                    for (; i < rotary_dims; i += 2) {
                        auto cosv = cos_sin[i];
                        auto sinv = cos_sin[i + 1];
                        dst[i] = cosv * src[i] - sinv * src[i + 1];
                        dst[i + 1] = sinv * src[i] + cosv * src[i + 1];
                    }
                }

                memcpy(dst + rotary_dims, src + rotary_dims, (head_size - rotary_dims) * sizeof(T));
            });
        } else {
            auto seq_len = t_src.size(0);
            auto batch_size = t_src.size(1);

            auto head_cnt = m_config.head_cnt;
            auto head_size = m_config.head_size;

            auto rotary_dims = m_config.rotary_ndims;

            parallel_for3d(seq_len, batch_size, head_cnt, [&](size_t p, size_t b, size_t h) {
                auto* src = t_src.ptr<T>(p, b, h * head_size);
                // [length, batch_size, ndims//2, 2]
                auto* cos_sin = &t_cos_sin.at<float>({p, b, 0, 0}, true);
                auto* dst = t_dst.ptr<T>(p, b, h, 0);

                if (m_rotaryKernel) {
                    execJitKernel(m_rotaryKernel, src, dst, cos_sin, nullptr);
                } else {
                    size_t i = 0;
                    for (; i < rotary_dims; i += 2) {
                        auto cosv = cos_sin[i];
                        auto sinv = cos_sin[i + 1];
                        dst[i] = cosv * src[i] - sinv * src[i + 1];
                        dst[i + 1] = sinv * src[i] + cosv * src[i + 1];
                    }
                }

                memcpy(dst + rotary_dims, src + rotary_dims, (head_size - rotary_dims) * sizeof(T));
            });
        }
    }
};

template <typename T>
struct RoPE::RoPEExecutorQwen : public RoPE::Executor {
    const op::internal::RoPE::Config& m_config;
    std::shared_ptr<kernel::JitKernelBase> m_rotaryKernel;

    RoPEExecutorQwen(const op::internal::RoPE::Config& config) : m_config(config) {
        jit_rotary_compile_params jcp;
        jcp.src_prc = precision_of<T>::value;
        jcp.dst_prc = precision_of<T>::value;
        jcp.rotary_ndims = config.rotary_ndims;
        jcp.interleave = false;
        m_rotaryKernel = createJitKernel(jcp);
    }

    void execute(const dnnl::stream& strm,
                 const std::vector<MemoryPtr>& inputs,
                 const std::vector<MemoryPtr>& outputs) override {
        ov::intel_cpu::PlainTensor t_src(inputs[0]);   // [batch, length, head_cnt*head_size * 3]
        ov::intel_cpu::PlainTensor t_cos(inputs[1]);   // [1, present-kv-length, 1, rotary_dims]
        ov::intel_cpu::PlainTensor t_sin(inputs[2]);   // [1, present-kv-length, 1, rotary_dims]
        ov::intel_cpu::PlainTensor t_dst(outputs[0]);  // [batch, length, head_cnt, head_size]>
        ov::intel_cpu::PlainTensor gather;

        auto rotary_dims = t_cos.size(3);

        if (m_config.slice_stop - m_config.slice_start > 0) {
            t_src = t_src.slice(2, m_config.slice_start, m_config.slice_stop);
        }
        if (m_config.gather_position_arg_id > 0) {
            gather.reset(inputs[m_config.gather_position_arg_id]);
        }

        auto batch_size = t_src.size(0);
        auto seq_len = t_src.size(1);
        auto head_cnt = m_config.head_cnt;
        auto head_size = m_config.head_size;
        auto present_kv_len = t_cos.size(1);

        parallel_for3d(batch_size, seq_len, head_cnt, [&](size_t b, size_t p, size_t h) {
            size_t sincos_pos;
            if (gather) {
                if (gather.m_rank == 4) {
                    sincos_pos = gather.at<int32_t>({b, h, p, 0}, true);
                } else {
                    sincos_pos = gather.at<int32_t>({b, p}, true);
                }
            } else {
                sincos_pos = present_kv_len - seq_len + p;
            }

            auto* src = t_src.ptr<T>(b, p, h * head_size);
            auto* cos = &t_cos.at<float>({b, sincos_pos, h, 0}, true);
            auto* sin = &t_sin.at<float>({b, sincos_pos, h, 0}, true);
            auto* dst = t_dst.ptr<T>(b, p, h);

            if (m_rotaryKernel) {
                execJitKernel(m_rotaryKernel, src, dst, cos, sin);
            } else {
                auto half_rotary_dims = rotary_dims / 2;
                size_t i = 0;
                for (; i < half_rotary_dims; i++) {
                    auto s0 = src[i];
                    auto s1 = src[i + half_rotary_dims];
                    dst[i] = cos[i] * s0 - sin[i] * s1;
                    dst[i + half_rotary_dims] = cos[i + half_rotary_dims] * s1 + sin[i + half_rotary_dims] * s0;
                }
            }

            memcpy(dst + rotary_dims, src + rotary_dims, (head_size - rotary_dims) * sizeof(T));
        });
    }
};

void RoPE::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    auto srcPrecision = getOriginalInputPrecisionAtPort(0);

    auto rtPrecision = srcPrecision;
    auto CosSinPrecision = ov::element::f32;
    bool can_inplace = false;

    if (m_config.is_qwen) {
        if (rtPrecision == ov::element::f16) {
            m_executor = std::make_shared<RoPEExecutorQwen<ov::float16>>(m_config);
        } else if (rtPrecision == ov::element::bf16) {
            m_executor = std::make_shared<RoPEExecutorQwen<ov::bfloat16>>(m_config);
        } else {
            m_executor = std::make_shared<RoPEExecutorQwen<float>>(m_config);
            rtPrecision = ov::element::f32;
        }
    } else if (m_config.is_chatglm) {
        if (rtPrecision == ov::element::f16) {
            m_executor = std::make_shared<RoPEExecutorChatGLM<ov::float16>>(m_config);
        } else if (rtPrecision == ov::element::bf16) {
            m_executor = std::make_shared<RoPEExecutorChatGLM<ov::bfloat16>>(m_config);
        } else {
            m_executor = std::make_shared<RoPEExecutorChatGLM<float>>(m_config);
            rtPrecision = ov::element::f32;
        }
    } else if (m_config.is_interleaved) {
        CPU_NODE_ASSERT(m_config.slice_start == 0, "slice_start must be 0 for interleaved mode");
        CPU_NODE_ASSERT(m_config.slice_stop == 0, "slice_stop must be 0 for interleaved mode");
        CPU_NODE_ASSERT(m_config.gather_position_arg_id == 0, "gather_position_arg_id must be 0 for interleaved mode");
        if (rtPrecision == ov::element::f16) {
            m_executor = std::make_shared<RoPEExecutorInterleaved<ov::float16>>(m_config);
        } else if (rtPrecision == ov::element::bf16) {
            m_executor = std::make_shared<RoPEExecutorInterleaved<ov::bfloat16>>(m_config);
        } else {
            m_executor = std::make_shared<RoPEExecutorInterleaved<float>>(m_config);
            rtPrecision = ov::element::f32;
        }
    } else {
        can_inplace = true;
        if (rtPrecision == ov::element::f16) {
            m_executor = std::make_shared<RoPEExecutorRotateHalf<ov::float16>>(m_config);
        } else if (rtPrecision == ov::element::bf16) {
            m_executor = std::make_shared<RoPEExecutorRotateHalf<ov::bfloat16>>(m_config);
        } else {
            m_executor = std::make_shared<RoPEExecutorRotateHalf<float>>(m_config);
            rtPrecision = ov::element::f32;
        }
        if (m_config.slice_stop - m_config.slice_start > 0 || m_config.input_trans0213) {
            can_inplace = false;
        }
    }

    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, CosSinPrecision, getInputShapeAtPort(1), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, CosSinPrecision, getInputShapeAtPort(2), false, -1);
    if (m_config.gather_position_arg_id > 0) {
        inPortConfigs.emplace_back(LayoutType::ncsp,
                                   ov::element::i32,
                                   getInputShapeAtPort(m_config.gather_position_arg_id),
                                   false,
                                   -1);
    }

    // initialize output port
    std::vector<PortConfigurator> outPortConfigs;
    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, can_inplace ? 0 : -1);

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void RoPE::execute(const dnnl::stream& strm) {
    std::vector<MemoryPtr> inputs(getParentEdges().size()), outputs(getChildEdges().size());
    for (size_t i = 0; i < inputs.size(); i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        outputs[i] = getDstMemoryAtPort(i);
    }
    m_executor->execute(strm, inputs, outputs);
}

bool RoPE::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto node = ov::as_type_ptr<const op::internal::RoPE>(op);
        if (!node) {
            errorMessage = "Only RoPE operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
