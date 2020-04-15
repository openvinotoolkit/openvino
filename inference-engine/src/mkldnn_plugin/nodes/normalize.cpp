// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "list.hpp"

#include "jit_generator.hpp"
#include <algorithm>
#include <cmath>
#include <ie_parallel.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "bf16transformer.h"

using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

#define GET_OFF(field) offsetof(jit_args_normalize, field)

struct jit_args_normalize {
    float *src;
    float *dst;
    float *weights;
    float *eps;
    float *sqr_sums;
    float *sqrt_sum;
    size_t stride;
    size_t work_amount;
};
//////////////////////////////////////////////////////////////////////////////
struct jit_uni_normalize_per_spatial_kernel {
    void (*ker_)(const jit_args_normalize *);

    void operator()(const jit_args_normalize *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_normalize_per_spatial_kernel(bool channel_shared) : ker_(nullptr) {
        is_channel_shared = channel_shared;
    }
    virtual ~jit_uni_normalize_per_spatial_kernel() {}
    bool is_channel_shared = true;
};

struct jit_uni_normalize_across_spatial_kernel {
    void (*ker_)(const jit_args_normalize *);

    void operator()(const jit_args_normalize *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_normalize_across_spatial_kernel(bool channel_shared) : ker_(nullptr) {
        is_channel_shared = channel_shared;
    }
    virtual ~jit_uni_normalize_across_spatial_kernel() {}
    bool is_channel_shared = true;
};

struct jit_uni_sqr_sum_kernel {
    void (*ker_)(const jit_args_normalize *);

    void operator()(const jit_args_normalize *args) {
        assert(ker_);
        ker_(args);
    }

    jit_uni_sqr_sum_kernel() : ker_(nullptr) {}
    virtual ~jit_uni_sqr_sum_kernel() {}
};

/////////////////////////////////////////////////////////////////////////////
template <cpu_isa_t isa>
struct jit_uni_normalize_across_spatial_kernel_f32
        : public jit_uni_normalize_across_spatial_kernel,
          public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_normalize_across_spatial_kernel_f32)

    explicit jit_uni_normalize_across_spatial_kernel_f32(bool channel_shared)
        : jit_uni_normalize_across_spatial_kernel(channel_shared), jit_generator() {
        this->preamble();
        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_sqrt_sum, ptr[reg_params + GET_OFF(sqrt_sum)]);
        mov(reg_weights, ptr[reg_params + GET_OFF(weights)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_stride, ptr[reg_params + GET_OFF(stride)]);

        Xbyak::Label div_scale_loop_label;
        Xbyak::Label div_scale_loop_end_label;
        uni_vbroadcastss(vmm_norm, ptr[reg_sqrt_sum]);

        uni_vbroadcastss(vmm_scale, ptr[reg_weights]);

        L(div_scale_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(div_scale_loop_end_label, T_NEAR);

            uni_vmovups(vmm_val, ptr[reg_src]);
            uni_vdivps(vmm_val, vmm_val, vmm_norm);
            uni_vmulps(vmm_val, vmm_val, vmm_scale);
            uni_vmovups(ptr[reg_dst], vmm_val);

            add(reg_src, reg_stride);
            add(reg_dst, reg_stride);
            sub(reg_work_amount, 1);

            jmp(div_scale_loop_label, T_NEAR);
        }
        L(div_scale_loop_end_label);
        this->postamble();
        ker_ = (decltype(ker_)) this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == sse42, Xbyak::Xmm, isa == avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_sqrt_sum = r10;
    Xbyak::Reg64 reg_weights = r11;
    Xbyak::Reg64 reg_work_amount = r12;
    Xbyak::Reg64 reg_stride = r13;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_val = Vmm(0);
    Vmm vmm_scale = Vmm(1);
    Vmm vmm_norm = Vmm(2);
};

template <cpu_isa_t isa>
struct jit_uni_sqr_sum_kernel_f32 : public jit_uni_sqr_sum_kernel,
                                    public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_sqr_sum_kernel_f32)

    jit_uni_sqr_sum_kernel_f32() : jit_uni_sqr_sum_kernel(), jit_generator() {
        this->preamble();
        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_sqr_sums, ptr[reg_params + GET_OFF(sqr_sums)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_stride, ptr[reg_params + GET_OFF(stride)]);

        Xbyak::Label sqr_sum_loop_label;
        Xbyak::Label sqr_sum_loop_end_label;

        uni_vpxor(vmm_sqr_sum, vmm_sqr_sum, vmm_sqr_sum);
        L(sqr_sum_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(sqr_sum_loop_end_label, T_NEAR);

            uni_vmovups(vmm_val, ptr[reg_src]);
            uni_vfmadd231ps(vmm_sqr_sum, vmm_val, vmm_val);

            add(reg_src, reg_stride);
            sub(reg_work_amount, 1);

            jmp(sqr_sum_loop_label, T_NEAR);
        }
        L(sqr_sum_loop_end_label);
        // hsum+store
        if (isa == sse42) {
            hsum_store(vmm_sqr_sum);
        } else if (isa == avx2) {
            Xbyak::Ymm ymm_sqr_sum = Xbyak::Ymm(vmm_sqr_sum.getIdx());
            vextractf128(xmm_aux1, ymm_sqr_sum, 0);
            vextractf128(xmm_aux2, ymm_sqr_sum, 1);
            addps(xmm_aux1, xmm_aux2);
            hsum_store(xmm_aux1);
        } else {
            Xbyak::Zmm zmm_sqr_sum = Xbyak::Zmm(vmm_sqr_sum.getIdx());
            vextractf32x4(xmm_aux1, zmm_sqr_sum, 0);
            vextractf32x4(xmm_aux2, zmm_sqr_sum, 1);
            addps(xmm_aux1, xmm_aux2);
            vextractf32x4(xmm_aux2, zmm_sqr_sum, 2);
            vextractf32x4(xmm_aux3, zmm_sqr_sum, 3);
            addps(xmm_aux2, xmm_aux3);
            addps(xmm_aux1, xmm_aux2);
            hsum_store(xmm_aux1);
        }
        this->postamble();
        ker_ = (decltype(ker_)) this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == sse42, Xbyak::Xmm, isa == avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_work_amount = r9;
    Xbyak::Reg64 reg_stride = r10;
    Xbyak::Reg64 reg_sqr_sums = rbp;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_val = Vmm(0);
    Vmm vmm_sqr_sum = Vmm(1);
    Xbyak::Xmm xmm_aux1 = Xbyak::Xmm(2);
    Xbyak::Xmm xmm_aux2 = Xbyak::Xmm(3);
    Xbyak::Xmm xmm_aux3 = Xbyak::Xmm(4);

    void hsum_store(Xbyak::Xmm xmm_sqr_sum) {
        movshdup(xmm_aux3, xmm_sqr_sum);  //  sqrt_sum:1,2,3,4; aux3:2,2,4,4
        addps(xmm_sqr_sum, xmm_aux3);     //  sqrt_sum:1+2,2+2,3+4,4+4
        movhlps(xmm_aux3, xmm_sqr_sum);   //  aux3:3+4,4+4,4,4
        addps(xmm_sqr_sum, xmm_aux3);     //  sqrt_sum:1+2+3+4,...
        movss(ptr[reg_sqr_sums], xmm_sqr_sum);
    }
};

template <cpu_isa_t isa>
struct jit_uni_normalize_per_spatial_kernel_f32
        : public jit_uni_normalize_per_spatial_kernel,
          public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_normalize_per_spatial_kernel_f32)

    explicit jit_uni_normalize_per_spatial_kernel_f32(bool channel_shared)
        : jit_uni_normalize_per_spatial_kernel(channel_shared), jit_generator() {
        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_weights, ptr[reg_params + GET_OFF(weights)]);
        mov(reg_eps, ptr[reg_params + GET_OFF(eps)]);
        mov(reg_stride, ptr[reg_params + GET_OFF(stride)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        Xbyak::Label norm2_loop_label;
        Xbyak::Label norm2_loop_end_label;
        Xbyak::Label div_loop_label;
        Xbyak::Label div_loop_end_label;

        mov(aux_reg_work_amount, reg_work_amount);
        mov(aux_reg_src, reg_src);
        uni_vpxor(vmm_sqrt_sum, vmm_sqrt_sum, vmm_sqrt_sum);
        uni_vbroadcastss(vmm_eps, ptr[reg_eps]);
        uni_vaddps(vmm_sqrt_sum, vmm_sqrt_sum, vmm_eps);

        L(norm2_loop_label);
        {
            cmp(aux_reg_work_amount, 0);
            jle(norm2_loop_end_label, T_NEAR);

            uni_vmovups(vmm_val, ptr[aux_reg_src]);
            uni_vfmadd231ps(vmm_sqrt_sum, vmm_val, vmm_val);

            add(aux_reg_src, reg_stride);
            sub(aux_reg_work_amount, 1);

            jmp(norm2_loop_label, T_NEAR);
        }

        L(norm2_loop_end_label);

        uni_vsqrtps(vmm_sqrt_sum, vmm_sqrt_sum);

        mov(aux_reg_work_amount, reg_work_amount);
        mov(aux_reg_src, reg_src);
        if (is_channel_shared) {
            uni_vbroadcastss(vmm_scale, ptr[reg_weights]);
        }
        L(div_loop_label);
        {
            cmp(aux_reg_work_amount, 0);
            jle(div_loop_end_label, T_NEAR);

            uni_vmovups(vmm_val, ptr[aux_reg_src]);

            uni_vdivps(vmm_val, vmm_val, vmm_sqrt_sum);

            if (!is_channel_shared) {
                uni_vbroadcastss(vmm_scale, ptr[reg_weights]);
                add(reg_weights, 1*sizeof(float));
            }
            uni_vmulps(vmm_val, vmm_val, vmm_scale);

            uni_vmovups(ptr[reg_dst], vmm_val);

            add(aux_reg_src, reg_stride);
            add(reg_dst, reg_stride);
            sub(aux_reg_work_amount, 1);

            jmp(div_loop_label, T_NEAR);
        }
        L(div_loop_end_label);

        this->postamble();

        ker_ = (decltype(ker_)) this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == sse42, Xbyak::Xmm, isa == avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 aux_reg_src = r9;
    Xbyak::Reg64 reg_dst = r10;
    Xbyak::Reg64 reg_weights = r11;
    Xbyak::Reg64 reg_work_amount = r12;
    Xbyak::Reg64 aux_reg_work_amount = r13;
    Xbyak::Reg64 reg_stride = r14;
    Xbyak::Reg64 reg_eps = r15;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_val = Vmm(0);
    Vmm vmm_sqrt_sum = Vmm(1);
    Vmm vmm_scale = Vmm(2);
    Vmm vmm_eps = Vmm(3);
};

/////////////////////////////////////////////////////////////////////////////
class NormalizeImpl : public ExtLayerBase {
public:
    explicit NormalizeImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.size() != 1)
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            if (layer->insData[0].lock()->getTensorDesc().getDims().size() < 2 ||
                layer->insData[0].lock()->getTensorDesc().getDims().size() > 4) {
                THROW_IE_EXCEPTION << "Normalize supports from 2D to 4D blobs!";
            }

            MemoryBlob::Ptr tweights = as<MemoryBlob>(layer->blobs.at("weights"));
            if (!tweights) {
                THROW_IE_EXCEPTION << layer->name << "Weights are not initialized or cannot be casted to MemoryBlob for layer Normalize with name '"
                    << layer->name << "'";
            }

            if (tweights->getTensorDesc().getPrecision() == Precision::FP32) {
                weights = tweights;
            } else if (tweights->getTensorDesc().getPrecision() == Precision::BF16) {
                MKLDNNPlugin::BF16Transformer transformer;
                weights = transformer.convertBF16ToFloat(tweights);
            } else {
                // Unknown non supported data type, return an error
                THROW_IE_EXCEPTION << layer->name << "Weights for layer Normalize wiht name '" << layer->name <<
                    "' has unsupported data type " << tweights->getTensorDesc().getPrecision();
            }
            across_spatial = layer->GetParamAsBool("across_spatial", false);
            channel_shared = layer->GetParamAsBool("channel_shared", false);
            eps = layer->GetParamAsFloat("eps");

            block_size = 1;
            if (across_spatial) {
                if (mayiuse(avx512_common)) {
                    normalize_across_spatial_kernel.reset(
                            new jit_uni_normalize_across_spatial_kernel_f32<avx512_common>(channel_shared));
                    sqr_sum_kernel.reset(
                            new jit_uni_sqr_sum_kernel_f32<avx512_common>());
                    block_size = 16;
                } else if (mayiuse(avx2)) {
                    normalize_across_spatial_kernel.reset(
                            new jit_uni_normalize_across_spatial_kernel_f32<avx2>(channel_shared));
                    sqr_sum_kernel.reset(
                            new jit_uni_sqr_sum_kernel_f32<avx2>());
                    block_size = 8;
                } else if (mayiuse(sse42)) {
                    normalize_across_spatial_kernel.reset(
                            new jit_uni_normalize_across_spatial_kernel_f32<sse42>(channel_shared));
                    sqr_sum_kernel.reset(
                            new jit_uni_sqr_sum_kernel_f32<sse42>());
                    block_size = 4;
                }
            } else {
                if (mayiuse(avx512_common)) {
                    normalize_per_spatial_kernel.reset(
                            new jit_uni_normalize_per_spatial_kernel_f32<avx512_common>(channel_shared));
                    block_size = 16;
                } else if (mayiuse(avx2)) {
                    normalize_per_spatial_kernel.reset(
                            new jit_uni_normalize_per_spatial_kernel_f32<avx2>(channel_shared));
                    block_size = 8;
                } else if (mayiuse(sse42)) {
                    normalize_per_spatial_kernel.reset(
                            new jit_uni_normalize_per_spatial_kernel_f32<sse42>(channel_shared));
                    block_size = 4;
                }
            }

            addConfig(layer, { { ConfLayout::PLN, false, 0 } }, { { ConfLayout::PLN, false, 0 } }, true);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr> &inputs,
            std::vector<Blob::Ptr> &outputs,
            ResponseDesc *resp) noexcept override {
        auto *src_data = inputs[0]->cbuffer().as<float *>();
        auto *dst_data = outputs[0]->buffer().as<float *>();
        float *scl = weights->buffer().as<float *>();

        int W = (inputs[0]->getTensorDesc().getDims().size() > 3)
                ? inputs[0]->getTensorDesc().getDims()[3]
                : 1;
        int H = (inputs[0]->getTensorDesc().getDims().size() > 2)
                ? inputs[0]->getTensorDesc().getDims()[2]
                : 1;
        int C = (inputs[0]->getTensorDesc().getDims().size() > 1)
                ? inputs[0]->getTensorDesc().getDims()[1]
                : 1;
        int B = (inputs[0]->getTensorDesc().getDims().size() > 0)
                ? inputs[0]->getTensorDesc().getDims()[0]
                : 1;

        for (int b = 0; b < B; b++) {
            float *src_data_b = src_data + b * C * H * W;
            float *dst_data_b = dst_data + b * C * H * W;
            if (across_spatial) {
                int tail_start_sqr_sum = 0;
                float addition_identity_value = 0;
                float sqrt_sum_kernel = 0;
                float sqrt_sum_tail = 0;
                if (sqr_sum_kernel) {
                    size_t advance = (H * W / block_size) * block_size;
                    sqrt_sum_kernel = parallel_sum(C, addition_identity_value, [&](int ic) -> float {
                        float sqr_sum_value = 0;
                        auto arg = jit_args_normalize();
                        arg.src = src_data_b + ic * advance;
                        arg.sqr_sums = static_cast<float*>(&sqr_sum_value);
                        arg.stride = block_size * sizeof(float);
                        arg.work_amount = H * W / block_size;
                        (*sqr_sum_kernel)(&arg);
                        return sqr_sum_value;
                    });
                    tail_start_sqr_sum = advance * C;
                }
                //  all or rest for sqr_sum
                int tail_num_sqr_sum = H * W * C - tail_start_sqr_sum;
                sqrt_sum_tail = parallel_sum(tail_num_sqr_sum, addition_identity_value, [&](int in) -> float {
                    return src_data_b[tail_start_sqr_sum + in] * src_data_b[tail_start_sqr_sum + in];
                });
                float sqrt_sum = sqrt_sum_kernel + sqrt_sum_tail + eps;
                sqrt_sum = std::sqrt(sqrt_sum);

                int tail_start_across_spatial = 0;
                if (normalize_across_spatial_kernel) {
                    tail_start_across_spatial = (H * W / block_size) * block_size;
                    parallel_for(C, [&](int ic) {  //  parallel for each channel, element*scl/sqrt_sum
                        auto arg = jit_args_normalize();
                        arg.src = src_data_b + ic * H * W;
                        arg.dst = dst_data_b + ic * H * W;
                        arg.weights = channel_shared ? scl : &scl[ic];
                        arg.sqrt_sum = &sqrt_sum;
                        arg.stride = block_size*sizeof(float);
                        arg.work_amount = H * W / block_size;

                        (*normalize_across_spatial_kernel)(&arg);
                        //  rest for this channel
                        for (int tail = tail_start_across_spatial; tail < H * W; tail++) {
                            dst_data_b[ic * H * W + tail] = src_data_b[ic * H * W + tail] / sqrt_sum;
                            dst_data_b[ic * H * W + tail] = channel_shared
                                    ? dst_data_b[ic * H * W + tail] * scl[0]
                                    : dst_data_b[ic * H * W + tail] * scl[ic];
                        }
                    });
                } else {
                    for (int c = 0; c < C; c++) {
                        int hw = 0;
                        float s = channel_shared ? scl[0] : scl[c];
                        for (; hw < H * W; hw++) {
                            dst_data_b[c * H * W + hw]
                                    = (src_data_b[c * H * W + hw] / sqrt_sum) * s;
                        }
                    }
                }
            } else {
                int tail_start_per_spatial = 0;
                if (normalize_per_spatial_kernel) {
                    int blocks_num = H * W / block_size;
                    parallel_for(blocks_num, [&](int ib) {
                        auto arg = jit_args_normalize();

                        arg.src = src_data_b + ib * block_size;
                        arg.dst = dst_data_b + ib * block_size;
                        arg.weights = scl;
                        arg.eps = &eps;
                        arg.stride = static_cast<size_t>((size_t)(H) * W * sizeof(float));
                        arg.work_amount = static_cast<size_t>(C);

                        (*normalize_per_spatial_kernel)(&arg);
                    });
                    tail_start_per_spatial = (H * W / block_size) * block_size;
                }
                parallel_for(H * W - tail_start_per_spatial, [&](int i) {
                    int offset = i + tail_start_per_spatial;

                    float norm = eps;
                    for (int c = 0; c < C; c++) {
                        const float *src_data_b_c = src_data_b + c * W * H;
                        norm += src_data_b_c[offset] * src_data_b_c[offset];
                    }

                    norm = std::sqrt(norm);

                    for (int c = 0; c < C; c++) {
                        const float *src_data_b_c = src_data_b + c * W * H;
                        float *dst_data_b_c = dst_data_b + c * W * H;

                        dst_data_b_c[offset] = channel_shared
                                ? (src_data_b_c[offset] / norm * scl[0])
                                : (src_data_b_c[offset] / norm * scl[c]);
                    }
                });
            }
        }

        return OK;
    }

private:
    int block_size;
    std::shared_ptr<jit_uni_normalize_per_spatial_kernel> normalize_per_spatial_kernel;
    std::shared_ptr<jit_uni_normalize_across_spatial_kernel> normalize_across_spatial_kernel;
    std::shared_ptr<jit_uni_sqr_sum_kernel> sqr_sum_kernel;

    MemoryBlob::Ptr weights;
    bool across_spatial = true;
    bool channel_shared = true;
    float eps = 1e-10f;
};

REG_FACTORY_FOR(ImplFactory<NormalizeImpl>, Normalize);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
