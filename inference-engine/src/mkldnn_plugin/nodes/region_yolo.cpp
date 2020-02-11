// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "list.hpp"
#include "base.hpp"
#include "common/defs.h"
#include "common/softmax.h"
#include <vector>
#include <algorithm>
#include <memory>
#include <ie_parallel.hpp>
#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"

using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

#define GET_OFF(field) offsetof(jit_args_logistic, field)

struct jit_args_logistic {
    const float* src;
    const float* dst;
    size_t work_amount;
};

struct jit_uni_logistic_kernel {
    void (*ker_)(const jit_args_logistic *);

    void operator()(const jit_args_logistic *args) { assert(ker_); ker_(args); }

    jit_uni_logistic_kernel() : ker_(nullptr) {}
    virtual ~jit_uni_logistic_kernel() {}
};

template <cpu_isa_t isa>
struct jit_uni_logistic_kernel_f32 : public jit_uni_logistic_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_logistic_kernel_f32)

    jit_uni_logistic_kernel_f32() : jit_uni_logistic_kernel(), jit_generator() {
        exp_injector.reset(new jit_uni_eltwise_injector_f32<isa>(this, alg_kind::eltwise_exp, 0.f, 0.f));

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_table, l_table);

        Xbyak::Label main_loop_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label exit_label;

        int step = vlen / sizeof(float);
        L(main_loop_label); {
            cmp(reg_work_amount, step);
            jl(tail_loop_label, T_NEAR);

            uni_vmovups(vmm_src, ptr[reg_src]);
            compute_kernel();
            uni_vmovups(ptr[reg_dst], vmm_src);

            add(reg_src, step * sizeof(float));
            add(reg_dst, step * sizeof(float));
            sub(reg_work_amount, step);

            jmp(main_loop_label, T_NEAR);
        }

        step = 1;
        L(tail_loop_label); {
            cmp(reg_work_amount, step);
            jl(exit_label, T_NEAR);

            movss(xmm_src, ptr[reg_src]);
            compute_kernel();
            movss(ptr[reg_dst], xmm_src);

            add(reg_src, step * sizeof(float));
            add(reg_dst, step * sizeof(float));
            sub(reg_work_amount, step);

            jmp(tail_loop_label, T_NEAR);
        }

        L(exit_label);

        this->postamble();

        exp_injector->prepare_table();

        prepare_table();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == sse42, Xbyak::Xmm, isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Address table_val(int index) { return ptr[reg_table + index * vlen]; }

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_table = r10;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_aux0 = Vmm(0);
    Vmm vmm_src = Vmm(1);
    Xbyak::Xmm xmm_src = Xbyak::Xmm(1);
    Vmm vmm_aux1 = Vmm(2);
    Vmm vmm_aux2 = Vmm(3);

    const Xbyak::Opmask k_mask = Xbyak::Opmask(1);

    Xbyak::Label l_table;

    std::shared_ptr<jit_uni_eltwise_injector_f32<isa>> exp_injector;

    void compute_kernel() {
        uni_vmovups(vmm_aux0, vmm_src);
        uni_vandps(vmm_aux0, vmm_aux0, table_val(0));
        uni_vorps(vmm_src, vmm_src, table_val(0));

        exp_injector->compute_vector_range(vmm_src.getIdx(), vmm_src.getIdx() + 1);

        uni_vmovups(vmm_aux1, vmm_src);
        uni_vaddps(vmm_aux1, vmm_aux1, table_val(1));
        uni_vdivps(vmm_src, vmm_src, vmm_aux1);

        uni_vmovups(vmm_aux2, table_val(1));
        uni_vsubps(vmm_aux2, vmm_aux2, vmm_src);

        if (isa == sse42) {
            uni_vblendvps(vmm_aux2, vmm_aux2, vmm_src, vmm_aux0);
            uni_vmovups(vmm_src, vmm_aux2);
        } else if (isa == avx2) {
            uni_vblendvps(vmm_src, vmm_aux2, vmm_src, vmm_aux0);
        } else {
            vptestmd(k_mask, vmm_aux0, vmm_aux0);
            vblendmps(vmm_src | k_mask, vmm_aux2, vmm_src);
        }
    }

    void prepare_table() {
        auto broadcast_int = [&](int val) {
            for (size_t d = 0; d < vlen / sizeof(float); ++d) {
                dd(val);
            }
        };

        align(64);
        L(l_table);

        broadcast_int(vals_for_logistic_activate.mask_sign);
        broadcast_int(vals_for_logistic_activate.float_1);
    }

    const struct vals_for_logistic_activate_type {
        int mask_sign = 0x80000000;  // 0 //  mask to extract sign
        int float_1   = 0x3f800000;  // 1 //  1.0f
    } vals_for_logistic_activate;
};

class RegionYoloImpl: public ExtLayerBase {
public:
    explicit RegionYoloImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            classes = layer->GetParamAsInt("classes");
            coords = layer->GetParamAsInt("coords");
            num = layer->GetParamAsInt("num");
            do_softmax = layer->GetParamAsBool("do_softmax", true);
            mask = layer->GetParamAsInts("mask", {});

            block_size = 1;
            if (mayiuse(avx512_common)) {
                logistic_kernel.reset(new jit_uni_logistic_kernel_f32<avx512_common>());
                block_size = 16;
            } else if (mayiuse(avx2)) {
                logistic_kernel.reset(new jit_uni_logistic_kernel_f32<avx2>());
                block_size = 8;
            } else if (mayiuse(sse42)) {
                logistic_kernel.reset(new jit_uni_logistic_kernel_f32<sse42>());
                block_size = 4;
            }

            softmax_kernel.reset(new SoftmaxGeneric());

            addConfig(layer, {DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const auto *src_data = inputs[0]->cbuffer().as<const float *>();
        auto *dst_data = outputs[0]->buffer().as<float *>();

        int mask_size = mask.size();

        int IW = (inputs[0]->getTensorDesc().getDims().size() > 3) ? inputs[0]->getTensorDesc().getDims()[3] : 1;
        int IH = (inputs[0]->getTensorDesc().getDims().size() > 2) ? inputs[0]->getTensorDesc().getDims()[2] : 1;
        int IC = (inputs[0]->getTensorDesc().getDims().size() > 1) ? inputs[0]->getTensorDesc().getDims()[1] : 1;
        int B = (inputs[0]->getTensorDesc().getDims().size() > 0) ? inputs[0]->getTensorDesc().getDims()[0] : 1;

        parallel_for(B * IC * IH * IW, [&](int i) {
            dst_data[i] = src_data[i];
        });

        int end_index = 0;
        int num_ = 0;
        if (do_softmax) {
            // Region layer (Yolo v2)
            end_index = IW * IH;
            num_ = num;
        } else {
            // Yolo layer (Yolo v3)
            end_index = IW * IH * (classes + 1);
            num_ = mask_size;
        }
        int inputs_size = IH * IW * num_ * (classes + coords + 1);
        int total_size = 2 * IH * IW;

        for (int b = 0; b < B; b++) {
            for (int n = 0; n < num_; n++) {
                int index = b * inputs_size + n * IW * IH * (classes + coords + 1);
                calculate_logistic(index, total_size, dst_data);

                index = b * inputs_size + IW * IH * (n * (classes + coords + 1) + coords);
                calculate_logistic(index, end_index, dst_data);
            }
        }

        if (do_softmax) {
            int index = IW * IH * (coords + 1);
            int batch_offset = inputs_size / num;
            for (int b = 0; b < B * num; b++)
                softmax_kernel->execute(src_data + index + b * batch_offset, dst_data + index + b * batch_offset, 1, classes, IH, IW);
        }

        return OK;
    }

private:
    int classes;
    int coords;
    int num;
    float do_softmax;
    std::vector<int> mask;

    int block_size;
    std::shared_ptr<jit_uni_logistic_kernel> logistic_kernel;
    std::shared_ptr<SoftmaxGeneric> softmax_kernel;

    union U {
        float as_float_value;
        int as_int_value;
    };

    inline float logistic_scalar(float src) {
        U aux2;
        aux2.as_float_value = src;
        int sign = aux2.as_int_value >> 31;
        if (sign == 0)
            src *= -1;

        src = std::exp(src);

        src = src / (src + 1);
        if (sign == 0)
            src = 1 - src;

        return src;
    }

    inline void calculate_logistic(int start_index, int count, float* dst_data) {
        if (logistic_kernel) {
            int blocks_num = div_up(count, block_size);
            parallel_for(blocks_num, [&](int ib) {
                int idx = ib * block_size;
                int work_amount = std::min(count - idx, block_size);

                auto arg = jit_args_logistic();
                arg.src = dst_data + start_index + idx;
                arg.dst = dst_data + start_index + idx;
                arg.work_amount = static_cast<size_t>(work_amount);

                (*logistic_kernel)(&arg);
            });
        } else {
            for (int i = 0; i < count; i++) {
                dst_data[i + start_index] = logistic_scalar(dst_data[i + start_index]);
            }
        }
    }
};

REG_FACTORY_FOR(ImplFactory<RegionYoloImpl>, RegionYolo);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
