// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cpu/x64/jit_generator.hpp>
#include <cassert>
#include <memory>
#include <vector>
#include "registers_pool.hpp"

namespace ov {
namespace intel_cpu {
namespace node {
namespace details {
namespace x64 = dnnl::impl::cpu::x64;
} // namespace details


class jit_uni_multiclass_nms_kernel {
public:
    struct Box {
        float score;
        int batch_idx;
        int class_idx;
        int box_idx;
    };

    struct jit_nms_call_args {
        float iou_threshold;
        float score_threshold;
        float nms_eta;
        float coordinates_offset;

        Box* boxes_ptr;
        int num_boxes;
        const float* coords_ptr;

        std::size_t* num_boxes_selected_ptr;
    };

    void operator()(const jit_nms_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    virtual ~jit_uni_multiclass_nms_kernel() = default;

    virtual void create_ker() = 0;

protected:
    void (*ker_)(const jit_nms_call_args*) = nullptr;
};


template <details::x64::cpu_isa_t isa>
struct jit_uni_multiclass_nms_kernel_impl : public jit_uni_multiclass_nms_kernel, public details::x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_multiclass_nms_kernel_impl)

    jit_uni_multiclass_nms_kernel_impl();

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override;

private:
    using Reg32 = Xbyak::Reg32;
    using Reg64 = Xbyak::Reg64;
    using Xmm = Xbyak::Xmm;
    using Operand = Xbyak::Operand;

    using Vmm = typename x64::cpu_isa_traits<isa>::Vmm;

    using PReg32 = RegistersPool::Reg<Reg32>;
    using PReg64 = RegistersPool::Reg<Reg64>;
    using PXmm = RegistersPool::Reg<Xmm>;
    using PVmm = RegistersPool::Reg<Vmm>;

    static constexpr unsigned VCMPPS_GT = 0x0e;
    static constexpr unsigned VCMPPS_GE = 0x0d;
    static constexpr unsigned simd_width = details::x64::cpu_isa_traits<isa>::vlen / sizeof(float);

    void inline_get_box_ptr(Reg64 boxes_ptr, Reg64 box_idx, Reg64 result);
    void inline_get_box_coords_ptr(Reg64 box_ptr, Reg64 coords_array_ptr, Reg64 result);
    void inline_get_box_coords_ptr(Reg64 boxes_ptr, Reg64 box_idx, Reg64 coords_vector_ptr, Reg64 result);
    void inline_pinsrd(const Vmm& x1, const Operand& op, const int imm);

private:
    Reg64 reg_params_;

    RegistersPool::Ptr reg_pool_ = RegistersPool::create<isa>({
        Reg64(Operand::RAX), Reg64(Operand::RBP), Reg64(Operand::RSI), Reg64(Operand::RDI)
    });
};


struct jit_uni_multiclass_nms_kernel_fallback : public jit_uni_multiclass_nms_kernel {
    void create_ker() override {
        ker_ = fallback;
    }

private:
    static void fallback(const jit_nms_call_args* args) {
        int num_boxes_selected = 0;
        float iou_threshold = args->iou_threshold;
        for (size_t i = 0; i < args->num_boxes; ++i) {
            bool box_is_selected = true;
            const bool scoreI_equal_to_threshold = (args->boxes_ptr[i].score == args->score_threshold);
            const float* const box_coords = &args->coords_ptr[args->boxes_ptr[i].box_idx * 4];
            for (int j = num_boxes_selected - 1; j >= 0; j--) {
                const float iou = intersection_over_union(box_coords,
                                                          &args->coords_ptr[args->boxes_ptr[j].box_idx * 4],
                                                          args->coordinates_offset);
                box_is_selected = (iou < iou_threshold);
                if (!box_is_selected || scoreI_equal_to_threshold) // TODO: scoreI_equal_to_threshold - bug in reference impl?
                    break;
            }
            if (box_is_selected) {
                if (iou_threshold > 0.5f) {
                    iou_threshold *= args->nms_eta;
                }
                args->boxes_ptr[num_boxes_selected++] = args->boxes_ptr[i];
            }
        }

        *args->num_boxes_selected_ptr = num_boxes_selected;
    }

    static float intersection_over_union(const float* boxesI, const float* boxesJ, const float norm) {
        float xminI = boxesI[0];
        float yminI = boxesI[1];
        float xmaxI = boxesI[2];
        float ymaxI = boxesI[3];
        float xminJ = boxesJ[0];
        float yminJ = boxesJ[1];
        float xmaxJ = boxesJ[2];
        float ymaxJ = boxesJ[3];

        float areaI = (ymaxI - yminI + norm) * (xmaxI - xminI + norm);
        float areaJ = (ymaxJ - yminJ + norm) * (xmaxJ - xminJ + norm);
        if (areaI <= 0.f || areaJ <= 0.f)
            return 0.f;

        float intersection_area = std::max(std::min(ymaxI, ymaxJ) - std::max(yminI, yminJ) + norm, 0.f) *
                                  std::max(std::min(xmaxI, xmaxJ) - std::max(xminI, xminJ) + norm, 0.f);
        return intersection_area / (areaI + areaJ - intersection_area);
    }
};

} // namespace node
} // namespace intel_cpu
} // namespace ov
