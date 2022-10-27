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

        float* xmin_ptr;
        float* ymin_ptr;
        float* xmax_ptr;
        float* ymax_ptr;

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
    using Vmm = typename x64::cpu_isa_traits<isa>::Vmm;

    using PReg32 = RegistersPool::Reg<Reg32>;
    using PReg64 = RegistersPool::Reg<Reg64>;
    using PVmm = RegistersPool::Reg<Vmm>;

    using Operand = Xbyak::Operand;

    static constexpr unsigned VCMPPS_GT = 0x0e;
    static constexpr unsigned VCMPPS_GE = 0x0d;
    static constexpr unsigned simd_width = details::x64::cpu_isa_traits<isa>::vlen / sizeof(float);

    void get_box_ptr(const Reg64& boxes_ptr, const Reg64& box_idx, const Reg64& result);
    void get_box_coords_ptr(const Reg64& box_ptr, const Reg64& coords_array_ptr, const Reg64& result);
    void load_simd_register(const Vmm& reg, const Reg64& buff_ptr, const Reg64& buff_size, const Reg64& index);
    void get_simd_tail_length(const Reg64& buff_size, const Reg64& index, const Reg64& result);

private:
    RegistersPool::Ptr reg_pool_;
};


struct jit_uni_multiclass_nms_kernel_fallback : public jit_uni_multiclass_nms_kernel {
    void create_ker() override {
        ker_ = fallback;
    }

private:
    static void fallback(const jit_nms_call_args* args) {
        int num_boxes_selected = 0;
        float iou_threshold = args->iou_threshold;
        for (int i = 0; i < args->num_boxes; ++i) {
            const Box& candidate_box = args->boxes_ptr[i];
            bool candidate_box_selected = true;
            const float* const candidate_box_coords = &args->coords_ptr[candidate_box.box_idx * 4];
            const bool scoreI_equal_to_threshold = (candidate_box.score == args->score_threshold);
            const int from_idx = scoreI_equal_to_threshold ? std::max(num_boxes_selected - 1, 0) : 0; // TODO: bug in reference impl?
            for (int j = from_idx; j < num_boxes_selected; ++j) {
                const float iou = intersection_over_union(candidate_box_coords,
                                                          args,
                                                          j,
                                                          args->coordinates_offset);
                candidate_box_selected = (iou < iou_threshold);
                if (!candidate_box_selected)
                    break;
            }
            if (candidate_box_selected) {
                if (iou_threshold > 0.5f) {
                    iou_threshold *= args->nms_eta;
                }
                args->boxes_ptr[num_boxes_selected] = candidate_box;
                args->xmin_ptr[num_boxes_selected] = candidate_box_coords[0];
                args->ymin_ptr[num_boxes_selected] = candidate_box_coords[1];
                args->xmax_ptr[num_boxes_selected] = candidate_box_coords[2];
                args->ymax_ptr[num_boxes_selected] = candidate_box_coords[3];
                ++num_boxes_selected;
            }
        }

        *args->num_boxes_selected_ptr = num_boxes_selected;
    }

    static float intersection_over_union(const float* boxI, const jit_nms_call_args* args, int j, const float norm) {
        float xminI = boxI[0];
        float yminI = boxI[1];
        float xmaxI = boxI[2];
        float ymaxI = boxI[3];
        float xminJ = args->xmin_ptr[j];
        float yminJ = args->ymin_ptr[j];
        float xmaxJ = args->xmax_ptr[j];
        float ymaxJ = args->ymax_ptr[j];

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
