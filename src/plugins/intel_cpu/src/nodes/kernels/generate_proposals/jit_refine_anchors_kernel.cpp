// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_refine_anchors_kernel.hpp"
#include "nodes/kernels/registers_pool.hpp"
#include "nodes/kernels/stack_allocator.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @code
 */
template <x64::cpu_isa_t isa>
void jit_refine_anchors_kernel_fp32<isa>::generate() {
    /** @code
        for (int anchor = 0; anchor < anchors_num; ++anchor) {
            const int a_idx = anchor_idx(h, w, anchor, 0);
            const int a_idx_offset = anchor_idx(h, w, anchor, 1) - anchor_idx(h, w, anchor, 0);
            float x0 = anchors[a_idx + 0 * a_idx_offset];
            float y0 = anchors[a_idx + 1 * a_idx_offset];
            float x1 = anchors[a_idx + 2 * a_idx_offset];
            float y1 = anchors[a_idx + 3 * a_idx_offset];

            const int d_idx = delta_idx(anchor, 0, h, w);
            const int d_idx_offset = delta_idx(anchor, 1, h, w) - delta_idx(anchor, 0, h, w);
            const float dx = deltas[d_idx + 0 * d_idx_offset];
            const float dy = deltas[d_idx + 1 * d_idx_offset];
            const float d_log_w = deltas[d_idx + 2 * d_idx_offset];
            const float d_log_h = deltas[d_idx + 3 * d_idx_offset];

            const float score = scores[score_idx(anchor, 0, h, w)];

            // width & height of box
            const float ww = x1 - x0 + coordinates_offset;
            const float hh = y1 - y0 + coordinates_offset;
            // center location of box
            const float ctr_x = x0 + 0.5f * ww;
            const float ctr_y = y0 + 0.5f * hh;

            // new center location according to deltas (dx, dy)
            const float pred_ctr_x = dx * ww + ctr_x;
            const float pred_ctr_y = dy * hh + ctr_y;
            // new width & height according to deltas d(log w), d(log h)
            const float pred_w = std::exp(std::min(d_log_w, max_delta_log_wh)) * ww;
            const float pred_h = std::exp(std::min(d_log_h, max_delta_log_wh)) * hh;

            // update upper-left corner location
            x0 = pred_ctr_x - 0.5f * pred_w;
            y0 = pred_ctr_y - 0.5f * pred_h;
            // update lower-right corner location
            x1 = pred_ctr_x + 0.5f * pred_w - coordinates_offset;
            y1 = pred_ctr_y + 0.5f * pred_h - coordinates_offset;

            // adjust new corner locations to be within the image region,
            x0 = std::max<float>(0.0f, std::min<float>(x0, img_w - coordinates_offset));
            y0 = std::max<float>(0.0f, std::min<float>(y0, img_h - coordinates_offset));
            x1 = std::max<float>(0.0f, std::min<float>(x1, img_w - coordinates_offset));
            y1 = std::max<float>(0.0f, std::min<float>(y1, img_h - coordinates_offset));

            // recompute new width & height
            const float box_w = x1 - x0 + coordinates_offset;
            const float box_h = y1 - y0 + coordinates_offset;

            int p_idx = proposal_idx(h, w, anchor, 0);
            proposals[p_idx + 0] = x0;
            proposals[p_idx + 1] = y0;
            proposals[p_idx + 2] = x1;
            proposals[p_idx + 3] = y1;
            proposals[p_idx + 4] = score;
            proposals[p_idx + 5] = (min_box_w <= box_w) * (min_box_h <= box_h) * 1.0;
        }
     */

    this->preamble();

    StackAllocator allocator{*this};

    xor_(reg_anchors_loop, reg_anchors_loop);
    xor_(reg_anchors_chunk, reg_anchors_chunk);
    xor_(reg_img_h, reg_img_h);
    xor_(reg_img_w, reg_img_w);
    mov(reg_anchors_loop.cvt32(), ptr[reg_params + offsetof(jit_refine_anchors_call_args, anchors_num)]);
    mov(reg_anchors_ptr, ptr[reg_params + offsetof(jit_refine_anchors_call_args, anchors)]);
    mov(reg_deltas_ptr, ptr[reg_params + offsetof(jit_refine_anchors_call_args, deltas)]);
    mov(reg_scores_ptr, ptr[reg_params + offsetof(jit_refine_anchors_call_args, scores)]);
    mov(reg_proposals_ptr, ptr[reg_params + offsetof(jit_refine_anchors_call_args, proposals)]);
    mov(reg_img_w.cvt32(), ptr[reg_params + offsetof(jit_refine_anchors_call_args, img_w)]);
    mov(reg_img_h.cvt32(), ptr[reg_params + offsetof(jit_refine_anchors_call_args, img_h)]);

    const size_t& vmm_reg_size_in_bytes = this->SIMD_WIDTH * sizeof(float);
    std::vector<Xbyak::Xmm> not_available_vmm;

    Xbyak::Label anchor_loop;
    Xbyak::Label loop_mask;
    Xbyak::Label l_mask;
    {
        StackAllocator::Address vmm_anchor_mask_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address vmm_ww_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address vmm_hh_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address vmm_coordinates_offset_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address reg_scale_0_5_addr{allocator, sizeof(float)};
        StackAllocator::Address vmm_scale_0_5_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address vmm_ctr_x_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address vmm_ctr_y_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address vmm_pred_ctr_x_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address vmm_pred_ctr_y_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address reg_max_delta_log_wh_addr{allocator, sizeof(float)};
        StackAllocator::Address vmm_max_delta_log_wh_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address vmm_pred_w_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address vmm_pred_h_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address reg_img_w_addr{allocator, sizeof(float)};
        StackAllocator::Address vmm_img_w_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address reg_img_h_addr{allocator, sizeof(float)};
        StackAllocator::Address vmm_img_h_addr{allocator, vmm_reg_size_in_bytes};
        StackAllocator::Address reg_0_0_addr{allocator, sizeof(float)};
        StackAllocator::Address vmm_0_0_addr{allocator, vmm_reg_size_in_bytes};

        L(anchor_loop);
        {
            mov(reg_anchors_chunk.cvt32(), this->SIMD_WIDTH);
            cmp(reg_anchors_loop.cvt32(), this->SIMD_WIDTH);
            jae(loop_mask);
            mov(reg_anchors_chunk, reg_anchors_loop);
            L(loop_mask);

            /** @code
                const int a_idx = anchor_idx(h, w, anchor, 0);
                const int a_idx_offset = anchor_idx(h, w, anchor, 1) - anchor_idx(h, w, anchor, 0);
                float x0 = anchors[a_idx + 0 * a_idx_offset];
                float y0 = anchors[a_idx + 1 * a_idx_offset];
                float x1 = anchors[a_idx + 2 * a_idx_offset];
                float y1 = anchors[a_idx + 3 * a_idx_offset];
             */
            not_available_vmm = {vmm_x0, vmm_y0, vmm_x1, vmm_y1};

            // Prepare indexes
            Vmm vmm_anchor_idx = this->get_free_vmm<Vmm>(not_available_vmm);
            Vmm vmm_anchor_anchor_offset = this->get_free_vmm<Vmm>(not_available_vmm);
            Vmm vmm_anchor_idx_offset = this->get_free_vmm<Vmm>(not_available_vmm);
            uni_vbroadcastss(vmm_anchor_idx, ptr[reg_params + offsetof(jit_refine_anchors_call_args, anchor_start_idx)]);
            uni_vbroadcastss(vmm_anchor_anchor_offset, ptr[reg_params + offsetof(jit_refine_anchors_call_args, anchor_anchor_offset)]);
            uni_vbroadcastss(vmm_anchor_idx_offset, ptr[reg_params + offsetof(jit_refine_anchors_call_args, anchor_idx_offset)]);
            mov(rbx, ptr[reg_params + offsetof(jit_refine_anchors_call_args, refine_anchor_indices)]);
            uni_vpmulld(vmm_anchor_anchor_offset, vmm_anchor_anchor_offset, ptr[rbx]);
            uni_vpaddd(vmm_anchor_idx, vmm_anchor_idx, vmm_anchor_anchor_offset);

            // Prepare mask
            Vmm vmm_anchor_mask = this->get_free_vmm<Vmm>(not_available_vmm);
            mov(rax.cvt32(), this->SIMD_WIDTH);
            sub(rax, reg_anchors_chunk);
            add(rax, 16);
            sub(rax.cvt32(), this->SIMD_WIDTH);
            mov(rbx, ptr[reg_params + offsetof(jit_refine_anchors_call_args, refine_anchor_masks)]);
            uni_vmovdqu(vmm_anchor_mask, ptr[rbx + rax * sizeof(float)]);
            uni_vmovdqu(vmm_anchor_mask_addr, vmm_anchor_mask);

            {
                // float x0 = anchors[a_idx + 0 * a_idx_offset];
                this->uni_vgatherdps(vmm_x0, reg_anchors_ptr, vmm_anchor_idx, sizeof(float), vmm_anchor_mask);
                // float y0 = anchors[a_idx + 1 * a_idx_offset];
                uni_vmovdqu(vmm_anchor_mask, vmm_anchor_mask_addr);
                uni_vpaddd(vmm_anchor_idx, vmm_anchor_idx, vmm_anchor_idx_offset);
                this->uni_vgatherdps(vmm_y0, reg_anchors_ptr, vmm_anchor_idx, sizeof(float), vmm_anchor_mask);
                // float x1 = anchors[a_idx + 2 * a_idx_offset];
                uni_vmovdqu(vmm_anchor_mask, vmm_anchor_mask_addr);
                uni_vpaddd(vmm_anchor_idx, vmm_anchor_idx, vmm_anchor_idx_offset);
                this->uni_vgatherdps(vmm_x1, reg_anchors_ptr, vmm_anchor_idx, sizeof(float), vmm_anchor_mask);
                // float y1 = anchors[a_idx + 3 * a_idx_offset];
                uni_vmovdqu(vmm_anchor_mask, vmm_anchor_mask_addr);
                uni_vpaddd(vmm_anchor_idx, vmm_anchor_idx, vmm_anchor_idx_offset);
                this->uni_vgatherdps(vmm_y1, reg_anchors_ptr, vmm_anchor_idx, sizeof(float), vmm_anchor_mask);
            }

            /** @code
                const int d_idx = delta_idx(anchor, 0, h, w);
                const int d_idx_offset = delta_idx(anchor, 1, h, w) - delta_idx(anchor, 0, h, w);
                const float dx = deltas[d_idx + 0 * d_idx_offset];
                const float dy = deltas[d_idx + 1 * d_idx_offset];
                const float d_log_w = deltas[d_idx + 2 * d_idx_offset];
                const float d_log_h = deltas[d_idx + 3 * d_idx_offset];
             */
            not_available_vmm = {vmm_x0, vmm_y0, vmm_x1, vmm_y1,
                                 vmm_dx, vmm_dy, vmm_d_log_w, vmm_d_log_h};

            // Prepare indexes
            Vmm vmm_delta_idx = this->get_free_vmm<Vmm>(not_available_vmm);
            Vmm vmm_delta_anchor_offset = this->get_free_vmm<Vmm>(not_available_vmm);
            Vmm vmm_delta_idx_offset = this->get_free_vmm<Vmm>(not_available_vmm);
            uni_vbroadcastss(vmm_delta_idx, ptr[reg_params + offsetof(jit_refine_anchors_call_args, delta_start_idx)]);
            uni_vbroadcastss(vmm_delta_anchor_offset, ptr[reg_params + offsetof(jit_refine_anchors_call_args, delta_anchor_offset)]);
            uni_vbroadcastss(vmm_delta_idx_offset, ptr[reg_params + offsetof(jit_refine_anchors_call_args, delta_idx_offset)]);
            mov(rbx, ptr[reg_params + offsetof(jit_refine_anchors_call_args, refine_anchor_indices)]);
            uni_vpmulld(vmm_delta_anchor_offset, vmm_delta_anchor_offset, ptr[rbx]);
            uni_vpaddd(vmm_delta_idx, vmm_delta_idx, vmm_delta_anchor_offset);

            // Prepare mask
            Vmm vmm_delta_mask = this->get_free_vmm<Vmm>(not_available_vmm);
            uni_vmovdqu(vmm_delta_mask, vmm_anchor_mask_addr);

            {
                // const float dx = deltas[d_idx + 0 * d_idx_offset];
                this->uni_vgatherdps(vmm_dx, reg_deltas_ptr, vmm_delta_idx, sizeof(float), vmm_delta_mask);
                // const float dy = deltas[d_idx + 1 * d_idx_offset];
                uni_vmovdqu(vmm_delta_mask, vmm_anchor_mask_addr);
                uni_vpaddd(vmm_delta_idx, vmm_delta_idx, vmm_delta_idx_offset);
                this->uni_vgatherdps(vmm_dy, reg_deltas_ptr, vmm_delta_idx, sizeof(float), vmm_delta_mask);
                // const float d_log_w = deltas[d_idx + 2 * d_idx_offset];
                uni_vmovdqu(vmm_delta_mask, vmm_anchor_mask_addr);
                uni_vpaddd(vmm_delta_idx, vmm_delta_idx, vmm_delta_idx_offset);
                this->uni_vgatherdps(vmm_d_log_w, reg_deltas_ptr, vmm_delta_idx, sizeof(float), vmm_delta_mask);
                // const float d_log_h = deltas[d_idx + 3 * d_idx_offset];
                uni_vmovdqu(vmm_delta_mask, vmm_anchor_mask_addr);
                uni_vpaddd(vmm_delta_idx, vmm_delta_idx, vmm_delta_idx_offset);
                this->uni_vgatherdps(vmm_d_log_h, reg_deltas_ptr, vmm_delta_idx, sizeof(float), vmm_delta_mask);
            }

            not_available_vmm = {vmm_x0, vmm_y0, vmm_x1, vmm_y1,
                                 vmm_dx, vmm_dy, vmm_d_log_w, vmm_d_log_h};
            Vmm vmm_temp = this->get_free_vmm<Vmm>(not_available_vmm);

            /** @code
                // width & height of box
                const float ww = x1 - x0 + coordinates_offset;
                const float hh = y1 - y0 + coordinates_offset;
             */
            Vmm vmm_ww = vmm_temp;
            Vmm vmm_hh = vmm_temp;
            uni_vbroadcastss(vmm_temp, ptr[reg_params + offsetof(jit_refine_anchors_call_args, coordinates_offset)]);
            uni_vmovdqu(vmm_coordinates_offset_addr, vmm_temp);
            // const float ww = x1 - x0 + coordinates_offset;
            uni_vsubps(vmm_ww, vmm_x1, vmm_x0);
            uni_vaddps(vmm_ww, vmm_ww, vmm_coordinates_offset_addr);
            uni_vmovdqu(vmm_ww_addr, vmm_ww);
            // const float hh = y1 - y0 + coordinates_offset;
            uni_vsubps(vmm_hh, vmm_y1, vmm_y0);
            uni_vaddps(vmm_hh, vmm_hh, vmm_coordinates_offset_addr);
            uni_vmovdqu(vmm_hh_addr, vmm_hh);

            /** @code
                // center location of box
                const float ctr_x = x0 + 0.5f * ww;
                const float ctr_y = y0 + 0.5f * hh;
             */
            Vmm vmm_ctr_x = vmm_temp;
            Vmm vmm_ctr_y = vmm_temp;
            mov(rax.cvt32(), dnnl::impl::float2int(0.5f));
            mov(reg_scale_0_5_addr, rax.cvt32());
            uni_vbroadcastss(vmm_temp, reg_scale_0_5_addr);
            uni_vmovdqu(vmm_scale_0_5_addr, vmm_temp);
            // const float ctr_x = x0 + 0.5f * ww;
            uni_vmovdqu(vmm_ww, vmm_ww_addr);
            uni_vmulps(vmm_ctr_x, vmm_ww, vmm_scale_0_5_addr);
            uni_vaddps(vmm_ctr_x, vmm_ctr_x, vmm_x0);
            uni_vmovdqu(vmm_ctr_x_addr, vmm_ctr_x);
            // const float ctr_y = y0 + 0.5f * hh;
            uni_vmovdqu(vmm_hh, vmm_hh_addr);
            uni_vmulps(vmm_ctr_y, vmm_hh, vmm_scale_0_5_addr);
            uni_vaddps(vmm_ctr_y, vmm_ctr_y, vmm_y0);
            uni_vmovdqu(vmm_ctr_y_addr, vmm_ctr_y);

            /** @code
                // new center location according to deltas (dx, dy)
                const float pred_ctr_x = dx * ww + ctr_x;
                const float pred_ctr_y = dy * hh + ctr_y;
             */
            Vmm vmm_pred_ctr_x = vmm_temp;
            Vmm vmm_pred_ctr_y = vmm_temp;
            // const float pred_ctr_x = dx * ww + ctr_x;
            uni_vmulps(vmm_pred_ctr_x, vmm_dx, vmm_ww_addr);
            uni_vaddps(vmm_pred_ctr_x, vmm_pred_ctr_x, vmm_ctr_x_addr);
            uni_vmovdqu(vmm_pred_ctr_x_addr, vmm_pred_ctr_x);
            // const float pred_ctr_y = dy * hh + ctr_y;
            uni_vmulps(vmm_pred_ctr_y, vmm_dy, vmm_hh_addr);
            uni_vaddps(vmm_pred_ctr_y, vmm_pred_ctr_y, vmm_ctr_y_addr);
            uni_vmovdqu(vmm_pred_ctr_y_addr, vmm_pred_ctr_y);

            /** @code
                // new width & height according to deltas d(log w), d(log h)
                const float pred_w = std::exp(std::min(d_log_w, max_delta_log_wh)) * ww;
                const float pred_h = std::exp(std::min(d_log_h, max_delta_log_wh)) * hh;
             */
            Vmm vmm_pred_w = vmm_temp;
            Vmm vmm_pred_h = vmm_temp;
            mov(rax.cvt32(), ptr[reg_params + offsetof(jit_refine_anchors_call_args, max_delta_log_wh)]);
            mov(reg_max_delta_log_wh_addr, rax.cvt32());
            uni_vbroadcastss(vmm_temp, reg_max_delta_log_wh_addr);
            uni_vmovdqu(vmm_max_delta_log_wh_addr, vmm_temp);
            // const float pred_w = std::exp(std::min(d_log_w, max_delta_log_wh)) * ww;
            uni_vminps(vmm_pred_w, vmm_d_log_w, vmm_max_delta_log_wh_addr);
            uni_expf(vmm_pred_w);
            uni_vmulps(vmm_pred_w, vmm_pred_w, vmm_ww_addr);
            uni_vmovdqu(vmm_pred_w_addr, vmm_pred_w);
            // const float pred_h = std::exp(std::min(d_log_h, max_delta_log_wh)) * hh;
            uni_vminps(vmm_pred_h, vmm_d_log_h, vmm_max_delta_log_wh_addr);
            uni_expf(vmm_pred_h);
            uni_vmulps(vmm_pred_h, vmm_pred_h, vmm_hh_addr);
            uni_vmovdqu(vmm_pred_h_addr, vmm_pred_h);

            /** @code
                // update upper-left corner location
                x0 = pred_ctr_x - 0.5f * pred_w;
                y0 = pred_ctr_y - 0.5f * pred_h;
             */
            // x0 = pred_ctr_x - 0.5f * pred_w;
            uni_vmovdqu(vmm_pred_w, vmm_pred_w_addr);
            uni_vmulps(vmm_x0, vmm_pred_w, vmm_scale_0_5_addr);
            uni_vmovdqu(vmm_pred_ctr_x, vmm_pred_ctr_x_addr);
            uni_vsubps(vmm_x0, vmm_pred_ctr_x, vmm_x0);
            // y0 = pred_ctr_y - 0.5f * pred_h;
            uni_vmovdqu(vmm_pred_h, vmm_pred_h_addr);
            uni_vmulps(vmm_y0, vmm_pred_h, vmm_scale_0_5_addr);
            uni_vmovdqu(vmm_pred_ctr_y, vmm_pred_ctr_y_addr);
            uni_vsubps(vmm_y0, vmm_pred_ctr_y, vmm_y0);

            /** @code
                // update lower-right corner location
                x1 = pred_ctr_x + 0.5f * pred_w - coordinates_offset;
                y1 = pred_ctr_y + 0.5f * pred_h - coordinates_offset;
             */
            // x1 = pred_ctr_x + 0.5f * pred_w - coordinates_offset;
            uni_vmovdqu(vmm_pred_w, vmm_pred_w_addr);
            uni_vmulps(vmm_x1, vmm_pred_w, vmm_scale_0_5_addr);
            uni_vsubps(vmm_x1, vmm_x1, vmm_coordinates_offset_addr);
            uni_vmovdqu(vmm_pred_ctr_x, vmm_pred_ctr_x_addr);
            uni_vaddps(vmm_x1, vmm_pred_ctr_x, vmm_x1);
            // y1 = pred_ctr_y + 0.5f * pred_h - coordinates_offset;
            uni_vmovdqu(vmm_pred_h, vmm_pred_h_addr);
            uni_vmulps(vmm_y1, vmm_pred_h, vmm_scale_0_5_addr);
            uni_vsubps(vmm_y1, vmm_y1, vmm_coordinates_offset_addr);
            uni_vmovdqu(vmm_pred_ctr_y, vmm_pred_ctr_y_addr);
            uni_vaddps(vmm_y1, vmm_pred_ctr_y, vmm_y1);

            /** @code
                // adjust new corner locations to be within the image region,
                x0 = std::max<float>(0.0f, std::min<float>(x0, img_w - coordinates_offset));
                y0 = std::max<float>(0.0f, std::min<float>(y0, img_h - coordinates_offset));
             */
            mov(reg_img_w_addr, reg_img_w.cvt32());
            uni_vbroadcastss(vmm_temp, reg_img_w_addr);
            uni_vsubps(vmm_temp, vmm_temp, vmm_coordinates_offset_addr);
            uni_vmovdqu(vmm_img_w_addr, vmm_temp);

            mov(reg_img_h_addr, reg_img_h.cvt32());
            uni_vbroadcastss(vmm_temp, reg_img_h_addr);
            uni_vsubps(vmm_temp, vmm_temp, vmm_coordinates_offset_addr);
            uni_vmovdqu(vmm_img_h_addr, vmm_temp);

            mov(rax.cvt32(), dnnl::impl::float2int(0.0f));
            mov(reg_0_0_addr, rax.cvt32());
            uni_vbroadcastss(vmm_temp, reg_0_0_addr);
            uni_vmovdqu(vmm_0_0_addr, vmm_temp);

            // x0 = std::max<float>(0.0f, std::min<float>(x0, img_w - coordinates_offset));
            uni_vminps(vmm_x0, vmm_x0, vmm_img_w_addr);
            uni_vmaxps(vmm_x0, vmm_x0, vmm_0_0_addr);
            // y0 = std::max<float>(0.0f, std::min<float>(y0, img_h - coordinates_offset));
            uni_vminps(vmm_y0, vmm_y0, vmm_img_h_addr);
            uni_vmaxps(vmm_y0, vmm_y0, vmm_0_0_addr);

            /** @code
                // adjust new corner locations to be within the image region,
                x1 = std::max<float>(0.0f, std::min<float>(x1, img_w - coordinates_offset));
                y1 = std::max<float>(0.0f, std::min<float>(y1, img_h - coordinates_offset));
             */
            // x1 = std::max<float>(0.0f, std::min<float>(x1, img_w - coordinates_offset));
            uni_vminps(vmm_x1, vmm_x1, vmm_img_w_addr);
            uni_vmaxps(vmm_x1, vmm_x1, vmm_0_0_addr);
            // y1 = std::max<float>(0.0f, std::min<float>(y1, img_h - coordinates_offset));
            uni_vminps(vmm_y1, vmm_y1, vmm_img_h_addr);
            uni_vmaxps(vmm_y1, vmm_y1, vmm_0_0_addr);

            /** @code
                int p_idx = proposal_idx(h, w, anchor, 0);
                proposals[p_idx + 0] = x0;
                proposals[p_idx + 1] = y0;
                proposals[p_idx + 2] = x1;
                proposals[p_idx + 3] = y1;
                ...
             */
            not_available_vmm = {vmm_x0, vmm_y0, vmm_x1, vmm_y1};

            // Prepare indexes
            Vmm vmm_proposals_idx = this->get_free_vmm<Vmm>(not_available_vmm);
            Vmm vmm_proposals_anchor_offset = this->get_free_vmm<Vmm>(not_available_vmm);
            Vmm vmm_proposals_idx_offset = this->get_free_vmm<Vmm>(not_available_vmm);
            uni_vbroadcastss(vmm_proposals_idx, ptr[reg_params + offsetof(jit_refine_anchors_call_args, proposal_start_idx)]);
            uni_vbroadcastss(vmm_proposals_anchor_offset, ptr[reg_params + offsetof(jit_refine_anchors_call_args, proposal_anchor_offset)]);
            uni_vbroadcastss(vmm_proposals_idx_offset, ptr[reg_params + offsetof(jit_refine_anchors_call_args, proposal_idx_offset)]);
            mov(rbx, ptr[reg_params + offsetof(jit_refine_anchors_call_args, refine_anchor_indices)]);
            uni_vpmulld(vmm_proposals_anchor_offset, vmm_proposals_anchor_offset, ptr[rbx]);
            uni_vpaddd(vmm_proposals_idx, vmm_proposals_idx, vmm_proposals_anchor_offset);

            // Prepare mask
            Vmm vmm_proposals_mask = this->get_free_vmm<Vmm>(not_available_vmm);
            uni_vmovdqu(vmm_proposals_mask, vmm_anchor_mask_addr);

            {
                // proposals[p_idx + 0] = x0;
                uni_vscatterdps(reg_proposals_ptr, vmm_proposals_idx, sizeof(float), vmm_x0, vmm_proposals_mask);
                // proposals[p_idx + 1] = y0;
                uni_vmovdqu(vmm_proposals_mask, vmm_anchor_mask_addr);
                uni_vpaddd(vmm_proposals_idx, vmm_proposals_idx, vmm_proposals_idx_offset);
                uni_vscatterdps(reg_proposals_ptr, vmm_proposals_idx, sizeof(float), vmm_y0, vmm_proposals_mask);
                // proposals[p_idx + 2] = x1;
                uni_vmovdqu(vmm_proposals_mask, vmm_anchor_mask_addr);
                uni_vpaddd(vmm_proposals_idx, vmm_proposals_idx, vmm_proposals_idx_offset);
                uni_vscatterdps(reg_proposals_ptr, vmm_proposals_idx, sizeof(float), vmm_x1, vmm_proposals_mask);
                // proposals[p_idx + 3] = y1;
                uni_vmovdqu(vmm_proposals_mask, vmm_anchor_mask_addr);
                uni_vpaddd(vmm_proposals_idx, vmm_proposals_idx, vmm_proposals_idx_offset);
                uni_vscatterdps(reg_proposals_ptr, vmm_proposals_idx, sizeof(float), vmm_y1, vmm_proposals_mask);
            }

            /** @code
                const float score = scores[score_idx(anchor, 0, h, w)];
             */
            Vmm vmm_score = this->get_free_vmm<Vmm>(not_available_vmm);

            // Prepare indexes
            Vmm vmm_score_idx = this->get_free_vmm<Vmm>(not_available_vmm);
            Vmm vmm_score_anchor_offset = this->get_free_vmm<Vmm>(not_available_vmm);
            uni_vbroadcastss(vmm_score_idx, ptr[reg_params + offsetof(jit_refine_anchors_call_args, score_start_idx)]);
            uni_vbroadcastss(vmm_score_anchor_offset, ptr[reg_params + offsetof(jit_refine_anchors_call_args, score_anchor_offset)]);
            mov(rbx, ptr[reg_params + offsetof(jit_refine_anchors_call_args, refine_anchor_indices)]);
            uni_vpmulld(vmm_score_anchor_offset, vmm_score_anchor_offset, ptr[rbx]);
            uni_vpaddd(vmm_score_idx, vmm_score_idx, vmm_score_anchor_offset);

            // Prepare mask
            Vmm vmm_score_mask = this->get_free_vmm<Vmm>(not_available_vmm);
            uni_vmovdqu(vmm_score_mask, vmm_anchor_mask_addr);

            {
                // const float score = scores[score_idx(anchor, 0, h, w)];
                this->uni_vgatherdps(vmm_score, reg_scores_ptr, vmm_score_idx, sizeof(float), vmm_score_mask);
            }

            /** @code
                int p_idx = proposal_idx(h, w, anchor, 0);
                ...
                proposals[p_idx + 4] = score;
                ...
             */
            {
                // proposals[p_idx + 4] = score;
                uni_vmovdqu(vmm_proposals_mask, vmm_anchor_mask_addr);
                uni_vpaddd(vmm_proposals_idx, vmm_proposals_idx, vmm_proposals_idx_offset);
                uni_vscatterdps(reg_proposals_ptr, vmm_proposals_idx, sizeof(float), vmm_score, vmm_proposals_mask);
            }

            /** @code
                // recompute new width & height
                const float box_w = x1 - x0 + coordinates_offset;
                const float box_h = y1 - y0 + coordinates_offset;
             */
            Vmm vmm_box_w = this->get_free_vmm<Vmm>(not_available_vmm);
            Vmm vmm_box_h = this->get_free_vmm<Vmm>(not_available_vmm);
            Vmm vmm_min_box_w = this->get_free_vmm<Vmm>(not_available_vmm);
            Vmm vmm_min_box_h = this->get_free_vmm<Vmm>(not_available_vmm);

            // const float box_w = x1 - x0 + coordinates_offset;
            uni_vsubps(vmm_box_w, vmm_x1, vmm_x0);
            uni_vaddps(vmm_box_w, vmm_box_w, vmm_coordinates_offset_addr);
            // const float box_h = y1 - y0 + coordinates_offset;
            uni_vsubps(vmm_box_h, vmm_y1, vmm_y0);
            uni_vaddps(vmm_box_h, vmm_box_h, vmm_coordinates_offset_addr);

            /** @code
                int p_idx = proposal_idx(h, w, anchor, 0);
                ...
                proposals[p_idx + 5] = (min_box_w <= box_w) * (min_box_h <= box_h) * 1.0;
             */
            uni_vbroadcastss(vmm_min_box_w, ptr[reg_params + offsetof(jit_refine_anchors_call_args, min_box_w)]);
            uni_vbroadcastss(vmm_min_box_h, ptr[reg_params + offsetof(jit_refine_anchors_call_args, min_box_h)]);
            uni_vcmpps(vmm_box_w, vmm_min_box_w, vmm_box_w, VCMPPS_LE);
            uni_vcmpps(vmm_box_h, vmm_min_box_h, vmm_box_h, VCMPPS_LE);
            uni_vpmulld(vmm_box_h, vmm_box_w, vmm_box_h);
            uni_vcvtdq2ps(vmm_box_h, vmm_box_h);

            {
                // proposals[p_idx + 5] = (min_box_w <= box_w) * (min_box_h <= box_h) * 1.0;
                uni_vmovdqu(vmm_proposals_mask, vmm_anchor_mask_addr);
                uni_vpaddd(vmm_proposals_idx, vmm_proposals_idx, vmm_proposals_idx_offset);
                uni_vscatterdps(reg_proposals_ptr, vmm_proposals_idx, sizeof(float), vmm_box_h, vmm_proposals_mask);
            }

            this->update_input_output_ptrs();

            // Free space for mask
            sub(reg_anchors_loop, reg_anchors_chunk);
        }
        ja(anchor_loop);
    }

    this->postamble();

    exp_injector->prepare_table();
}

template <x64::cpu_isa_t isa>
void jit_refine_anchors_kernel_fp32<isa>::update_input_output_ptrs() {
    xor_(reg_num_proc_elem, reg_num_proc_elem);
    mov(reg_num_proc_elem, reg_anchors_chunk);
    imul(reg_num_proc_elem.cvt32(), dword[reg_params + offsetof(jit_refine_anchors_call_args, anchor_anchor_offset)]);
    imul(reg_num_proc_elem, reg_num_proc_elem, sizeof(float));
    add(reg_anchors_ptr, reg_num_proc_elem);

    xor_(reg_num_proc_elem, reg_num_proc_elem);
    mov(reg_num_proc_elem, reg_anchors_chunk);
    imul(reg_num_proc_elem.cvt32(), dword[reg_params + offsetof(jit_refine_anchors_call_args, delta_anchor_offset)]);
    imul(reg_num_proc_elem, reg_num_proc_elem, sizeof(float));
    add(reg_deltas_ptr, reg_num_proc_elem);

    xor_(reg_num_proc_elem, reg_num_proc_elem);
    mov(reg_num_proc_elem, reg_anchors_chunk);
    imul(reg_num_proc_elem.cvt32(), dword[reg_params + offsetof(jit_refine_anchors_call_args, score_anchor_offset)]);
    imul(reg_num_proc_elem, reg_num_proc_elem, sizeof(float));
    add(reg_scores_ptr, reg_num_proc_elem);

    xor_(reg_num_proc_elem, reg_num_proc_elem);
    mov(reg_num_proc_elem, reg_anchors_chunk);
    imul(reg_num_proc_elem.cvt32(), dword[reg_params + offsetof(jit_refine_anchors_call_args, proposal_anchor_offset)]);
    imul(reg_num_proc_elem, reg_num_proc_elem, sizeof(float));
    add(reg_proposals_ptr, reg_num_proc_elem);
}

template struct jit_refine_anchors_kernel_fp32<x64::avx512_core>;
template struct jit_refine_anchors_kernel_fp32<x64::avx2>;
template struct jit_refine_anchors_kernel_fp32<x64::sse41>;

}
}
