// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include <ngraph/op/proposal.hpp>
#include "ie_parallel.hpp"
#include "proposal.h"

using namespace InferenceEngine;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

std::vector<float> generate_anchors(proposal_conf &conf) {
    auto base_size = conf.base_size_;
    auto coordinates_offset = conf.coordinates_offset;
    auto round_ratios = conf.round_ratios;

    auto num_ratios = conf.ratios.size();
    auto ratios = conf.ratios.data();

    auto num_scales = conf.scales.size();
    auto scales = conf.scales.data();

    std::vector<float> anchors(num_scales * num_ratios * 4);
    auto anchors_ptr = anchors.data();

    // base box's width & height & center location
    const float base_area = static_cast<float>(base_size * base_size);
    const float half_base_size = base_size * 0.5f;
    const float center = 0.5f * (base_size - coordinates_offset);

    // enumerate all transformed boxes
    for (int ratio = 0; ratio < num_ratios; ++ratio) {
        // transformed width & height for given ratio factors
        float ratio_w;
        float ratio_h;
        if (round_ratios) {
            ratio_w = std::roundf(std::sqrt(base_area / ratios[ratio]));
            ratio_h = std::roundf(ratio_w * ratios[ratio]);
        } else {
            ratio_w = std::sqrt(base_area / ratios[ratio]);
            ratio_h = ratio_w * ratios[ratio];
        }

        float * const p_anchors_wm = anchors_ptr + 0 * num_ratios * num_scales + ratio * num_scales;
        float * const p_anchors_hm = anchors_ptr + 1 * num_ratios * num_scales + ratio * num_scales;
        float * const p_anchors_wp = anchors_ptr + 2 * num_ratios * num_scales + ratio * num_scales;
        float * const p_anchors_hp = anchors_ptr + 3 * num_ratios * num_scales + ratio * num_scales;

        for (int scale = 0; scale < num_scales; ++scale) {
            // transformed width & height for given scale factors
            const float scale_w = 0.5f * (ratio_w * scales[scale] - coordinates_offset);
            const float scale_h = 0.5f * (ratio_h * scales[scale] - coordinates_offset);

            // (x1, y1, x2, y2) for transformed box
            p_anchors_wm[scale] = center - scale_w;
            p_anchors_hm[scale] = center - scale_h;
            p_anchors_wp[scale] = center + scale_w;
            p_anchors_hp[scale] = center + scale_h;

            if (conf.shift_anchors) {
                p_anchors_wm[scale] -= half_base_size;
                p_anchors_hm[scale] -= half_base_size;
                p_anchors_wp[scale] -= half_base_size;
                p_anchors_hp[scale] -= half_base_size;
            }
        }
    }
    return anchors;
}

void enumerate_proposals_cpu(const float* bottom4d, const float* d_anchor4d, const float* anchors,
                             float* proposals, const int num_anchors, const int bottom_H,
                             const int bottom_W, const float img_H, const float img_W,
                             const float min_box_H, const float min_box_W, const int feat_stride,
                             const float box_coordinate_scale, const float box_size_scale,
                             float coordinates_offset, bool initial_clip, bool swap_xy, bool clip_before_nms) {
    const int bottom_area = bottom_H * bottom_W;

    const float* p_anchors_wm = anchors + 0 * num_anchors;
    const float* p_anchors_hm = anchors + 1 * num_anchors;
    const float* p_anchors_wp = anchors + 2 * num_anchors;
    const float* p_anchors_hp = anchors + 3 * num_anchors;

    parallel_for2d(bottom_H, bottom_W, [&](size_t h, size_t w) {
        const float x = static_cast<float>((swap_xy ? h : w) * feat_stride);
        const float y = static_cast<float>((swap_xy ? w : h) * feat_stride);

        const float* p_box   = d_anchor4d + h * bottom_W + w;
        const float* p_score = bottom4d   + h * bottom_W + w;

        float* p_proposal = proposals + (h * bottom_W + w) * num_anchors * 5;

        for (int anchor = 0; anchor < num_anchors; ++anchor) {
            const float dx = p_box[(anchor * 4 + 0) * bottom_area] / box_coordinate_scale;
            const float dy = p_box[(anchor * 4 + 1) * bottom_area] / box_coordinate_scale;

            const float d_log_w = p_box[(anchor * 4 + 2) * bottom_area] / box_size_scale;
            const float d_log_h = p_box[(anchor * 4 + 3) * bottom_area] / box_size_scale;

            const float score = p_score[anchor * bottom_area];

            float x0 = x + p_anchors_wm[anchor];
            float y0 = y + p_anchors_hm[anchor];
            float x1 = x + p_anchors_wp[anchor];
            float y1 = y + p_anchors_hp[anchor];

            if (initial_clip) {
                // adjust new corner locations to be within the image region
                x0 = std::max<float>(0.0f, std::min<float>(x0, img_W));
                y0 = std::max<float>(0.0f, std::min<float>(y0, img_H));
                x1 = std::max<float>(0.0f, std::min<float>(x1, img_W));
                y1 = std::max<float>(0.0f, std::min<float>(y1, img_H));
            }

            // width & height of box
            const float ww = x1 - x0 + coordinates_offset;
            const float hh = y1 - y0 + coordinates_offset;
            // center location of box
            const float ctr_x = x0 + 0.5f * ww;
            const float ctr_y = y0 + 0.5f * hh;

            // new center location according to gradient (dx, dy)
            const float pred_ctr_x = dx * ww + ctr_x;
            const float pred_ctr_y = dy * hh + ctr_y;
            // new width & height according to gradient d(log w), d(log h)
            const float pred_w = std::exp(d_log_w) * ww;
            const float pred_h = std::exp(d_log_h) * hh;

            // update upper-left corner location
            x0 = pred_ctr_x - 0.5f * pred_w;
            y0 = pred_ctr_y - 0.5f * pred_h;
            // update lower-right corner location
            x1 = pred_ctr_x + 0.5f * pred_w;
            y1 = pred_ctr_y + 0.5f * pred_h;

            // adjust new corner locations to be within the image region,
            if (clip_before_nms) {
                x0 = std::max<float>(0.0f, std::min<float>(x0, img_W - coordinates_offset));
                y0 = std::max<float>(0.0f, std::min<float>(y0, img_H - coordinates_offset));
                x1 = std::max<float>(0.0f, std::min<float>(x1, img_W - coordinates_offset));
                y1 = std::max<float>(0.0f, std::min<float>(y1, img_H - coordinates_offset));
            }

            // recompute new width & height
            const float box_w = x1 - x0 + coordinates_offset;
            const float box_h = y1 - y0 + coordinates_offset;

            p_proposal[5*anchor + 0] = x0;
            p_proposal[5*anchor + 1] = y0;
            p_proposal[5*anchor + 2] = x1;
            p_proposal[5*anchor + 3] = y1;
            p_proposal[5*anchor + 4] = (min_box_W <= box_w) * (min_box_H <= box_h) * score;
        }
    });
}

void unpack_boxes(const float* p_proposals, float* unpacked_boxes, int pre_nms_topn, bool store_prob) {
    if (store_prob) {
        parallel_for(pre_nms_topn, [&](size_t i) {
            unpacked_boxes[0 * pre_nms_topn + i] = p_proposals[5 * i + 0];
            unpacked_boxes[1 * pre_nms_topn + i] = p_proposals[5 * i + 1];
            unpacked_boxes[2 * pre_nms_topn + i] = p_proposals[5 * i + 2];
            unpacked_boxes[3 * pre_nms_topn + i] = p_proposals[5 * i + 3];
            unpacked_boxes[4 * pre_nms_topn + i] = p_proposals[5 * i + 4];
        });
    } else {
        parallel_for(pre_nms_topn, [&](size_t i) {
            unpacked_boxes[0 * pre_nms_topn + i] = p_proposals[5 * i + 0];
            unpacked_boxes[1 * pre_nms_topn + i] = p_proposals[5 * i + 1];
            unpacked_boxes[2 * pre_nms_topn + i] = p_proposals[5 * i + 2];
            unpacked_boxes[3 * pre_nms_topn + i] = p_proposals[5 * i + 3];
        });
    }
}

void retrieve_rois_cpu(const int num_rois, const int item_index,
                              const int num_proposals,
                              const float* proposals, const int roi_indices[],
                              float* rois, int post_nms_topn_,
                              bool normalize, float img_h, float img_w, bool clip_after_nms, float* probs) {
    const float *src_x0 = proposals + 0 * num_proposals;
    const float *src_y0 = proposals + 1 * num_proposals;
    const float *src_x1 = proposals + 2 * num_proposals;
    const float *src_y1 = proposals + 3 * num_proposals;
    const float *src_probs = proposals + 4 * num_proposals;

    parallel_for(num_rois, [&](size_t roi) {
        int index = roi_indices[roi];

        float x0 = src_x0[index];
        float y0 = src_y0[index];
        float x1 = src_x1[index];
        float y1 = src_y1[index];

        if (clip_after_nms) {
            x0 = std::max<float>(0.0f, std::min<float>(x0, img_w));
            y0 = std::max<float>(0.0f, std::min<float>(y0, img_h));
            x1 = std::max<float>(0.0f, std::min<float>(x1, img_w));
            y1 = std::max<float>(0.0f, std::min<float>(y1, img_h));
        }

        if (normalize) {
            x0 /= img_w;
            y0 /= img_h;
            x1 /= img_w;
            y1 /= img_h;
        }

        rois[roi * 5 + 0] = static_cast<float>(item_index);
        rois[roi * 5 + 1] = x0;
        rois[roi * 5 + 2] = y0;
        rois[roi * 5 + 3] = x1;
        rois[roi * 5 + 4] = y1;

        if (probs)
            probs[roi] = src_probs[index];
    });

    if (num_rois < post_nms_topn_) {
        for (int i = 5 * num_rois; i < 5 * post_nms_topn_; i++) {
            rois[i] = 0.f;
        }

        // marker at end of boxes list
        rois[num_rois * 5 + 0] = -1;
    }
}

} // anonymous namespace

void nms_cpu(const int num_boxes, int is_dead[],
             const float* boxes, int index_out[], std::size_t* const num_out,
             const int base_index, const float nms_thresh, const int max_num_out,
             float coordinates_offset) {
    const int num_proposals = num_boxes;
    std::size_t count = 0;

    const float* x0 = boxes + 0 * num_proposals;
    const float* y0 = boxes + 1 * num_proposals;
    const float* x1 = boxes + 2 * num_proposals;
    const float* y1 = boxes + 3 * num_proposals;

    for (int box = 0; box < num_boxes; ++box) {
        if (is_dead[box])
            continue;

        index_out[count++] = base_index + box;
        if (count == max_num_out)
            break;

        int tail = box + 1;

        for (; tail < num_boxes; ++tail) {
            float res = 0.0f;

            const float x0i = x0[box];
            const float y0i = y0[box];
            const float x1i = x1[box];
            const float y1i = y1[box];

            const float x0j = x0[tail];
            const float y0j = y0[tail];
            const float x1j = x1[tail];
            const float y1j = y1[tail];

            if (x0i <= x1j && y0i <= y1j && x0j <= x1i && y0j <= y1i) {
                // overlapped region (= box)
                const float x0 = std::max<float>(x0i, x0j);
                const float y0 = std::max<float>(y0i, y0j);
                const float x1 = std::min<float>(x1i, x1j);
                const float y1 = std::min<float>(y1i, y1j);

                // intersection area
                const float width  = std::max<float>(0.0f,  x1 - x0 + coordinates_offset);
                const float height = std::max<float>(0.0f,  y1 - y0 + coordinates_offset);
                const float area   = width * height;

                // area of A, B
                const float A_area = (x1i - x0i + coordinates_offset) * (y1i - y0i + coordinates_offset);
                const float B_area = (x1j - x0j + coordinates_offset) * (y1j - y0j + coordinates_offset);

                // IoU
                res = area / (A_area + B_area - area);
            }

            if (nms_thresh < res)
                is_dead[tail] = 1;
        }
    }

    *num_out = count;
}

void Proposal::createPrimitive() {
    jit_uni_nms_proposal_kernel::jit_nms_conf jcp { conf.post_nms_topn_, conf.nms_thresh_, conf.coordinates_offset };
    std::unique_ptr<jit_uni_nms_proposal_kernel> nms_kernel;
    if (mayiuse(avx512_core)) {
        nms_kernel.reset(new jit_uni_nms_proposal_kernel_impl<avx512_core> { jcp });
    } else if (mayiuse(x64::avx2)) {
        nms_kernel.reset(new jit_uni_nms_proposal_kernel_impl<x64::avx2> { jcp });
    } else if (mayiuse(sse41)) {
        nms_kernel.reset(new jit_uni_nms_proposal_kernel_impl<sse41> { jcp });
    } else {
        DEBUG_LOG("Unable to create JIT version of Proposal due to unsupported ISA."
            " Non-JIT version of proposal will be executed.");
    }
    if (nms_kernel) {
        nms_kernel->create_ker();
        nms_kernel_ = std::move(nms_kernel);
    }
}

bool Proposal::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto proposal0Op = ngraph::as_type_ptr<const ngraph::op::v0::Proposal>(op);
        const auto proposal4Op = ngraph::as_type_ptr<const ngraph::op::v4::Proposal>(op);
        if (!proposal0Op && !proposal4Op) {
            errorMessage = "Node is not an instance of the Proposal from the operations set v0 or v4.";
            return false;
        }
        auto proposalOp = std::dynamic_pointer_cast<const ngraph::op::v0::Proposal>(op);
        if (proposalOp->get_attrs().framework != "tensorflow" && !proposalOp->get_attrs().framework.empty()) {
            errorMessage = "Unsupported framework attribute: " + proposalOp->get_attrs().framework;
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Proposal::Proposal(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng,
        WeightsSharing::Ptr &cache) : Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto proposalOp = std::dynamic_pointer_cast<const ngraph::op::v0::Proposal>(op);
    auto proposalAttrs = proposalOp->get_attrs();

    conf.feat_stride_ = proposalAttrs.feat_stride;
    conf.base_size_ = proposalAttrs.base_size;
    conf.min_size_ = proposalAttrs.min_size;
    conf.pre_nms_topn_ = proposalAttrs.pre_nms_topn;
    conf.post_nms_topn_ = proposalAttrs.post_nms_topn;
    conf.nms_thresh_ = proposalAttrs.nms_thresh;
    conf.box_coordinate_scale_ = proposalAttrs.box_coordinate_scale;
    conf.box_size_scale_ = proposalAttrs.box_size_scale;
    conf.scales = proposalAttrs.scale;
    conf.ratios = proposalAttrs.ratio;
    conf.normalize_ = proposalAttrs.normalize;
    conf.clip_before_nms = proposalAttrs.clip_before_nms;
    conf.clip_after_nms = proposalAttrs.clip_after_nms;
    conf.anchors_shape_0 = conf.ratios.size() * conf.scales.size();

    if (proposalAttrs.framework == "tensorflow") {
        conf.coordinates_offset = 0.0f;
        conf.initial_clip = true;
        conf.shift_anchors = true;
        conf.round_ratios = false;
        conf.swap_xy = true;
    } else {
        conf.coordinates_offset = 1.0f;
        conf.initial_clip = false;
        conf.shift_anchors = false;
        conf.round_ratios = true;
        conf.swap_xy = false;
    }

    anchors = generate_anchors(conf);
    roi_indices.resize(conf.post_nms_topn_);

    store_prob = op->get_output_size() == 2;
}

void Proposal::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    if (store_prob) {
        addSupportedPrimDesc({{LayoutType::ncsp, Precision::FP32},
                              {LayoutType::ncsp, Precision::FP32},
                              {LayoutType::ncsp, Precision::FP32}},
                             {{LayoutType::ncsp, Precision::FP32},
                              {LayoutType::ncsp, Precision::FP32}},
                             impl_desc_type::ref_any);
    } else {
        addSupportedPrimDesc({{LayoutType::ncsp, Precision::FP32},
                              {LayoutType::ncsp, Precision::FP32},
                              {LayoutType::ncsp, Precision::FP32}},
                             {{LayoutType::ncsp, Precision::FP32}},
                             impl_desc_type::ref_any);
    }
}

void Proposal::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Proposal::executeImpl(const float *input0, const float *input1, std::vector<size_t> dims0,
                           std::array<float, 4> img_info, const float *anchors, int *roi_indices, float *output0,
                           float *output1, proposal_conf &conf) {
    // Prepare memory
    const float *p_bottom_item = input0;
    const float *p_d_anchor_item = input1;

    float *p_roi_item = output0;
    float *p_prob_item = output1;
    auto store_prob = p_prob_item != nullptr;

    // bottom shape: (2 x num_anchors) x H x W
    const int bottom_H = dims0[2];
    const int bottom_W = dims0[3];

    // input image height & width
    const float img_H = img_info[conf.swap_xy ? 1 : 0];
    const float img_W = img_info[conf.swap_xy ? 0 : 1];

    // scale factor for height & width
    const float scale_H = img_info[2];
    const float scale_W = img_info[3];

    // minimum box width & height
    const float min_box_H = conf.min_size_ * scale_H;
    const float min_box_W = conf.min_size_ * scale_W;

    // number of all proposals = num_anchors * H * W
    const int num_proposals = conf.anchors_shape_0 * bottom_H * bottom_W;

    // number of top-n proposals before NMS
    const int pre_nms_topn = std::min<int>(num_proposals, conf.pre_nms_topn_);

    // number of final RoIs
    std::size_t num_rois = 0;

    // enumerate all proposals
    //   num_proposals = num_anchors * H * W
    //   (x1, y1, x2, y2, score) for each proposal
    // NOTE: for bottom, only foreground scores are passed
    struct ProposalBox {
        float x0;
        float y0;
        float x1;
        float y1;
        float score;
    };
    std::vector<ProposalBox> proposals_(num_proposals);
    const int unpacked_boxes_buffer_size = store_prob ? 5 * pre_nms_topn : 4 * pre_nms_topn;
    std::vector<float> unpacked_boxes(unpacked_boxes_buffer_size);
    std::vector<int> is_dead(pre_nms_topn, 0);

    // Execute
    int nn = dims0[0];
    for (int n = 0; n < nn; ++n) {
        enumerate_proposals_cpu(p_bottom_item + num_proposals + n * num_proposals * 2,
                                p_d_anchor_item + n * num_proposals * 4,
                                anchors, reinterpret_cast<float *>(&proposals_[0]),
                                conf.anchors_shape_0, bottom_H, bottom_W, img_H, img_W,
                                min_box_H, min_box_W, conf.feat_stride_,
                                conf.box_coordinate_scale_, conf.box_size_scale_,
                                conf.coordinates_offset, conf.initial_clip, conf.swap_xy, conf.clip_before_nms);
        std::partial_sort(proposals_.begin(), proposals_.begin() + pre_nms_topn, proposals_.end(),
                          [](const ProposalBox &struct1, const ProposalBox &struct2) {
                              return (struct1.score > struct2.score);
                          });

        unpack_boxes(reinterpret_cast<float *>(&proposals_[0]), &unpacked_boxes[0], pre_nms_topn, store_prob);
        if (n > 0)
            std::fill(is_dead.begin(), is_dead.end(), 0);
#ifdef __GNUC__
        if (__builtin_expect(static_cast<bool>(nms_kernel_), true)) {
#else
        if (nms_kernel_) {
#endif
            jit_uni_nms_proposal_kernel::jit_nms_call_args args {
                pre_nms_topn,
                is_dead.data(),
                unpacked_boxes.data(),
                &unpacked_boxes[2 * pre_nms_topn],
                &unpacked_boxes[pre_nms_topn],
                &unpacked_boxes[3 * pre_nms_topn],
                roi_indices,
                &num_rois
            };
            nms_kernel_->operator()(&args);
        } else {
            nms_cpu(pre_nms_topn, &is_dead[0], &unpacked_boxes[0], roi_indices, &num_rois, 0, conf.nms_thresh_,
                conf.post_nms_topn_, conf.coordinates_offset);
        }

        float* p_probs = store_prob ? p_prob_item + n * conf.post_nms_topn_ : nullptr;
        retrieve_rois_cpu(num_rois, n, pre_nms_topn, &unpacked_boxes[0], roi_indices,
                          p_roi_item + n * conf.post_nms_topn_ * 5,
                          conf.post_nms_topn_, conf.normalize_, img_H, img_W, conf.clip_after_nms, p_probs);
    }
}

void Proposal::execute(dnnl::stream strm) {
    try {
        const float* probabilitiesData = reinterpret_cast<const float *>(getParentEdgeAt(PROBABILITIES_IN_IDX)->getMemoryPtr()->GetPtr());
        const float* anchorsData = reinterpret_cast<const float *>(getParentEdgeAt(ANCHORS_IN_IDX)->getMemoryPtr()->GetPtr());
        const float* imgInfoData = reinterpret_cast<const float *>(getParentEdgeAt(IMG_INFO_IN_IDX)->getMemoryPtr()->GetPtr());
        float* outRoiData = reinterpret_cast <float *>(getChildEdgesAtPort(ROI_OUT_IDX)[0]->getMemoryPtr()->GetPtr());
        float* outProbData = nullptr;
        if (store_prob)
            outProbData = reinterpret_cast <float *>(getChildEdgesAtPort(PROBABILITIES_OUT_IDX)[0]->getMemoryPtr()->GetPtr());

        auto inProbDims = getParentEdgeAt(0)->getMemory().getStaticDims();
        const size_t imgInfoSize = getParentEdgeAt(2)->getMemory().getStaticDims()[0];

        // input image height & width
        const float imgHeight = imgInfoData[0];
        const float imgWidth = imgInfoData[1];
        if (!std::isnormal(imgHeight) || !std::isnormal(imgWidth) || (imgHeight < 0.f) || (imgWidth < 0.f)) {
            IE_THROW() << "Proposal operation image info input must have positive image height and width.";
        }

        // scale factor for height & width
        const float scaleHeight = imgInfoData[2];
        const float scaleWidth = imgInfoSize == 4 ? imgInfoData[3] : scaleHeight;
        if (!std::isfinite(scaleHeight) || !std::isfinite(scaleWidth) || (scaleHeight < 0.f) || (scaleWidth < 0.f)) {
            IE_THROW() << "Proposal operation image info input must have non negative scales.";
        }

        executeImpl(probabilitiesData, anchorsData, inProbDims, { imgHeight, imgWidth, scaleHeight, scaleWidth },
            anchors.data(), roi_indices.data(), outRoiData, outProbData, conf);
    } catch (const InferenceEngine::Exception& e) {
        std::string errorMsg = e.what();
        IE_THROW() << errorMsg;
    }
}

bool Proposal::created() const {
    return getType() == Type::Proposal;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
