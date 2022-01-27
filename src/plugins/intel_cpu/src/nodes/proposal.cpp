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

static std::vector<float> generate_anchors(proposal_conf &conf) {
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
