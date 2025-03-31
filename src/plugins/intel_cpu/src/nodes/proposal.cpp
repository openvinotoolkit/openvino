// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/proposal.hpp"

#include <string>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "proposal.h"

namespace ov::intel_cpu::node {

static std::vector<float> generate_anchors(proposal_conf& conf) {
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
    const auto base_area = static_cast<float>(base_size * base_size);
    const float half_base_size = base_size * 0.5f;
    const float center = 0.5f * (base_size - coordinates_offset);

    // enumerate all transformed boxes
    for (size_t ratio = 0; ratio < num_ratios; ++ratio) {
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

        float* const p_anchors_wm = anchors_ptr + 0 * num_ratios * num_scales + ratio * num_scales;
        float* const p_anchors_hm = anchors_ptr + 1 * num_ratios * num_scales + ratio * num_scales;
        float* const p_anchors_wp = anchors_ptr + 2 * num_ratios * num_scales + ratio * num_scales;
        float* const p_anchors_hp = anchors_ptr + 3 * num_ratios * num_scales + ratio * num_scales;

        for (size_t scale = 0; scale < num_scales; ++scale) {
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

bool Proposal::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto proposal0Op = ov::as_type_ptr<const ov::op::v0::Proposal>(op);
        const auto proposal4Op = ov::as_type_ptr<const ov::op::v4::Proposal>(op);
        if (!proposal0Op && !proposal4Op) {
            errorMessage = "Node is not an instance of the Proposal from the operations set v0 or v4.";
            return false;
        }
        auto proposalOp = ov::as_type_ptr<const ov::op::v0::Proposal>(op);
        if (proposalOp->get_attrs().framework != "tensorflow" && !proposalOp->get_attrs().framework.empty()) {
            errorMessage = "Unsupported framework attribute: " + proposalOp->get_attrs().framework;
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Proposal::Proposal(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    auto proposalOp = ov::as_type_ptr<const ov::op::v0::Proposal>(op);
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
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    if (store_prob) {
        addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32},
                              {LayoutType::ncsp, ov::element::f32},
                              {LayoutType::ncsp, ov::element::f32}},
                             {{LayoutType::ncsp, ov::element::f32}, {LayoutType::ncsp, ov::element::f32}},
                             impl_desc_type::ref_any);
    } else {
        addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32},
                              {LayoutType::ncsp, ov::element::f32},
                              {LayoutType::ncsp, ov::element::f32}},
                             {{LayoutType::ncsp, ov::element::f32}},
                             impl_desc_type::ref_any);
    }
}

void Proposal::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void Proposal::execute(const dnnl::stream& strm) {
    try {
        const auto* probabilitiesData = getSrcDataAtPortAs<const float>(PROBABILITIES_IN_IDX);
        const auto* anchorsData = getSrcDataAtPortAs<const float>(ANCHORS_IN_IDX);
        const auto* imgInfoData = getSrcDataAtPortAs<const float>(IMG_INFO_IN_IDX);
        auto* outRoiData = reinterpret_cast<float*>(getDstDataAtPort(ROI_OUT_IDX));
        float* outProbData = nullptr;
        if (store_prob) {
            outProbData = reinterpret_cast<float*>(getDstDataAtPort(PROBABILITIES_OUT_IDX));
        }

        auto inProbDims = getParentEdgeAt(0)->getMemory().getStaticDims();
        const size_t imgInfoSize = getParentEdgeAt(2)->getMemory().getStaticDims()[0];

        // input image height & width
        const float imgHeight = imgInfoData[0];
        const float imgWidth = imgInfoData[1];
        if (!std::isnormal(imgHeight) || !std::isnormal(imgWidth) || (imgHeight < 0.f) || (imgWidth < 0.f)) {
            THROW_CPU_NODE_ERR("image info input must have positive image height and width.");
        }

        // scale factor for height & width
        const float scaleHeight = imgInfoData[2];
        const float scaleWidth = imgInfoSize == 4 ? imgInfoData[3] : scaleHeight;
        if (!std::isfinite(scaleHeight) || !std::isfinite(scaleWidth) || (scaleHeight < 0.f) || (scaleWidth < 0.f)) {
            THROW_CPU_NODE_ERR("image info input must have non negative scales.");
        }

        ov::Extensions::Cpu::XARCH::proposal_exec(probabilitiesData,
                                                  anchorsData,
                                                  inProbDims,
                                                  {imgHeight, imgWidth, scaleHeight, scaleWidth},
                                                  anchors.data(),
                                                  roi_indices.data(),
                                                  outRoiData,
                                                  outProbData,
                                                  conf);
    } catch (const ov::Exception& e) {
        THROW_CPU_NODE_ERR(e.what());
    }
}

bool Proposal::created() const {
    return getType() == Type::Proposal;
}

}  // namespace ov::intel_cpu::node
