// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <string>
#include <cmath>
#include <vector>

#include "common/tensor_desc_creator.h"
#include "proposal_imp.hpp"
#include <ngraph/op/proposal.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

static
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

class ProposalImpl : public ExtLayerBase {
public:
    bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            auto proposal0Op = ngraph::as_type_ptr<const ngraph::op::v0::Proposal>(op);
            auto proposal4Op = ngraph::as_type_ptr<const ngraph::op::v4::Proposal>(op);
            if (!proposal0Op && !proposal4Op) {
                errorMessage = "Node is not an instance of the Proposal from the operations set v0 or v4.";
                return false;
            }
            auto proposalOp = std::dynamic_pointer_cast<const ngraph::op::v0::Proposal>(op);
            // [NM] TODO: Enable after fix Issue: 53750
            // if (proposalOp->get_attrs().framework != "tensorflow" && !proposalOp->get_attrs().framework.empty()) {
            //     errorMessage = "Unsupported framework attribute: " + proposalOp->get_attrs().framework;
            //     return false;
            // }
        } catch (...) {
            return false;
        }
        return true;
    }

    explicit ProposalImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
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
            if (store_prob) {
                addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32},
                               {TensorDescCreatorTypes::ncsp, Precision::FP32},
                               {TensorDescCreatorTypes::ncsp, Precision::FP32}},
                              {{TensorDescCreatorTypes::ncsp, Precision::FP32},
                               {TensorDescCreatorTypes::ncsp, Precision::FP32}});
            } else {
                addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32},
                               {TensorDescCreatorTypes::ncsp, Precision::FP32},
                               {TensorDescCreatorTypes::ncsp, Precision::FP32}},
                              {{TensorDescCreatorTypes::ncsp, Precision::FP32}});
            }
        } catch (const InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
            throw;
        }
    }

    StatusCode execute(std::vector<Blob::Ptr> &inputs, std::vector<Blob::Ptr> &outputs,
                       ResponseDesc *resp) noexcept override {
        try {
            const float* probabilitiesData = inputs[PROBABILITIES_IN_IDX]->cbuffer().as<const float*>() +
                inputs[PROBABILITIES_IN_IDX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            const float* anchorsData = inputs[ANCHORS_IN_IDX]->cbuffer().as<const float*>() +
                inputs[ANCHORS_IN_IDX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            const float* imgInfoData = inputs[IMG_INFO_IN_IDX]->cbuffer().as<const float*>() +
                inputs[IMG_INFO_IN_IDX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            float* outRoiData = outputs[ROI_OUT_IDX]->buffer().as<float*>() +
                outputs[ROI_OUT_IDX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            float* outProbData = nullptr;
            if (store_prob)
                outProbData = outputs[PROBABILITIES_OUT_IDX]->buffer().as<float*>() +
                    outputs[PROBABILITIES_OUT_IDX]->getTensorDesc().getBlockingDesc().getOffsetPadding();

            auto inProbDims = inputs[0]->getTensorDesc().getDims();
            const size_t imgInfoSize = inputs[2]->getTensorDesc().getDims()[0];

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

            XARCH::proposal_exec(probabilitiesData, anchorsData, inProbDims,
                    {imgHeight, imgWidth, scaleHeight, scaleWidth}, anchors.data(), roi_indices.data(), outRoiData, outProbData, conf);

            return OK;
        } catch (const InferenceEngine::Exception& e) {
            if (resp) {
                std::string errorMsg = e.what();
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
    }

private:
    const size_t PROBABILITIES_IN_IDX = 0lu;
    const size_t ANCHORS_IN_IDX = 1lu;
    const size_t IMG_INFO_IN_IDX = 2lu;
    const size_t ROI_OUT_IDX = 0lu;
    const size_t PROBABILITIES_OUT_IDX = 1lu;

    proposal_conf conf;
    std::vector<float> anchors;
    std::vector<int> roi_indices;
    bool store_prob;  // store blob with proposal probabilities
};

REG_FACTORY_FOR(ProposalImpl, Proposal);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
