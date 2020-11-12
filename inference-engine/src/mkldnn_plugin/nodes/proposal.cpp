// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include "proposal_imp.hpp"
#include <string>
#include <cmath>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

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
    explicit ProposalImpl(const CNNLayer *layer) {
        try {
            if (layer->insData.size() != 3 || (layer->outData.size() != 1 && layer->outData.size() != 2))
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            if (layer->insData[0].lock()->getTensorDesc().getDims().size() != 4)
                THROW_IE_EXCEPTION << "Proposal supports only 4D blobs!";

            conf.feat_stride_ = static_cast<size_t>(layer->GetParamAsInt("feat_stride"));
            conf.base_size_ = static_cast<size_t>(layer->GetParamAsInt("base_size"));
            conf.min_size_ = static_cast<size_t>(layer->GetParamAsInt("min_size"));
            conf.pre_nms_topn_ = layer->GetParamAsInt("pre_nms_topn");
            conf.post_nms_topn_ = layer->GetParamAsInt("post_nms_topn");
            conf.nms_thresh_ = layer->GetParamAsFloat("nms_thresh");
            conf.box_coordinate_scale_ = layer->GetParamAsFloat("box_coordinate_scale", 1.0);
            conf.box_size_scale_ = layer->GetParamAsFloat("box_size_scale", 1.0);
            conf.scales = layer->GetParamAsFloats("scale", {});
            conf.ratios = layer->GetParamAsFloats("ratio", {});
            conf.normalize_ = layer->GetParamAsBool("normalize", false);
            conf.clip_before_nms = layer->GetParamAsBool("clip_before_nms", true);
            conf.clip_after_nms = layer->GetParamAsBool("clip_after_nms", false);

            conf.anchors_shape_0 = conf.ratios.size() * conf.scales.size();

            std::string framework_ = layer->GetParamAsString("framework", "");
            if (framework_ == "tensorflow") {
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

            store_prob = layer->outData.size() == 2;
            if (store_prob) {
                addConfig(layer, {DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)},
                                 {DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)});
            } else {
                addConfig(layer, {DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)},
                                 {DataConfigurator(ConfLayout::PLN)});
            }
        } catch (const InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr> &inputs, std::vector<Blob::Ptr> &outputs,
                       ResponseDesc *resp) noexcept override {
        try {
            if (inputs.size() != 3 || outputs.empty()) {
                THROW_IE_EXCEPTION << "Incorrect number of input or output edges!";
            }

            // Prepare memory
            const float *p_bottom_item = inputs[0]->buffer();
            const float *p_d_anchor_item = inputs[1]->buffer();
            const float *p_img_info_cpu = inputs[2]->buffer();
            float *p_roi_item = outputs[0]->buffer();
            float *p_prob_item = nullptr;
            if (store_prob)
                p_prob_item = outputs[1]->buffer();

            auto dims0 = inputs[0]->getTensorDesc().getDims();
            auto img_info_dims = inputs[2]->getTensorDesc().getDims();
            if (img_info_dims.size() != 2)
                THROW_IE_EXCEPTION << "Size of im_info tensor for Proposal is incorrect! Size of im_info must be 2. "
                                   << "Now im_info size is " << img_info_dims.size() << ".";

            if (img_info_dims[1] != 3 && img_info_dims[1] != 4)
                THROW_IE_EXCEPTION << "Shape of im_info tensor for Proposal is incorrect! "
                                   << "Shape of im_info must be of  [1, 3] or [1, 4]! "
                                   << "Now shape of im_info is" << img_info_dims[0] << ", " << img_info_dims[1] << "].";

            size_t img_info_size = img_info_dims[1];


            // input image height & width
            const float img_H = p_img_info_cpu[0];
            const float img_W = p_img_info_cpu[1];
            if (!std::isnormal(img_H) || !std::isnormal(img_W) || (img_H < 0.f) || (img_W < 0.f)) {
                THROW_IE_EXCEPTION << "Proposal operation image info input must have positive image height and width.";
            }

            // scale factor for height & width
            const float scale_H = p_img_info_cpu[2];
            const float scale_W = img_info_size == 4 ? p_img_info_cpu[3] : scale_H;
            if (!std::isfinite(scale_H) || !std::isfinite(scale_W) || (scale_H < 0.f) || (scale_W < 0.f)) {
                THROW_IE_EXCEPTION << "Proposal operation image info input must have non negative scales.";
            }

            XARCH::proposal_exec(p_bottom_item, p_d_anchor_item, dims0,
                    {img_H, img_W, scale_H, scale_W}, anchors.data(), roi_indices.data(), p_roi_item, p_prob_item, conf);

            return OK;
        } catch (const InferenceEngine::details::InferenceEngineException& e) {
            if (resp) {
                std::string errorMsg = e.what();
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
    }

private:
    proposal_conf conf;
    std::vector<float> anchors;
    std::vector<int> roi_indices;
    bool store_prob;  // store blob with proposal probabilities
};

REG_FACTORY_FOR(ProposalImpl, Proposal);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
