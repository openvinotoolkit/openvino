// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

struct simpler_nms_roi_t {
    float x0, y0, x1, y1;

    static inline float clamp_v(const float v, const float v_min, const float v_max) {
        return std::max(v_min, std::min(v, v_max));
    }

    float area() const { return std::max<float>(0, y1 - y0 + 1) * std::max<float>(0, x1 - x0 + 1); }

    simpler_nms_roi_t intersect(simpler_nms_roi_t other) const {
        return {
            std::max(x0, other.x0),
            std::max(y0, other.y0),
            std::min(x1, other.x1),
            std::min(y1, other.y1)
        };
    }
    simpler_nms_roi_t clamp(simpler_nms_roi_t other) const {
        return {
            clamp_v(x0, other.x0, other.x1),
            clamp_v(y0, other.y0, other.y1),
            clamp_v(x1, other.x0, other.x1),
            clamp_v(y1, other.y0, other.y1)
        };
    }
};

struct simpler_nms_delta_t { float shift_x, shift_y, log_w, log_h; };
struct simpler_nms_proposal_t { simpler_nms_roi_t roi; float confidence; size_t ord; };
struct simpler_nms_anchor { float start_x; float start_y; float end_x; float end_y; };


static void CalcBasicParams(const simpler_nms_anchor& base_anchor,
        float& width, float& height, float& x_center, float& y_center) {
    width  = base_anchor.end_x - base_anchor.start_x + 1.0f;
    height = base_anchor.end_y - base_anchor.start_y + 1.0f;

    x_center = base_anchor.start_x + 0.5f * (width - 1.0f);
    y_center = base_anchor.start_y + 0.5f * (height - 1.0f);
}


static void MakeAnchors(const std::vector<float>& ws, const std::vector<float>& hs,
                        float x_center, float y_center, std::vector<simpler_nms_anchor>& anchors) {
    unsigned int len = ws.size();
    anchors.clear();
    anchors.resize(len);

    for (unsigned int i = 0 ; i < len ; i++) {
        // transpose to create the anchor
        anchors[i].start_x = x_center - 0.5f * (ws[i] - 1.0f);
        anchors[i].start_y = y_center - 0.5f * (hs[i] - 1.0f);
        anchors[i].end_x   = x_center + 0.5f * (ws[i] - 1.0f);
        anchors[i].end_y   = y_center + 0.5f * (hs[i] - 1.0f);
    }
}


static void CalcAnchors(const simpler_nms_anchor& base_anchor, const std::vector<float>& scales,
                        std::vector<simpler_nms_anchor>& anchors) {
    float width = 0.0f, height = 0.0f, x_center = 0.0f, y_center = 0.0f;

    CalcBasicParams(base_anchor, width, height, x_center, y_center);

    unsigned int num_scales = scales.size();
    std::vector<float> ws(num_scales), hs(num_scales);

    for (unsigned int i = 0 ; i < num_scales ; i++) {
        ws[i] = width * scales[i];
        hs[i] = height * scales[i];
    }

    MakeAnchors(ws, hs, x_center, y_center, anchors);
}


static void CalcRatioAnchors(const simpler_nms_anchor& base_anchor, const std::vector<float>& ratios,
                             std::vector<simpler_nms_anchor>& ratio_anchors) {
    float width = 0.0f, height = 0.0f, x_center = 0.0f, y_center = 0.0f;

    CalcBasicParams(base_anchor, width, height, x_center, y_center);

    float size = width * height;

    unsigned int num_ratios = ratios.size();

    std::vector<float> ws(num_ratios), hs(num_ratios);

    for (unsigned int i = 0 ; i < num_ratios ; i++) {
        float new_size = size / ratios[i];
        ws[i] = round(sqrt(new_size));
        hs[i] = round(ws[i] * ratios[i]);
    }

    MakeAnchors(ws, hs, x_center, y_center, ratio_anchors);
}

void GenerateAnchors(unsigned int base_size, const std::vector<float>& ratios,
        const std::vector<float> scales, simpler_nms_anchor *anchors) {
    float end = static_cast<float>(base_size - 1);  // because we start at zero

    simpler_nms_anchor base_anchor = {0.0f, 0.0f, end, end};

    std::vector<simpler_nms_anchor> ratio_anchors;
    CalcRatioAnchors(base_anchor, ratios, ratio_anchors);

    for (size_t i = 0, index = 0; i < ratio_anchors.size() ; i++) {
        std::vector<simpler_nms_anchor> temp_anchors;
        CalcAnchors(ratio_anchors[i], scales, temp_anchors);

        for (size_t j = 0 ; j < temp_anchors.size() ; j++) {
            anchors[index++] = temp_anchors[j];
        }
    }
}

std::vector<simpler_nms_roi_t> simpler_nms_perform_nms(
        const std::vector<simpler_nms_proposal_t>& proposals,
        float iou_threshold,
        size_t top_n) {
    std::vector<simpler_nms_roi_t> res;
    res.reserve(top_n);
    for (const auto & prop : proposals) {
        const auto bbox = prop.roi;
        const float area = bbox.area();

        // For any realistic WL, this condition is true for all top_n values anyway
        if (prop.confidence > 0) {
            bool overlaps = std::any_of(res.begin(), res.end(), [&](const simpler_nms_roi_t& res_bbox) {
                float interArea = bbox.intersect(res_bbox).area();
                float unionArea = res_bbox.area() + area - interArea;
                return interArea > iou_threshold * unionArea;
            });

            if (!overlaps) {
                res.push_back(bbox);
                if (res.size() == top_n) break;
            }
        }
    }

    return res;
}

inline void sort_and_keep_at_most_top_n(
        std::vector<simpler_nms_proposal_t>& proposals,
        size_t top_n) {
    const auto cmp_fn = [](const simpler_nms_proposal_t& a,
                           const simpler_nms_proposal_t& b) {
        return a.confidence > b.confidence || (a.confidence == b.confidence && a.ord > b.ord);
    };

    if (proposals.size() > top_n) {
        std::partial_sort(proposals.begin(), proposals.begin() + top_n, proposals.end(), cmp_fn);
        proposals.resize(top_n);
    } else {
        std::sort(proposals.begin(), proposals.end(), cmp_fn);
    }
}

inline simpler_nms_roi_t simpler_nms_gen_bbox(
        const simpler_nms_anchor& box,
        const simpler_nms_delta_t& delta,
        int anchor_shift_x,
        int anchor_shift_y) {
    auto anchor_w = box.end_x - box.start_x + 1;
    auto anchor_h = box.end_y - box.start_y + 1;
    auto center_x = box.start_x + anchor_w * .5f;
    auto center_y = box.start_y + anchor_h *.5f;

    float pred_center_x = delta.shift_x * anchor_w + center_x + anchor_shift_x;
    float pred_center_y = delta.shift_y * anchor_h + center_y + anchor_shift_y;
    float half_pred_w = exp(delta.log_w) * anchor_w * .5f;
    float half_pred_h = exp(delta.log_h) * anchor_h * .5f;

    return { pred_center_x - half_pred_w,
             pred_center_y - half_pred_h,
             pred_center_x + half_pred_w,
             pred_center_y + half_pred_h };
}

class SimplerNMSImpl : public ExtLayerBase {
public:
    explicit SimplerNMSImpl(const CNNLayer *layer) {
        try {
            if (layer->insData.size() != 3 || layer->outData.size() != 1)
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            if (layer->insData[0].lock()->getTensorDesc().getDims().size() != 4)
                THROW_IE_EXCEPTION << "SimplerNMS supports only 4D blobs!";

            min_box_size_ = layer->GetParamAsInt("min_bbox_size");
            feat_stride_ = layer->GetParamAsInt("feat_stride");
            pre_nms_topn_ = layer->GetParamAsInt("pre_nms_topn");
            post_nms_topn_ = layer->GetParamAsInt("post_nms_topn");
            iou_threshold_ = layer->GetParamAsFloat("iou_threshold");
            scales = layer->GetParamAsFloats("scale", {});

            unsigned int default_size = 16;

            ratios = {0.5f, 1.0f, 2.0f};

            anchors_.resize(ratios.size() * scales.size());
            simpler_nms_anchor *anchors = &anchors_[0];

            GenerateAnchors(default_size, ratios, scales, anchors);

            // Fill config information
            if (layer->outData[0]->getTensorDesc().getDims().size() != 2 ||
                    layer->insData[0].lock()->getTensorDesc().getDims().size() != 4)
                THROW_IE_EXCEPTION << "Unsupported dimensions!";

            addConfig(layer, {DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)},
                      {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr> &inputs, std::vector<Blob::Ptr> &outputs,
                       ResponseDesc *resp) noexcept override {
        int cls_idx = 0;
        int delta_idx = 1;

        Blob::Ptr src_cls = inputs[cls_idx];
        Blob::Ptr src_delta = inputs[delta_idx];

        if (src_cls->getTensorDesc().getDims()[1] > src_delta->getTensorDesc().getDims()[1]) {
            cls_idx = 1;
            delta_idx = 0;

            src_cls = inputs[cls_idx];
            src_delta = inputs[delta_idx];
        }

        int anchors_num = 3 * 3;
        const auto * anchors = (const simpler_nms_anchor*)&anchors_[0];

        int H = src_cls->getTensorDesc().getDims()[2];
        int W = src_cls->getTensorDesc().getDims()[3];

        int SZ = H * W;

        float *dst = outputs[0]->buffer().as<float*>();

        const float* cls_scores = src_cls->buffer().as<const float*>();
        const float* delta_pred = src_delta->buffer().as<const float*>();
        const float* im_info = inputs[2]->buffer().as<const float*>();

        int IW = static_cast<int>(im_info[1]);
        int IH = static_cast<int>(im_info[0]);
        int IS = static_cast<int>(im_info[2]);

        int scaled_min_bbox_size = min_box_size_ * IS;

        std::vector<simpler_nms_proposal_t> sorted_proposals_confidence;

        for (auto y = 0; y < H; ++y) {
            int anchor_shift_y = y * feat_stride_;

            for (auto x = 0; x < W; ++x) {
                int anchor_shift_x = x * feat_stride_;
                int location_index = y * W + x;

                // we assume proposals are grouped by window location
                for (int anchor_index = 0; anchor_index < anchors_num ; anchor_index++) {
                    float dx0 = delta_pred[location_index + SZ * (anchor_index * 4 + 0)];
                    float dy0 = delta_pred[location_index + SZ * (anchor_index * 4 + 1)];
                    float dx1 = delta_pred[location_index + SZ * (anchor_index * 4 + 2)];
                    float dy1 = delta_pred[location_index + SZ * (anchor_index * 4 + 3)];

                    simpler_nms_delta_t bbox_delta { dx0, dy0, dx1, dy1 };

                    float proposal_confidence =
                            cls_scores[location_index + SZ * (anchor_index + anchors_num * 1)];

                    simpler_nms_roi_t tmp_roi = simpler_nms_gen_bbox(anchors[anchor_index], bbox_delta, anchor_shift_x, anchor_shift_y);
                    simpler_nms_roi_t roi = tmp_roi.clamp({ 0, 0, static_cast<float>(IW - 1), static_cast<float>(IH - 1)});

                    int bbox_w = static_cast<int>(roi.x1 - roi.x0) + 1;
                    int bbox_h = static_cast<int>(roi.y1 - roi.y0) + 1;

                    if (bbox_w >= scaled_min_bbox_size && bbox_h >= scaled_min_bbox_size) {
                        simpler_nms_proposal_t proposal { roi, proposal_confidence, sorted_proposals_confidence.size() };
                        sorted_proposals_confidence.push_back(proposal);
                    }
                }
            }
        }

        sort_and_keep_at_most_top_n(sorted_proposals_confidence, pre_nms_topn_);
        auto res = simpler_nms_perform_nms(sorted_proposals_confidence, iou_threshold_, post_nms_topn_);

        size_t res_num_rois = res.size();

        for (size_t i = 0; i < res_num_rois; ++i) {
            dst[5 * i + 0] = 0;    // roi_batch_ind, always zero on test time
            dst[5 * i + 1] = res[i].x0;
            dst[5 * i + 2] = res[i].y0;
            dst[5 * i + 3] = res[i].x1;
            dst[5 * i + 4] = res[i].y1;
        }
        return OK;
    }

private:
    int min_box_size_;
    int feat_stride_;
    int pre_nms_topn_;
    int post_nms_topn_;
    float iou_threshold_;

    std::vector<float> scales;
    std::vector<float> ratios;

    std::vector<simpler_nms_anchor> anchors_;
};

REG_FACTORY_FOR(ImplFactory<SimplerNMSImpl>, SimplerNMS);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
