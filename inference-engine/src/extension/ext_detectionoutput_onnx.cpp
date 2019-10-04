// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cassert>
#include <cfloat>
#include <vector>
#include <cmath>
#include <string>
#include <utility>
#include <algorithm>
#include "ie_parallel.hpp"


namespace {
struct Indexer {
  const std::vector<int> dims_;
  int total_{1};

  explicit Indexer(const std::vector<int>& dims) : dims_(dims) {
      total_ = 1;
      for (size_t i = 0; i < dims_.size(); ++i) {
          total_ *= dims_[i];
      }
  }

  int operator()(const std::vector<int>& idx) const {
      int flat_idx = 0;
      assert(idx.size() == dims_.size());
      for (size_t i = 0; i < dims_.size(); ++i) {
          assert(0 <= idx[i] && idx[i] < dims_[i]);
          flat_idx = flat_idx * dims_[i] + idx[i];
      }
      assert(flat_idx < total_);
      return flat_idx;
  }
};
}  // namespace


namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

static
void refine_boxes(const float* boxes, const float* deltas, const float* weights, const float* scores,
                  float* refined_boxes, float* refined_boxes_areas, float* refined_scores,
                  const int rois_num, const int classes_num,
                  const float img_H, const float img_W,
                  const float max_delta_log_wh,
                  float coordinates_offset) {
    Indexer box_idx({rois_num, 4});
    Indexer delta_idx({rois_num, classes_num, 4});
    Indexer score_idx({rois_num, classes_num});

    Indexer refined_box_idx({classes_num, rois_num, 4});
    Indexer refined_score_idx({classes_num, rois_num});

    for (int roi_idx = 0; roi_idx < rois_num; ++roi_idx) {
        float x0 = boxes[box_idx({roi_idx, 0})];
        float y0 = boxes[box_idx({roi_idx, 1})];
        float x1 = boxes[box_idx({roi_idx, 2})];
        float y1 = boxes[box_idx({roi_idx, 3})];

        if (x1 - x0 <= 0 || y1 - y0 <= 0) {
            continue;
        }

        // width & height of box
        const float ww = x1 - x0 + coordinates_offset;
        const float hh = y1 - y0 + coordinates_offset;
        // center location of box
        const float ctr_x = x0 + 0.5f * ww;
        const float ctr_y = y0 + 0.5f * hh;

        for (int class_idx = 1; class_idx < classes_num; ++class_idx) {
            const float dx = deltas[delta_idx({roi_idx, class_idx, 0})] / weights[0];
            const float dy = deltas[delta_idx({roi_idx, class_idx, 1})] / weights[1];
            const float d_log_w = deltas[delta_idx({roi_idx, class_idx, 2})] / weights[2];
            const float d_log_h = deltas[delta_idx({roi_idx, class_idx, 3})] / weights[3];

            // new center location according to deltas (dx, dy)
            const float pred_ctr_x = dx * ww + ctr_x;
            const float pred_ctr_y = dy * hh + ctr_y;
            // new width & height according to deltas d(log w), d(log h)
            const float pred_w = std::exp(std::min(d_log_w, max_delta_log_wh)) * ww;
            const float pred_h = std::exp(std::min(d_log_h, max_delta_log_wh)) * hh;

            // update upper-left corner location
            float x0_new = pred_ctr_x - 0.5f * pred_w;
            float y0_new = pred_ctr_y - 0.5f * pred_h;
            // update lower-right corner location
            float x1_new = pred_ctr_x + 0.5f * pred_w - coordinates_offset;
            float y1_new = pred_ctr_y + 0.5f * pred_h - coordinates_offset;

            // adjust new corner locations to be within the image region,
            x0_new = std::max<float>(0.0f, std::min<float>(x0_new, img_W - coordinates_offset));
            y0_new = std::max<float>(0.0f, std::min<float>(y0_new, img_H - coordinates_offset));
            x1_new = std::max<float>(0.0f, std::min<float>(x1_new, img_W - coordinates_offset));
            y1_new = std::max<float>(0.0f, std::min<float>(y1_new, img_H - coordinates_offset));

            // recompute new width & height
            const float box_w = x1_new - x0_new + coordinates_offset;
            const float box_h = y1_new - y0_new + coordinates_offset;

            refined_boxes[refined_box_idx({class_idx, roi_idx, 0})] = x0_new;
            refined_boxes[refined_box_idx({class_idx, roi_idx, 1})] = y0_new;
            refined_boxes[refined_box_idx({class_idx, roi_idx, 2})] = x1_new;
            refined_boxes[refined_box_idx({class_idx, roi_idx, 3})] = y1_new;

            refined_boxes_areas[refined_score_idx({class_idx, roi_idx})] = box_w * box_h;

            refined_scores[refined_score_idx({class_idx, roi_idx})] = scores[score_idx({roi_idx, class_idx})];
        }
    }
}

template <typename T>
static bool SortScorePairDescend(const std::pair<float, T>& pair1,
                                 const std::pair<float, T>& pair2) {
    return pair1.first > pair2.first;
}


struct ConfidenceComparator {
    explicit ConfidenceComparator(const float* conf_data) : _conf_data(conf_data) {}

    bool operator()(int idx1, int idx2) {
        if (_conf_data[idx1] > _conf_data[idx2]) return true;
        if (_conf_data[idx1] < _conf_data[idx2]) return false;
        return idx1 < idx2;
    }

    const float* _conf_data;
};

static inline float JaccardOverlap(const float *decoded_bbox,
                                   const float *bbox_sizes,
                                   const int idx1,
                                   const int idx2,
                                   const float coordinates_offset = 1) {
    float xmin1 = decoded_bbox[idx1 * 4 + 0];
    float ymin1 = decoded_bbox[idx1 * 4 + 1];
    float xmax1 = decoded_bbox[idx1 * 4 + 2];
    float ymax1 = decoded_bbox[idx1 * 4 + 3];

    float xmin2 = decoded_bbox[idx2 * 4 + 0];
    float ymin2 = decoded_bbox[idx2 * 4 + 1];
    float ymax2 = decoded_bbox[idx2 * 4 + 3];
    float xmax2 = decoded_bbox[idx2 * 4 + 2];

    if (xmin2 > xmax1 || xmax2 < xmin1 || ymin2 > ymax1 || ymax2 < ymin1) {
        return 0.0f;
    }

    float intersect_xmin = std::max(xmin1, xmin2);
    float intersect_ymin = std::max(ymin1, ymin2);
    float intersect_xmax = std::min(xmax1, xmax2);
    float intersect_ymax = std::min(ymax1, ymax2);

    float intersect_width  = intersect_xmax - intersect_xmin + coordinates_offset;
    float intersect_height = intersect_ymax - intersect_ymin + coordinates_offset;

    if (intersect_width <= 0 || intersect_height <= 0) {
        return 0.0f;
    }

    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = bbox_sizes[idx1];
    float bbox2_size = bbox_sizes[idx2];

    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
}


static void nms_cf(const float* conf_data,
                          const float* bboxes,
                          const float* sizes,
                          int* buffer,
                          int* indices,
                          int& detections,
                          const int boxes_num,
                          const int pre_nms_topn,
                          const int post_nms_topn,
                          const float confidence_threshold,
                          const float nms_threshold) {
    int count = 0;
    for (int i = 0; i < boxes_num; ++i) {
        if (conf_data[i] > confidence_threshold) {
            indices[count] = i;
            count++;
        }
    }

    int num_output_scores = (pre_nms_topn == -1 ? count : std::min<int>(pre_nms_topn, count));

    std::partial_sort_copy(indices, indices + count,
                           buffer, buffer + num_output_scores,
                           ConfidenceComparator(conf_data));

    detections = 0;
    for (int i = 0; i < num_output_scores; ++i) {
        const int idx = buffer[i];

        bool keep = true;
        for (int k = 0; k < detections; ++k) {
            const int kept_idx = indices[k];
            float overlap = JaccardOverlap(bboxes, sizes, idx, kept_idx);
            if (overlap > nms_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            indices[detections] = idx;
            detections++;
        }
    }

    detections = (post_nms_topn == -1 ? detections : std::min<int>(post_nms_topn, detections));
}


class ExperimentalDetectronDetectionOutputImpl: public ExtLayerBase {
private:
    const int INPUT_ROIS {0};
    const int INPUT_DELTAS {1};
    const int INPUT_SCORES {2};
    const int INPUT_IM_INFO {3};

    const int OUTPUT_BOXES {0};
    const int OUTPUT_CLASSES {1};
    const int OUTPUT_SCORES {2};

public:
    explicit ExperimentalDetectronDetectionOutputImpl(const CNNLayer* layer) {
        try {
            score_threshold_ = layer->GetParamAsFloat("score_threshold");
            nms_threshold_ = layer->GetParamAsFloat("nms_threshold");
            max_delta_log_wh_ = layer->GetParamAsFloat("max_delta_log_wh");
            classes_num_ = layer->GetParamAsInt("num_classes");
            max_detections_per_class_ = layer->GetParamAsInt("post_nms_count");
            max_detections_per_image_ = layer->GetParamAsInt("max_detections_per_image");
            class_agnostic_box_regression_ = layer->GetParamAsBool("class_agnostic_box_regression", false);
            deltas_weights_ = layer->GetParamAsFloats("deltas_weights");

            std::vector<DataConfigurator> inputs_layouts(layer->insData.size(), DataConfigurator(ConfLayout::PLN));
            std::vector<DataConfigurator> outputs_layouts(layer->outData.size(), DataConfigurator(ConfLayout::PLN));
            addConfig(layer, inputs_layouts, outputs_layouts);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const int rois_num = inputs[INPUT_ROIS]->getTensorDesc().getDims()[0];
        assert(classes_num_ == static_cast<int>(inputs[INPUT_SCORES]->getTensorDesc().getDims()[1]));
        assert(4 * classes_num_ == static_cast<int>(inputs[INPUT_DELTAS]->getTensorDesc().getDims()[1]));

        const auto* boxes = inputs[INPUT_ROIS]->buffer().as<const float *>();
        const auto* deltas = inputs[INPUT_DELTAS]->buffer().as<const float *>();
        const auto* scores = inputs[INPUT_SCORES]->buffer().as<const float *>();
        const auto* im_info = inputs[INPUT_IM_INFO]->buffer().as<const float *>();

        auto* output_boxes = outputs[OUTPUT_BOXES]->buffer().as<float *>();
        auto* output_scores = outputs[OUTPUT_SCORES]->buffer().as<float *>();
        auto* output_classes = outputs[OUTPUT_CLASSES]->buffer().as<float *>();

        const float img_H = im_info[0];
        const float img_W = im_info[1];

        // Apply deltas.
        std::vector<float> refined_boxes(classes_num_ * rois_num * 4, 0);
        std::vector<float> refined_scores(classes_num_ * rois_num, 0);
        std::vector<float> refined_boxes_areas(classes_num_ * rois_num, 0);
        Indexer refined_box_idx({classes_num_, rois_num, 4});
        Indexer refined_score_idx({classes_num_, rois_num});

        refine_boxes(boxes, deltas, &deltas_weights_[0], scores,
                     &refined_boxes[0], &refined_boxes_areas[0], &refined_scores[0],
                     rois_num, classes_num_,
                     img_H, img_W,
                     max_delta_log_wh_,
                     1.0f);

        // Apply NMS class-wise.
        std::vector<int> buffer(rois_num, 0);
        std::vector<int> indices(classes_num_ * rois_num, 0);
        std::vector<int> detections_per_class(classes_num_, 0);
        int total_detections_num = 0;

        for (int class_idx = 1; class_idx < classes_num_; ++class_idx) {
            nms_cf(&refined_scores[refined_score_idx({class_idx, 0})],
                   &refined_boxes[refined_box_idx({class_idx, 0, 0})],
                   &refined_boxes_areas[refined_score_idx({class_idx, 0})],
                   &buffer[0],
                   &indices[total_detections_num],
                   detections_per_class[class_idx],
                   rois_num,
                   -1,
                   max_detections_per_class_,
                   score_threshold_,
                   nms_threshold_);
            total_detections_num += detections_per_class[class_idx];
        }

        // Leave only max_detections_per_image_ detections.
        // confidence, <class, index>
        std::vector<std::pair<float, std::pair<int, int>>> conf_index_class_map;

        int indices_offset = 0;
        for (int c = 0; c < classes_num_; ++c) {
            int n = detections_per_class[c];
            for (int i = 0; i < n; ++i) {
                int idx = indices[indices_offset + i];
                float score = refined_scores[refined_score_idx({c, idx})];
                conf_index_class_map.push_back(std::make_pair(score, std::make_pair(c, idx)));
            }
            indices_offset += n;
        }

        assert(max_detections_per_image_ > 0);
        if (total_detections_num > max_detections_per_image_) {
            std::partial_sort(conf_index_class_map.begin(),
                              conf_index_class_map.begin() + max_detections_per_image_,
                              conf_index_class_map.end(),
                              SortScorePairDescend<std::pair<int, int>>);
            conf_index_class_map.resize(max_detections_per_image_);
            total_detections_num = max_detections_per_image_;
        }

        // Fill outputs.
        memset(output_boxes, 0, max_detections_per_image_ * 4 * sizeof(float));
        memset(output_scores, 0, max_detections_per_image_ * sizeof(float));
        memset(output_classes, 0, max_detections_per_image_ * sizeof(float));

        int i = 0;
        for (const auto & detection : conf_index_class_map) {
            float score = detection.first;
            int cls = detection.second.first;
            int idx = detection.second.second;
            output_boxes[4 * i + 0] = refined_boxes[refined_box_idx({cls, idx, 0})];
            output_boxes[4 * i + 1] = refined_boxes[refined_box_idx({cls, idx, 1})];
            output_boxes[4 * i + 2] = refined_boxes[refined_box_idx({cls, idx, 2})];
            output_boxes[4 * i + 3] = refined_boxes[refined_box_idx({cls, idx, 3})];
            output_scores[i] = score;
            output_classes[i] = static_cast<float>(cls);
            ++i;
        }

        return OK;
    }

private:
    float score_threshold_;
    float nms_threshold_;
    float max_delta_log_wh_;
    int classes_num_;
    int max_detections_per_class_;
    int max_detections_per_image_;
    bool class_agnostic_box_regression_;
    std::vector<float> deltas_weights_;
};



REG_FACTORY_FOR(ImplFactory<ExperimentalDetectronDetectionOutputImpl>, ExperimentalDetectronDetectionOutput);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
