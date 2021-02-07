// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cfloat>
#include <vector>
#include <cmath>
#include <string>
#include <utility>
#include <algorithm>

#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class DetectionOutputImpl: public ExtLayerBase {
public:
    explicit DetectionOutputImpl(const CNNLayer* layer);

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override;

private:
    enum CodeType {
        CORNER = 1,
        CENTER_SIZE = 2,
    };

    template<typename DType>
    struct  DCoord {
        float xmin;
        float ymin;
        float xmax;
        float ymax;
    };

    template<typename DType>
    struct DBox {
        int prior;
        DType batch = -1;
        DType label = -1;
        DType score = -1;

        union {
            DCoord<DType> coord = { -1.0, -1.0, -1.0, -1.0 };
            float c[4];
        };
    };

    template<typename DType>
    struct SortElemDescend {
        DBox<DType> box;

        SortElemDescend(const DBox<DType>& b) {
            box = b;
        }

        SortElemDescend() {
        }

        bool operator<(const SortElemDescend& other) const {
            if (box.label == other.box.label && box.score == other.box.score) {
                return (box.prior < other.box.prior);
            } else {
                return (box.label < other.box.label) ||
                       (box.label == other.box.label && box.score >= other.box.score);
            }
        }
    };

    const int idx_location = 0;
    const int idx_confidence = 1;
    const int idx_priors = 2;
    const int idx_arm_confidence = 3;
    const int idx_arm_location = 4;

    int _num_classes = 0;
    int _background_label_id = 0;
    int _top_k = 0;
    int _variance_encoded_in_target = 0;
    int _keep_top_k = 0;
    CodeType _code_type = CORNER;

    bool _share_location    = false;
    bool _clip_before_nms   = false;  // clip bounding boxes before nms step
    bool _clip_after_nms    = false;  // clip bounding boxes after nms step
    bool _decrease_label_id = false;

    bool _with_add_box_pred = false;

    int _image_width = 0;
    int _image_height = 0;
    int _prior_size = 4;
    bool _normalized = true;
    int _offset = 0;

    float _nms_threshold = 0.0f;
    float _confidence_threshold = 0.0f;
    float _objectness_score = 0.0f;

    int _num_loc_classes = 0;
    int _num_priors = 0;
    bool _priors_batches = false;

    template<typename DType>
        inline void ExtractLocation(DType* out, const DType* p_prior_data,
                                    const DType* p_loc_data, const DType* p_loc_var_data,
                                    const DType* p_arm_data,
                                    int prior, int label);

    template<typename DType>
        inline void TransformLocations(DType* out, const DType* prior_data,
                                       const DType* loc_data, const DType* var);

    template<typename DType>
        inline DType CalculateOverlap(const DType *a, const DType *b);
};

DetectionOutputImpl::DetectionOutputImpl(const CNNLayer* layer) {
    try {
        if (layer->insData.size() != 3 && layer->insData.size() != 5)
            THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << layer->name;
        if (layer->outData.empty())
            THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << layer->name;

        _num_classes = layer->GetParamAsInt("num_classes");
        _background_label_id = layer->GetParamAsInt("background_label_id", 0);
        _top_k = layer->GetParamAsInt("top_k", -1);
        _variance_encoded_in_target = layer->GetParamAsBool("variance_encoded_in_target", false);
        _keep_top_k = layer->GetParamAsInt("keep_top_k", -1);
        _nms_threshold = layer->GetParamAsFloat("nms_threshold");
        _confidence_threshold = layer->GetParamAsFloat("confidence_threshold", -FLT_MAX);
        _share_location = layer->GetParamAsBool("share_location", true);
        _clip_before_nms = layer->GetParamAsBool("clip_before_nms", false) ||
                           layer->GetParamAsBool("clip", false);  // for backward compatibility
        _clip_after_nms = layer->GetParamAsBool("clip_after_nms", false);
        _decrease_label_id = layer->GetParamAsBool("decrease_label_id", false);
        _normalized = layer->GetParamAsBool("normalized", true);
        _image_height = layer->GetParamAsInt("input_height", 1);
        _image_width = layer->GetParamAsInt("input_width", 1);
        _prior_size = _normalized ? 4 : 5;
        _offset = _normalized ? 0 : 1;
        _num_loc_classes = _share_location ? 1 : _num_classes;

        _with_add_box_pred = layer->insData.size() == 5;
        _objectness_score = layer->GetParamAsFloat("objectness_score", 0.0f);

        std::string code_type_str = layer->GetParamAsString("code_type", "caffe.PriorBoxParameter.CORNER");
        _code_type = (code_type_str == "caffe.PriorBoxParameter.CENTER_SIZE" ? CodeType::CENTER_SIZE
                                                                             : CodeType::CORNER);

        _num_priors = static_cast<int>(layer->insData[idx_priors].lock()->getDims().back() / _prior_size);
        _priors_batches = layer->insData[idx_priors].lock()->getDims().front() != 1;

        if (_num_priors * _num_loc_classes * 4 != static_cast<int>(layer->insData[idx_location].lock()->getDims()[1]))
            THROW_IE_EXCEPTION << "Number of priors must match number of location predictions ("
                               << _num_priors * _num_loc_classes * 4 << " vs "
                               << layer->insData[idx_location].lock()->getDims()[1] << ")";

        if (_num_priors * _num_classes != static_cast<int>(layer->insData[idx_confidence].lock()->getTensorDesc().getDims().back()))
            THROW_IE_EXCEPTION << "Number of priors must match number of confidence predictions.";

        if (_decrease_label_id && _background_label_id != 0)
            THROW_IE_EXCEPTION << "Cannot use decrease_label_id and background_label_id parameter simultaneously.";

        std::vector<DataConfigurator> in_data_conf(layer->insData.size(), DataConfigurator(ConfLayout::PLN, Precision::FP32));
        addConfig(layer, in_data_conf, {DataConfigurator(ConfLayout::PLN, Precision::FP32)});
    } catch (InferenceEngine::details::InferenceEngineException &ex) {
        errorMsg = ex.what();
    }
}

StatusCode DetectionOutputImpl::execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                   ResponseDesc *resp) noexcept {
    float *dst_data = outputs[0]->buffer();

    const float *loc_data       = inputs[idx_location]->buffer().as<const float *>();
    const float *conf_data      = inputs[idx_confidence]->buffer().as<const float *>();
    const float *prior_data     = inputs[idx_priors]->buffer().as<const float *>();
    const float *arm_conf_data  = inputs.size() > 3 ? inputs[idx_arm_confidence]->buffer().as<const float *>() : nullptr;
    const float *arm_loc_data   = inputs.size() > 4 ? inputs[idx_arm_location]->buffer().as<const float *>() : nullptr;

    const int MB = inputs[idx_confidence]->getTensorDesc().getDims()[0];
    const int DETECTION_SIZE = outputs[0]->getTensorDesc().getDims()[3];
    const int num_results = outputs[0]->getTensorDesc().getDims()[2];

    if (DETECTION_SIZE != 7) {
        return NOT_IMPLEMENTED;
    }

    int dst_data_size = 0;

    if (_keep_top_k > 0) {
        dst_data_size = MB * _keep_top_k * DETECTION_SIZE * sizeof(float);
    } else if (_top_k > 0) {
        dst_data_size = MB * _top_k * _num_classes * DETECTION_SIZE * sizeof(float);
    } else {
        dst_data_size = MB * _num_classes * _num_priors * DETECTION_SIZE * sizeof(float);
    }

    if (dst_data_size > outputs[0]->byteSize()) {
        return OUT_OF_BOUNDS;
    }

    memset(dst_data, 0, dst_data_size);

    DBox<float> box;
    DBox<float> empty_box;

    std::vector<SortElemDescend<float>> boxes;
    std::vector<SortElemDescend<float>> output_boxes;

    int class_start_from = (_background_label_id == 0 ? 1 : 0);

    float* p_dst_data = dst_data;
    int results_count = 0;

    const float* p_loc_data = nullptr;
    const float* p_arm_data = nullptr;

    for (int mb = 0; mb < MB; ++mb) {
        const float* p_conf_data = conf_data + (mb * _num_classes * _num_priors);

        if (_share_location) {
            size_t offt = (mb * _num_priors * 4);
            if (_with_add_box_pred) {
                p_arm_data = arm_loc_data + offt;
            }
            p_loc_data = loc_data + offt;
        } else {
            size_t offt = (mb * _num_priors * _num_loc_classes * 4);
            if (_with_add_box_pred) {
                p_arm_data = arm_loc_data + offt;
            }
            p_loc_data = loc_data + offt;
        }

        const float* p_prior_data = prior_data;
        const float* p_var_data = prior_data + _num_priors * _prior_size;

        if (_priors_batches) {
            p_prior_data += _variance_encoded_in_target ? mb * _num_priors * _prior_size : 2 * mb * _num_priors * _prior_size;
            p_var_data += _variance_encoded_in_target ? 0 : 2 * mb * _num_priors * _prior_size;
        }

        boxes.clear();

        box.batch = mb;

        if (_decrease_label_id) {
            for (int p = 0; p < _num_priors; ++p) {
                const float* pprior_data = p_prior_data + p * 4 + _offset * (p + 1);
                const float* ploc_var_data = p_var_data + p *  4;

                float score = -1.0f;
                int label = -1;

                for (int c = class_start_from; c < _num_classes; ++c) {
                    if (c == _background_label_id) {
                        continue;
                    }

                    float temp = p_conf_data[p * _num_classes + c];

                    if (_with_add_box_pred) {
                        if (arm_conf_data[mb * _num_priors * 2 + p * 2 + 1] < _objectness_score) {
                            temp = 0.0;
                        }
                    }

                    if (temp > score) {
                        score = temp;
                        label = c;
                    }
                }

                if (score >= _confidence_threshold && label != -1) {
                    box.label = static_cast<float>(label - 1);
                    box.score = score;
                    box.prior = p;

                    ExtractLocation(box.c, pprior_data, p_loc_data, ploc_var_data, p_arm_data, p, label);

                    boxes.push_back(box);
                }
            }
        } else {
            for (int p = 0; p < _num_priors; ++p) {
                const float* pprior_data = p_prior_data + p * 4 + _offset * (p + 1);
                const float* ploc_var_data = p_var_data + p *  4;

                for (int c = class_start_from; c < _num_classes; ++c) {
                    if (c == _background_label_id) {
                        continue;
                    }

                    float score = p_conf_data[p * _num_classes + c];

                    if (_with_add_box_pred) {
                        if (arm_conf_data[mb * _num_priors * 2 + p * 2 + 1] < _objectness_score) {
                            score = 0.0;
                        }
                    }

                    if (score > _confidence_threshold) {
                        box.label = static_cast<float>(c);
                        box.score = score;
                        box.prior = p;

                        ExtractLocation(box.c, pprior_data, p_loc_data, ploc_var_data, p_arm_data, p, c);

                        boxes.push_back(box);
                    }
                }
            }
        }

        if (boxes.empty()) {
            continue;
        }

        std::stable_sort(boxes.begin(), boxes.end());

        int detections_total = static_cast<int>(boxes.size());
        int to_keep = std::max<int>(_top_k, _keep_top_k);

        if (to_keep > -1 && detections_total > to_keep) {
            std::sort(boxes.begin(), boxes.end(),
                      [](const SortElemDescend<float>& box1, const SortElemDescend<float>& box2) {
                          return (box1.box.score > box2.box.score);
                      });

            boxes.resize(to_keep);
            std::stable_sort(boxes.begin(), boxes.end());
            detections_total = static_cast<int>(boxes.size());
        }

        for (int i = 0; i < detections_total; i++) {
            const DBox<float>& iBox = boxes[i].box;

            if (iBox.batch == -1) {
                continue;
            }

            for (int j = i + 1; j < detections_total; ++j) {
                DBox<float>& jBox = boxes[j].box;

                if (jBox.batch == -1) {
                    continue;
                }

                if (iBox.label != jBox.label) {
                    continue;
                }

                float iou = CalculateOverlap(iBox.c, jBox.c);

                if (iou >= _nms_threshold) {
                    jBox.batch = -1;
                }
            }
        }

        boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
            [](const SortElemDescend<float>& box) {
                return (box.box.batch == -1);
            }), boxes.end());

        detections_total = static_cast<int>(boxes.size());

        if (_keep_top_k > -1 && detections_total > _keep_top_k) {
            std::sort(boxes.begin(), boxes.end(),
                      [](const SortElemDescend<float>& box1, const SortElemDescend<float>& box2) {
                          return (box1.box.score > box2.box.score);
                      });

            boxes.resize(_keep_top_k);
            std::stable_sort(boxes.begin(), boxes.end());
            detections_total = static_cast<int>(boxes.size());
        }

        if (_clip_after_nms) {
            std::for_each(boxes.begin(), boxes.end(), [](SortElemDescend<float>& box) {
                box.box.coord.xmin = (std::max)(0.0f, (std::min)(1.0f, box.box.coord.xmin));
                box.box.coord.ymin = (std::max)(0.0f, (std::min)(1.0f, box.box.coord.ymin));
                box.box.coord.xmax = (std::max)(0.0f, (std::min)(1.0f, box.box.coord.xmax));
                box.box.coord.ymax = (std::max)(0.0f, (std::min)(1.0f, box.box.coord.ymax));
            });
        }

        for (int i = 0; i < detections_total; i++, results_count++, p_dst_data += DETECTION_SIZE) {
            memcpy(p_dst_data, &boxes[i].box.batch, sizeof(box));
        }
    }

    if (results_count < num_results) {
        // marker at end of boxes list
        *p_dst_data = -1;
    }

    return OK;
}

template<typename DType>
DType DetectionOutputImpl::CalculateOverlap(const DType* a, const DType* b) {
    DType w = std::max(DType(0), std::min(a[2], b[2]) - std::max(a[0], b[0]));
    DType h = std::max(DType(0), std::min(a[3], b[3]) - std::max(a[1], b[1]));
    DType i = w * h;
    DType u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    return (u <= 0.0f) ? static_cast<DType>(0) : static_cast<DType>(i / u);
}

template<typename DType>
void DetectionOutputImpl::ExtractLocation(DType* out, const DType* p_prior_data,
                                          const DType* p_loc_data, const DType* p_loc_var_data,
                                          const DType* p_arm_data,
                                          int prior, int label) {
    const DType* parm_data;
    const DType* ploc_data;

    if (_share_location) {
        size_t offt = prior * 4;
        if (_with_add_box_pred) {
            parm_data = p_arm_data + offt;
        }
        ploc_data = p_loc_data + offt;
    } else {
        size_t offt = (label + prior * _num_loc_classes) * 4;
        if (_with_add_box_pred) {
            parm_data = p_arm_data + offt;
        }
        ploc_data = p_loc_data + offt;
    }

    if (_with_add_box_pred) {
        TransformLocations(out, p_prior_data, parm_data, p_loc_var_data);
        TransformLocations(out, out, ploc_data, p_loc_var_data);
    } else {
        TransformLocations(out, p_prior_data, ploc_data, p_loc_var_data);
    }
}

template<typename DType>
void DetectionOutputImpl::TransformLocations(DType* out, const DType* prior_data,
                                            const DType* loc_data, const DType* var) {
    DType prior_xmin = prior_data[0];
    DType prior_ymin = prior_data[1];
    DType prior_xmax = prior_data[2];
    DType prior_ymax = prior_data[3];

    DType loc_xmin = loc_data[0];
    DType loc_ymin = loc_data[1];
    DType loc_xmax = loc_data[2];
    DType loc_ymax = loc_data[3];

    if (!_variance_encoded_in_target) {
        loc_xmin *= var[0];
        loc_ymin *= var[1];
        loc_xmax *= var[2];
        loc_ymax *= var[3];
    }

    if (!_normalized) {
        prior_xmin /= _image_width;
        prior_ymin /= _image_height;
        prior_xmax /= _image_width;
        prior_ymax /= _image_height;
    }

    DType prior_w = prior_xmax - prior_xmin;
    DType prior_h = prior_ymax - prior_ymin;

    DType xmin;
    DType ymin;
    DType xmax;
    DType ymax;

    switch (_code_type) {
        case CORNER: {
            xmin = prior_xmin + loc_xmin;
            ymin = prior_ymin + loc_ymin;
            xmax = prior_xmax + loc_xmax;
            ymax = prior_ymax + loc_ymax;
        }
        break;

        case CENTER_SIZE: {
            DType prior_center_x = static_cast<DType>((prior_xmin + prior_xmax) / 2.f);
            DType prior_center_y = static_cast<DType>((prior_ymin + prior_ymax) / 2.f);

            DType box_center_x = loc_xmin * prior_w + prior_center_x;
            DType box_center_y = loc_ymin * prior_h + prior_center_y;

            DType box_width = static_cast<DType>(exp(loc_xmax) * prior_w / 2.0f);
            DType box_height = static_cast<DType>(exp(loc_ymax) * prior_h / 2.0f);

            xmin = box_center_x - box_width;
            ymin = box_center_y - box_height;
            xmax = box_center_x + box_width;
            ymax = box_center_y + box_height;
        }
        break;
    }

    if (_clip_before_nms) {
        xmin = (std::max)(0.0f, (std::min)(1.0f, xmin));
        ymin = (std::max)(0.0f, (std::min)(1.0f, ymin));
        xmax = (std::max)(0.0f, (std::min)(1.0f, xmax));
        ymax = (std::max)(0.0f, (std::min)(1.0f, ymax));
    }

    out[0] = xmin;
    out[1] = ymin;
    out[2] = xmax;
    out[3] = ymax;
}

REG_FACTORY_FOR(DetectionOutputImpl, DetectionOutput);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
