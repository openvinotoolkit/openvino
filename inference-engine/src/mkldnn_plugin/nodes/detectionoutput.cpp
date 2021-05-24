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
#include "caseless.hpp"
#include "ie_parallel.hpp"
#include "common/tensor_desc_creator.h"
#include <ngraph/op/detection_output.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

template <typename T>
static bool SortScorePairDescend(const std::pair<float, T>& pair1,
                                 const std::pair<float, T>& pair2) {
    return pair1.first > pair2.first;
}

class DetectionOutputImpl: public ExtLayerBase {
public:
    bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            auto doOp = ngraph::as_type_ptr<const ngraph::op::v0::DetectionOutput>(op);
            if (!doOp) {
                errorMessage = "Node is not an instance of the DetectionOutput from the operations set v0.";
                return false;
            }
            if (!details::CaselessEq<std::string>()(doOp->get_attrs().code_type, "caffe.PriorBoxParameter.CENTER_SIZE") &&
                    !details::CaselessEq<std::string>()(doOp->get_attrs().code_type, "caffe.PriorBoxParameter.CORNER")) {
                errorMessage = "Unsupported code_type attribute.";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    explicit DetectionOutputImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }
            if (op->get_input_size() != 3 && op->get_input_size() != 5)
                IE_THROW() <<  "Invalid number of input edges.";

            if (op->get_output_size() != 1)
                IE_THROW() << "Invalid number of output edges.";

            auto doOp = ngraph::as_type_ptr<const ngraph::op::v0::DetectionOutput>(op);
            auto attributes = doOp->get_attrs();

            _num_classes = attributes.num_classes;
            _background_label_id = attributes.background_label_id;
            _top_k = attributes.top_k;
            _variance_encoded_in_target = attributes.variance_encoded_in_target;
            _keep_top_k = attributes.keep_top_k[0];
            _nms_threshold = attributes.nms_threshold;
            _confidence_threshold = attributes.confidence_threshold;
            _share_location = attributes.share_location;
            _clip_before_nms = attributes.clip_before_nms;
            _clip_after_nms = attributes.clip_after_nms;
            _decrease_label_id = attributes.decrease_label_id;
            _normalized = attributes.normalized;
            _image_height = attributes.input_height;
            _image_width = attributes.input_width;
            _prior_size = _normalized ? 4 : 5;
            _offset = _normalized ? 0 : 1;
            _num_loc_classes = _share_location ? 1 : _num_classes;

            with_add_box_pred = op->get_input_size() == 5;
            _objectness_score = attributes.objectness_score;

            _code_type = (details::CaselessEq<std::string>()(attributes.code_type, "caffe.PriorBoxParameter.CENTER_SIZE") ?
                CodeType::CENTER_SIZE : CodeType::CORNER);

            _num_priors = static_cast<int>(op->get_input_shape(idx_priors).back() / _prior_size);
            _priors_batches = op->get_input_shape(idx_priors).front() != 1;

            if (_num_priors * _num_loc_classes * 4 != static_cast<int>(op->get_input_shape(idx_location)[1]))
                IE_THROW() << "Number of priors must match number of location predictions ("
                                   << _num_priors * _num_loc_classes * 4 << " vs "
                                   << op->get_input_shape(idx_location)[1] << ")";

            if (_num_priors * _num_classes != static_cast<int>(op->get_input_shape(idx_confidence).back()))
                IE_THROW() << "Number of priors must match number of confidence predictions.";

            if (_decrease_label_id && _background_label_id != 0)
                IE_THROW() << "Cannot use decrease_label_id and background_label_id parameter simultaneously.";

            _num = static_cast<int>(op->get_input_shape(idx_confidence)[0]);

            _decoded_bboxes.resize(_num * _num_classes * _num_priors * 4);
            _buffer.resize(_num * _num_classes * _num_priors);
            _indices.resize(_num * _num_classes * _num_priors);
            _detections_count.resize(_num * _num_classes);
            _bbox_sizes.resize(_num * _num_classes * _num_priors);
            _num_priors_actual.resize(_num);

            const auto &confSize = op->get_input_shape(idx_confidence);
            _reordered_conf.resize(std::accumulate(confSize.begin(), confSize.end(), 1, std::multiplies<size_t>()));

            std::vector<DataConfigurator> inDataConfigurators(op->get_input_size(), {TensorDescCreatorTypes::ncsp, Precision::FP32});
            addConfig(op, inDataConfigurators,
                          {{TensorDescCreatorTypes::ncsp, Precision::FP32}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
            throw;
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        float *dst_data = outputs[0]->buffer();

        const float *loc_data    = inputs[idx_location]->buffer().as<const float *>();
        const float *conf_data   = inputs[idx_confidence]->buffer().as<const float *>();
        const float *prior_data  = inputs[idx_priors]->buffer().as<const float *>();
        const float *arm_conf_data = inputs.size() > 3 ? inputs[idx_arm_confidence]->buffer().as<const float *>() : nullptr;
        const float *arm_loc_data = inputs.size() > 4 ? inputs[idx_arm_location]->buffer().as<const float *>() : nullptr;

        const int N = inputs[idx_confidence]->getTensorDesc().getDims()[0];

        float *decoded_bboxes_data = _decoded_bboxes.data();
        float *reordered_conf_data = _reordered_conf.data();
        float *bbox_sizes_data     = _bbox_sizes.data();
        int *detections_data       = _detections_count.data();
        int *buffer_data           = _buffer.data();
        int *indices_data          = _indices.data();
        int *num_priors_actual     = _num_priors_actual.data();

        for (int n = 0; n < N; ++n) {
            const float *ppriors = prior_data;
            const float *prior_variances = prior_data + _num_priors*_prior_size;
            if (_priors_batches) {
                ppriors += _variance_encoded_in_target ? n*_num_priors*_prior_size : 2*n*_num_priors*_prior_size;
                prior_variances += _variance_encoded_in_target ? 0 : 2*n*_num_priors*_prior_size;
            }

            if (_share_location) {
                const float *ploc = loc_data + n*4*_num_priors;
                float *pboxes = decoded_bboxes_data + n*4*_num_priors;
                float *psizes = bbox_sizes_data + n*_num_priors;

                if (with_add_box_pred) {
                    const float *p_arm_loc = arm_loc_data + n*4*_num_priors;
                    decodeBBoxes(ppriors, p_arm_loc, prior_variances, pboxes, psizes, num_priors_actual, n, _offset, _prior_size);
                    decodeBBoxes(pboxes, ploc, prior_variances, pboxes, psizes, num_priors_actual, n, 0, 4, false);
                } else {
                    decodeBBoxes(ppriors, ploc, prior_variances, pboxes, psizes, num_priors_actual, n, _offset, _prior_size);
                }
            } else {
                for (int c = 0; c < _num_loc_classes; ++c) {
                    if (c == _background_label_id) {
                        continue;
                    }
                    const float *ploc = loc_data + n*4*_num_loc_classes*_num_priors + c*4;
                    float *pboxes = decoded_bboxes_data + n*4*_num_loc_classes*_num_priors + c*4*_num_priors;
                    float *psizes = bbox_sizes_data + n*_num_loc_classes*_num_priors + c*_num_priors;
                    if (with_add_box_pred) {
                        const float *p_arm_loc = arm_loc_data + n*4*_num_loc_classes*_num_priors + c*4;
                        decodeBBoxes(ppriors, p_arm_loc, prior_variances, pboxes, psizes, num_priors_actual, n, _offset, _prior_size);
                        decodeBBoxes(pboxes, ploc, prior_variances, pboxes, psizes, num_priors_actual, n, 0, 4, false);
                    } else {
                        decodeBBoxes(ppriors, ploc, prior_variances, pboxes, psizes, num_priors_actual, n, _offset, _prior_size);
                    }
                }
            }
        }

        if (with_add_box_pred) {
            for (int n = 0; n < N; ++n) {
                for (int p = 0; p < _num_priors; ++p) {
                    if (arm_conf_data[n*_num_priors*2 + p * 2 + 1] < _objectness_score) {
                        for (int c = 0; c < _num_classes; ++c) {
                            reordered_conf_data[n*_num_priors*_num_classes + c*_num_priors + p] = c == _background_label_id ? 1.0f : 0.0f;
                        }
                    } else {
                        for (int c = 0; c < _num_classes; ++c) {
                            reordered_conf_data[n*_num_priors*_num_classes + c*_num_priors + p] = conf_data[n*_num_priors*_num_classes + p*_num_classes + c];
                        }
                    }
                }
            }
        } else {
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < _num_classes; ++c) {
                    for (int p = 0; p < _num_priors; ++p) {
                        reordered_conf_data[n*_num_priors*_num_classes + c*_num_priors + p] = conf_data[n*_num_priors*_num_classes + p*_num_classes + c];
                    }
                }
            }
        }

        memset(detections_data, 0, N*_num_classes*sizeof(int));

        for (int n = 0; n < N; ++n) {
            int detections_total = 0;

            if (!_decrease_label_id) {
                // Caffe style
                parallel_for(_num_classes, [&](int c) {
                    if (c != _background_label_id) {  // Ignore background class
                        int *pindices    = indices_data + n*_num_classes*_num_priors + c*_num_priors;
                        int *pbuffer     = buffer_data + c*_num_priors;
                        int *pdetections = detections_data + n*_num_classes + c;

                        const float *pconf = reordered_conf_data + n*_num_classes*_num_priors + c*_num_priors;
                        const float *pboxes;
                        const float *psizes;
                        if (_share_location) {
                            pboxes = decoded_bboxes_data + n*4*_num_priors;
                            psizes = bbox_sizes_data + n*_num_priors;
                        } else {
                            pboxes = decoded_bboxes_data + n*4*_num_classes*_num_priors + c*4*_num_priors;
                            psizes = bbox_sizes_data + n*_num_classes*_num_priors + c*_num_priors;
                        }

                        nms_cf(pconf, pboxes, psizes, pbuffer, pindices, *pdetections, num_priors_actual[n]);
                    }
                });
            } else {
                // MXNet style
                int *pindices = indices_data + n*_num_classes*_num_priors;
                int *pbuffer = buffer_data;
                int *pdetections = detections_data + n*_num_classes;

                const float *pconf = reordered_conf_data + n*_num_classes*_num_priors;
                const float *pboxes = decoded_bboxes_data + n*4*_num_loc_classes*_num_priors;
                const float *psizes = bbox_sizes_data + n*_num_loc_classes*_num_priors;

                nms_mx(pconf, pboxes, psizes, pbuffer, pindices, pdetections, _num_priors);
            }

            for (int c = 0; c < _num_classes; ++c) {
                detections_total += detections_data[n*_num_classes + c];
            }

            if (_keep_top_k > -1 && detections_total > _keep_top_k) {
                std::vector<std::pair<float, std::pair<int, int>>> conf_index_class_map;

                for (int c = 0; c < _num_classes; ++c) {
                    int detections = detections_data[n*_num_classes + c];
                    int *pindices = indices_data + n*_num_classes*_num_priors + c*_num_priors;

                    float *pconf  = reordered_conf_data + n*_num_classes*_num_priors + c*_num_priors;

                    for (int i = 0; i < detections; ++i) {
                        int idx = pindices[i];
                        conf_index_class_map.push_back(std::make_pair(pconf[idx], std::make_pair(c, idx)));
                    }
                }

                std::sort(conf_index_class_map.begin(), conf_index_class_map.end(),
                          SortScorePairDescend<std::pair<int, int>>);
                conf_index_class_map.resize(_keep_top_k);

                // Store the new indices.
                memset(detections_data + n*_num_classes, 0, _num_classes * sizeof(int));

                for (size_t j = 0; j < conf_index_class_map.size(); ++j) {
                    int label = conf_index_class_map[j].second.first;
                    int idx = conf_index_class_map[j].second.second;
                    int *pindices = indices_data + n * _num_classes * _num_priors + label * _num_priors;
                    pindices[detections_data[n*_num_classes + label]] = idx;
                    detections_data[n*_num_classes + label]++;
                }
            }
        }

        const int num_results = outputs[0]->getTensorDesc().getDims()[2];
        const int DETECTION_SIZE = outputs[0]->getTensorDesc().getDims()[3];
        if (DETECTION_SIZE != 7) {
            return NOT_IMPLEMENTED;
        }

        int dst_data_size = 0;
        if (_keep_top_k > 0)
            dst_data_size = N * _keep_top_k * DETECTION_SIZE * sizeof(float);
        else if (_top_k > 0)
            dst_data_size = N * _top_k * _num_classes * DETECTION_SIZE * sizeof(float);
        else
            dst_data_size = N * _num_classes * _num_priors * DETECTION_SIZE * sizeof(float);

        if (dst_data_size > outputs[0]->byteSize()) {
            return OUT_OF_BOUNDS;
        }
        memset(dst_data, 0, dst_data_size);

        int count = 0;
        for (int n = 0; n < N; ++n) {
            const float *pconf   = reordered_conf_data + n * _num_priors * _num_classes;
            const float *pboxes  = decoded_bboxes_data + n*_num_priors*4*_num_loc_classes;
            const int *pindices  = indices_data + n*_num_classes*_num_priors;

            for (int c = 0; c < _num_classes; ++c) {
                for (int i = 0; i < detections_data[n*_num_classes + c]; ++i) {
                    int idx = pindices[c*_num_priors + i];

                    dst_data[count * DETECTION_SIZE + 0] = static_cast<float>(n);
                    dst_data[count * DETECTION_SIZE + 1] = static_cast<float>(_decrease_label_id ? c-1 : c);
                    dst_data[count * DETECTION_SIZE + 2] = pconf[c*_num_priors + idx];

                    float xmin = _share_location ? pboxes[idx*4 + 0] :
                                 pboxes[c*4*_num_priors + idx*4 + 0];
                    float ymin = _share_location ? pboxes[idx*4 + 1] :
                                 pboxes[c*4*_num_priors + idx*4 + 1];
                    float xmax = _share_location ? pboxes[idx*4 + 2] :
                                 pboxes[c*4*_num_priors + idx*4 + 2];
                    float ymax = _share_location ? pboxes[idx*4 + 3] :
                                 pboxes[c*4*_num_priors + idx*4 + 3];

                    if (_clip_after_nms) {
                        xmin = (std::max)(0.0f, (std::min)(1.0f, xmin));
                        ymin = (std::max)(0.0f, (std::min)(1.0f, ymin));
                        xmax = (std::max)(0.0f, (std::min)(1.0f, xmax));
                        ymax = (std::max)(0.0f, (std::min)(1.0f, ymax));
                    }

                    dst_data[count * DETECTION_SIZE + 3] = xmin;
                    dst_data[count * DETECTION_SIZE + 4] = ymin;
                    dst_data[count * DETECTION_SIZE + 5] = xmax;
                    dst_data[count * DETECTION_SIZE + 6] = ymax;

                    ++count;
                }
            }
        }

        if (count < num_results) {
            // marker at end of boxes list
            dst_data[count * DETECTION_SIZE + 0] = -1;
        }

        return OK;
    }

private:
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
    int _code_type = 0;

    bool _share_location    = false;
    bool _clip_before_nms   = false;  // clip bounding boxes before nms step
    bool _clip_after_nms    = false;  // clip bounding boxes after nms step
    bool _decrease_label_id = false;

    bool with_add_box_pred = false;

    int _image_width = 0;
    int _image_height = 0;
    int _prior_size = 4;
    bool _normalized = true;
    int _offset = 0;

    float _nms_threshold = 0.0f;
    float _confidence_threshold = 0.0f;
    float _objectness_score = 0.0f;

    int _num = 0;
    int _num_loc_classes = 0;
    int _num_priors = 0;
    bool _priors_batches = false;

    enum CodeType {
        CORNER = 1,
        CENTER_SIZE = 2,
    };

    void decodeBBoxes(const float *prior_data, const float *loc_data, const float *variance_data,
                      float *decoded_bboxes, float *decoded_bbox_sizes, int* num_priors_actual, int n, const int& offs, const int& pr_size,
                      bool decodeType = true); // after ARM = false

    void nms_cf(const float *conf_data, const float *bboxes, const float *sizes,
                int *buffer, int *indices, int &detections, int num_priors_actual);

    void nms_mx(const float *conf_data, const float *bboxes, const float *sizes,
                int *buffer, int *indices, int *detections, int num_priors_actual);

    std::vector<float> _decoded_bboxes;
    std::vector<int> _buffer;
    std::vector<int> _indices;
    std::vector<int> _detections_count;
    std::vector<float> _reordered_conf;
    std::vector<float> _bbox_sizes;
    std::vector<int> _num_priors_actual;
};

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
                                   const int idx2) {
    float xmin1 = decoded_bbox[idx1*4 + 0];
    float ymin1 = decoded_bbox[idx1*4 + 1];
    float xmax1 = decoded_bbox[idx1*4 + 2];
    float ymax1 = decoded_bbox[idx1*4 + 3];

    float xmin2 = decoded_bbox[idx2*4 + 0];
    float ymin2 = decoded_bbox[idx2*4 + 1];
    float xmax2 = decoded_bbox[idx2*4 + 2];
    float ymax2 = decoded_bbox[idx2*4 + 3];

    if (xmin2 > xmax1 || xmax2 < xmin1 || ymin2 > ymax1 || ymax2 < ymin1) {
        return 0.0f;
    }

    float intersect_xmin = (std::max)(xmin1, xmin2);
    float intersect_ymin = (std::max)(ymin1, ymin2);
    float intersect_xmax = (std::min)(xmax1, xmax2);
    float intersect_ymax = (std::min)(ymax1, ymax2);

    float intersect_width  = intersect_xmax - intersect_xmin;
    float intersect_height = intersect_ymax - intersect_ymin;

    if (intersect_width <= 0 || intersect_height <= 0) {
        return 0.0f;
    }

    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = bbox_sizes[idx1];
    float bbox2_size = bbox_sizes[idx2];

    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
}

void DetectionOutputImpl::decodeBBoxes(const float *prior_data,
                                       const float *loc_data,
                                       const float *variance_data,
                                       float *decoded_bboxes,
                                       float *decoded_bbox_sizes,
                                       int* num_priors_actual,
                                       int n,
                                       const int& offs,
                                       const int& pr_size,
                                       bool decodeType) {
    num_priors_actual[n] = _num_priors;
    if (!_normalized && decodeType) {
        int num = 0;
        for (; num < _num_priors; ++num) {
            float batch_id = prior_data[num * pr_size + 0];
            if (batch_id == -1.f) {
                num_priors_actual[n] = num;
                break;
            }
        }
    }
    parallel_for(num_priors_actual[n], [&](int p) {
        float new_xmin = 0.0f;
        float new_ymin = 0.0f;
        float new_xmax = 0.0f;
        float new_ymax = 0.0f;

        float prior_xmin = prior_data[p*pr_size + 0 + offs];
        float prior_ymin = prior_data[p*pr_size + 1 + offs];
        float prior_xmax = prior_data[p*pr_size + 2 + offs];
        float prior_ymax = prior_data[p*pr_size + 3 + offs];

        float loc_xmin = loc_data[4*p*_num_loc_classes + 0];
        float loc_ymin = loc_data[4*p*_num_loc_classes + 1];
        float loc_xmax = loc_data[4*p*_num_loc_classes + 2];
        float loc_ymax = loc_data[4*p*_num_loc_classes + 3];

        if (!_normalized) {
            prior_xmin /= _image_width;
            prior_ymin /= _image_height;
            prior_xmax /= _image_width;
            prior_ymax /= _image_height;
        }

        if (_code_type == CodeType::CORNER) {
            if (_variance_encoded_in_target) {
                // variance is encoded in target, we simply need to add the offset predictions.
                new_xmin = prior_xmin + loc_xmin;
                new_ymin = prior_ymin + loc_ymin;
                new_xmax = prior_xmax + loc_xmax;
                new_ymax = prior_ymax + loc_ymax;
            } else {
                new_xmin = prior_xmin + variance_data[p*4 + 0] * loc_xmin;
                new_ymin = prior_ymin + variance_data[p*4 + 1] * loc_ymin;
                new_xmax = prior_xmax + variance_data[p*4 + 2] * loc_xmax;
                new_ymax = prior_ymax + variance_data[p*4 + 3] * loc_ymax;
            }
        } else if (_code_type == CodeType::CENTER_SIZE) {
            float prior_width    =  prior_xmax - prior_xmin;
            float prior_height   =  prior_ymax - prior_ymin;
            float prior_center_x = (prior_xmin + prior_xmax) / 2.0f;
            float prior_center_y = (prior_ymin + prior_ymax) / 2.0f;

            float decode_bbox_center_x, decode_bbox_center_y;
            float decode_bbox_width, decode_bbox_height;

            if (_variance_encoded_in_target) {
                // variance is encoded in target, we simply need to restore the offset predictions.
                decode_bbox_center_x = loc_xmin * prior_width  + prior_center_x;
                decode_bbox_center_y = loc_ymin * prior_height + prior_center_y;
                decode_bbox_width  = std::exp(loc_xmax) * prior_width;
                decode_bbox_height = std::exp(loc_ymax) * prior_height;
            } else {
                // variance is encoded in bbox, we need to scale the offset accordingly.
                decode_bbox_center_x = variance_data[p*4 + 0] * loc_xmin * prior_width + prior_center_x;
                decode_bbox_center_y = variance_data[p*4 + 1] * loc_ymin * prior_height + prior_center_y;
                decode_bbox_width    = std::exp(variance_data[p*4 + 2] * loc_xmax) * prior_width;
                decode_bbox_height   = std::exp(variance_data[p*4 + 3] * loc_ymax) * prior_height;
            }

            new_xmin = decode_bbox_center_x - decode_bbox_width  / 2.0f;
            new_ymin = decode_bbox_center_y - decode_bbox_height / 2.0f;
            new_xmax = decode_bbox_center_x + decode_bbox_width  / 2.0f;
            new_ymax = decode_bbox_center_y + decode_bbox_height / 2.0f;
        }

        if (_clip_before_nms) {
            new_xmin = (std::max)(0.0f, (std::min)(1.0f, new_xmin));
            new_ymin = (std::max)(0.0f, (std::min)(1.0f, new_ymin));
            new_xmax = (std::max)(0.0f, (std::min)(1.0f, new_xmax));
            new_ymax = (std::max)(0.0f, (std::min)(1.0f, new_ymax));
        }

        decoded_bboxes[p*4 + 0] = new_xmin;
        decoded_bboxes[p*4 + 1] = new_ymin;
        decoded_bboxes[p*4 + 2] = new_xmax;
        decoded_bboxes[p*4 + 3] = new_ymax;

        decoded_bbox_sizes[p] = (new_xmax - new_xmin) * (new_ymax - new_ymin);
    });
}

void DetectionOutputImpl::nms_cf(const float* conf_data,
                          const float* bboxes,
                          const float* sizes,
                          int* buffer,
                          int* indices,
                          int& detections,
                          int num_priors_actual) {
    int count = 0;
    for (int i = 0; i < num_priors_actual; ++i) {
        if (conf_data[i] > _confidence_threshold) {
            indices[count] = i;
            count++;
        }
    }

    int num_output_scores = (_top_k == -1 ? count : (std::min)(_top_k, count));

    std::partial_sort_copy(indices, indices + count,
                           buffer, buffer + num_output_scores,
                           ConfidenceComparator(conf_data));

    for (int i = 0; i < num_output_scores; ++i) {
        const int idx = buffer[i];

        bool keep = true;
        for (int k = 0; k < detections; ++k) {
            const int kept_idx = indices[k];
            float overlap = JaccardOverlap(bboxes, sizes, idx, kept_idx);
            if (overlap > _nms_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            indices[detections] = idx;
            detections++;
        }
    }
}

void DetectionOutputImpl::nms_mx(const float* conf_data,
                          const float* bboxes,
                          const float* sizes,
                          int* buffer,
                          int* indices,
                          int* detections,
                          int num_priors_actual) {
    int count = 0;
    for (int i = 0; i < num_priors_actual; ++i) {
        float conf = -1;
        int id = 0;
        for (int c = 1; c < _num_classes; ++c) {
            float temp = conf_data[c*_num_priors + i];
            if (temp > conf) {
                conf = temp;
                id = c;
            }
        }

        if (id > 0 && conf >= _confidence_threshold) {
            indices[count++] = id*_num_priors + i;
        }
    }

    int num_output_scores = (_top_k == -1 ? count : (std::min)(_top_k, count));

    std::partial_sort_copy(indices, indices + count,
                           buffer, buffer + num_output_scores,
                           ConfidenceComparator(conf_data));

    for (int i = 0; i < num_output_scores; ++i) {
        const int idx = buffer[i];
        const int cls = idx/_num_priors;
        const int prior = idx%_num_priors;

        int &ndetection = detections[cls];
        int *pindices = indices + cls*_num_priors;

        bool keep = true;
        for (int k = 0; k < ndetection; ++k) {
            const int kept_idx = pindices[k];
            float overlap = 0.0f;
            if (_share_location) {
                overlap = JaccardOverlap(bboxes, sizes, prior, kept_idx);
            } else {
                overlap = JaccardOverlap(bboxes, sizes, cls*_num_priors + prior, cls*_num_priors + kept_idx);
            }
            if (overlap > _nms_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            pindices[ndetection++] = prior;
        }
    }
}

REG_FACTORY_FOR(DetectionOutputImpl, DetectionOutput);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
