// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <csignal>
#include <ctime>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <numeric>
#include <cfloat>

#include "bbox_util.h"

static const float  PI = 3.14159265358979323846;

static const float  ORIENT_MAX = PI;
static const float  ORIENT_MIN = -PI;
static const float  ORIENT_RNG = (ORIENT_MAX-ORIENT_MIN);

template<typename Dtype>
Dtype get_orient_by_bin_index(Dtype bin_index, int num_orient_bins) {
    Dtype bin_size = (Dtype)ORIENT_RNG / (Dtype)num_orient_bins;
    Dtype orient = (Dtype)ORIENT_MIN + (bin_index+1)*bin_size;

    return orient;
}

// Explicit initialization.
template float get_orient_by_bin_index(float bin_index, int num_orient_bins);
template double get_orient_by_bin_index(double bin_index, int num_orient_bins);

float get_orient_by_class_index(int class_index, int num_orient_classes) {
    switch (class_index) {
        case 0:
            return -0.5f * PI;
        case 1:
            return 0.5f * PI;
        default:
            return 0;
    }
}

template<typename Dtype, typename ArrayDtype>
Dtype get_orientation_impl(const ArrayDtype& bin_vals, bool interpolate_orientation) {
    int max_index = -1;

    Dtype max_score = 0;
    int num_bins = bin_vals.size();
    for (int i=0; i < num_bins; i++) {
        if (bin_vals[i] > max_score) {
            max_score = bin_vals[i];
            max_index = i;
        }
    }

    if (num_bins == 3) {
        Dtype orient = get_orient_by_class_index(max_index, num_bins);
        return orient;
    }

    Dtype bin_index = 0;
    if (interpolate_orientation) {
        // to be implemented soon
        int left_index = ((max_index-1)+num_bins)%num_bins;
        int right_index = ((max_index+1))%num_bins;

        Dtype left_val = bin_vals[left_index];
        Dtype right_val = bin_vals[right_index];
        Dtype x2 = (Dtype)((right_val - left_val)/(2*(left_val+right_val-2*max_score)));
        bin_index = (Dtype)max_index - x2;
    } else {
        bin_index = (Dtype)max_index;
    }

    return get_orient_by_bin_index(bin_index, num_bins);
}

float get_orientation(const std::vector<float>& bin_vals, bool interpolate_orientation) {
    return get_orientation_impl<float, std::vector<float> >(bin_vals, interpolate_orientation);
}

template<typename Dtype>
Dtype get_orientation(const ArrayWrapper<Dtype>& bin_vals, bool interpolate_orientation) {
    return get_orientation_impl<Dtype, ArrayWrapper<Dtype> >(bin_vals, interpolate_orientation);
}

// Explicit initialization.
template float get_orientation(const ArrayWrapper<float>& bin_vals, bool interpolate_orientation);
template double get_orientation(const ArrayWrapper<double>& bin_vals, bool interpolate_orientation);

bool SortBBoxAscend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
    return bbox1.score < bbox2.score;
}

bool SortBBoxDescend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
    return bbox1.score > bbox2.score;
}

bool SortBBoxIndexPairDescend(const std::pair<NormalizedBBox, int>& bbox1, const std::pair<NormalizedBBox, int>& bbox2) {
    return SortBBoxDescend(bbox1.first, bbox2.first);
}

template <typename T>
bool SortDetectionResultPairAscend(const std::pair<DetectionResult, T>& pair1,
                                   const std::pair<DetectionResult, T>& pair2) {
    return pair1.first.mScore < pair2.first.mScore;
}

// Explicit initialization.
template <typename T>
bool SortDetectionResultPairDescend(const std::pair<DetectionResult, T>& pair1,
                                    const std::pair<DetectionResult, T>& pair2) {
    return pair1.first.mScore > pair2.first.mScore;
}

template <typename T>
bool SortScorePairAscend(const std::pair<float, T>& pair1,
                         const std::pair<float, T>& pair2) {
    return pair1.first < pair2.first;
}

// Explicit initialization.
template bool SortScorePairAscend(const std::pair<float, int>& pair1,
                                  const std::pair<float, int>& pair2);
template bool SortScorePairAscend(const std::pair<float, std::pair<int, int> >& pair1,
                                  const std::pair<float, std::pair<int, int> >& pair2);

template <typename T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
    return pair1.first > pair2.first;
}

template <typename T>
bool SortScorePairDescendStable(const std::pair<float, T>& pair1,
                                const std::pair<float, T>& pair2) {
    if (pair1.first > pair2.first) return true;
    if (pair1.first < pair2.first) return false;
    return pair1.second < pair2.second;
}

template <typename T>
struct ScoresIndexedComparator {
    explicit ScoresIndexedComparator(const std::vector<T>& scores) : _scores(scores) {}

    bool operator()(int idx1, int idx2) {
        if (_scores[idx1] > _scores[idx2]) return true;
        if (_scores[idx1] < _scores[idx2]) return false;
        return idx1 < idx2;
    }

    const std::vector<T>& _scores;
};


// Explicit initialization.
template bool SortDetectionResultPairDescend(const std::pair<DetectionResult, int>& pair1,
                                             const std::pair<DetectionResult, int>& pair2);
template bool SortDetectionResultPairDescend(const std::pair<DetectionResult, std::pair<int, int> >& pair1,
                                             const std::pair<DetectionResult, std::pair<int, int> >& pair2);

template bool SortScorePairDescend(const std::pair<float, int>& pair1,
                                   const std::pair<float, int>& pair2);
template bool SortScorePairDescend(const std::pair<float, std::pair<int, int> >& pair1,
                                   const std::pair<float, std::pair<int, int> >& pair2);


template bool SortScorePairDescendStable(const std::pair<float, int>& pair1,
                                         const std::pair<float, int>& pair2);
template bool SortScorePairDescendStable(const std::pair<float, std::pair<int, int> >& pair1,
                                         const std::pair<float, std::pair<int, int> >& pair2);

NormalizedBBox UnitBBox() {
    NormalizedBBox unit_bbox;
    unit_bbox.set_xmin(0.);
    unit_bbox.set_ymin(0.);
    unit_bbox.set_xmax(1.);
    unit_bbox.set_ymax(1.);
    return unit_bbox;
}

template<typename BBoxType>
void IntersectBBox_impl(const BBoxType& bbox1, const BBoxType& bbox2, BBoxType* intersect_bbox) {
    if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin() ||
        bbox2.ymin() > bbox1.ymax() || bbox2.ymax() < bbox1.ymin()) {
        // Return [0, 0, 0, 0] if there is no intersection.
        intersect_bbox->set_xmin(0);
        intersect_bbox->set_ymin(0);
        intersect_bbox->set_xmax(0);
        intersect_bbox->set_ymax(0);
    } else {
        intersect_bbox->set_xmin(std::max(bbox1.xmin(), bbox2.xmin()));
        intersect_bbox->set_ymin(std::max(bbox1.ymin(), bbox2.ymin()));
        intersect_bbox->set_xmax(std::min(bbox1.xmax(), bbox2.xmax()));
        intersect_bbox->set_ymax(std::min(bbox1.ymax(), bbox2.ymax()));
    }
}

void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2, NormalizedBBox* intersect_bbox) {
    IntersectBBox_impl(bbox1, bbox2, intersect_bbox);
}

template<typename Dtype>
void IntersectBBox(const BBox<Dtype>& bbox1, const BBox<Dtype>& bbox2, BBox<Dtype>* intersect_bbox) {
    IntersectBBox_impl(bbox1, bbox2, intersect_bbox);
}

// Explicit initialization.
template void IntersectBBox(const BBox<float>& bbox1, const BBox<float>& bbox2,
                            BBox<float>* intersect_bbox);
template void IntersectBBox(const BBox<double>& bbox1, const BBox<double>& bbox2,
                            BBox<double>* intersect_bbox);


template<typename Dtype, typename BBoxType>
Dtype BBoxSize_impl(const BBoxType& bbox, const bool normalized) {
    if (bbox.xmax() < bbox.xmin() || bbox.ymax() < bbox.ymin()) {
        // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
        return 0;
    }
    Dtype width = bbox.xmax() - bbox.xmin();
    Dtype height = bbox.ymax() - bbox.ymin();
    if (normalized) {
        return width * height;
    } else {
        // If bbox is not within range [0, 1].
        return (width + 1) * (height + 1);
    }
}

float BBoxSize(const NormalizedBBox& bbox, const bool normalized) {
    return BBoxSize_impl<float, NormalizedBBox>(bbox, normalized);
}

template<typename Dtype>
Dtype BBoxSize(const BBox<Dtype>& bbox, const bool normalized) {
    return BBoxSize_impl<Dtype, BBox<Dtype> >(bbox, normalized);
}

// Explicit initialization.
template float BBoxSize(const BBox<float>& bbox, const bool normalized);
template double BBoxSize(const BBox<double>& bbox, const bool normalized);

void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clip_bbox) {
    clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), 1.f), 0.f));
    clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), 1.f), 0.f));
    clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), 1.f), 0.f));
    clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), 1.f), 0.f));

    clip_bbox->set_size(BBoxSize(*clip_bbox));
    clip_bbox->difficult = bbox.difficult;
}

template<typename Dtype>
void ClipBBox(const BBox<Dtype>& bbox, BBox<Dtype>* clip_bbox) {
    clip_bbox->set_xmin(std::max<Dtype>(std::min<Dtype>(bbox.xmin(), 1.0), 0.0));
    clip_bbox->set_ymin(std::max<Dtype>(std::min<Dtype>(bbox.ymin(), 1.0), 0.0));
    clip_bbox->set_xmax(std::max<Dtype>(std::min<Dtype>(bbox.xmax(), 1.0), 0.0));
    clip_bbox->set_ymax(std::max<Dtype>(std::min<Dtype>(bbox.ymax(), 1.0), 0.0));
}

// Explicit initialization.
template void ClipBBox(const BBox<float>& bbox, BBox<float>* clip_bbox);
template void ClipBBox(const BBox<double>& bbox, BBox<double>* clip_bbox);

void ScaleBBox(const NormalizedBBox& bbox, const int height, const int width,
               NormalizedBBox* scale_bbox) {
    scale_bbox->set_xmin(bbox.xmin() * width);
    scale_bbox->set_ymin(bbox.ymin() * height);
    scale_bbox->set_xmax(bbox.xmax() * width);
    scale_bbox->set_ymax(bbox.ymax() * height);

    bool normalized = !(width > 1 || height > 1);

    scale_bbox->set_size(BBoxSize(*scale_bbox, normalized));
    scale_bbox->difficult = bbox.difficult;
}

template<typename Dtype>
void ScaleBBox(const BBox<Dtype>& bbox, const int height, const int width,
               BBox<Dtype>* scale_bbox) {
    scale_bbox->set_xmin(bbox.xmin() * width);
    scale_bbox->set_ymin(bbox.ymin() * height);
    scale_bbox->set_xmax(bbox.xmax() * width);
    scale_bbox->set_ymax(bbox.ymax() * height);
}

// Explicit initialization.
template void ScaleBBox(const BBox<float>& bbox, const int height, const int width,
                        BBox<float>* scale_bbox);
template void ScaleBBox(const BBox<double>& bbox, const int height, const int width,
                        BBox<double>* scale_bbox);

void LocateBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox, NormalizedBBox* loc_bbox) {
    float src_width = src_bbox.xmax() - src_bbox.xmin();
    float src_height = src_bbox.ymax() - src_bbox.ymin();
    loc_bbox->set_xmin(src_bbox.xmin() + bbox.xmin() * src_width);
    loc_bbox->set_ymin(src_bbox.ymin() + bbox.ymin() * src_height);
    loc_bbox->set_xmax(src_bbox.xmin() + bbox.xmax() * src_width);
    loc_bbox->set_ymax(src_bbox.ymin() + bbox.ymax() * src_height);
    loc_bbox->difficult = bbox.difficult;
    loc_bbox->orientation = src_bbox.orientation;
}


bool ProjectBBox_GetRatio(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox, NormalizedBBox* proj_bbox, float& ratio) {
    if (bbox.xmin() >= src_bbox.xmax() || bbox.xmax() <= src_bbox.xmin() ||
        bbox.ymin() >= src_bbox.ymax() || bbox.ymax() <= src_bbox.ymin() ||
        bbox.xmax() <= bbox.xmin() || bbox.ymax() <= bbox.ymin()) {
        return false;
    }

    float src_width = src_bbox.xmax() - src_bbox.xmin();
    float src_height = src_bbox.ymax() - src_bbox.ymin();

    proj_bbox->set_xmin((bbox.xmin() - src_bbox.xmin()) / src_width);
    proj_bbox->set_ymin((bbox.ymin() - src_bbox.ymin()) / src_height);
    proj_bbox->set_xmax((bbox.xmax() - src_bbox.xmin()) / src_width);
    proj_bbox->set_ymax((bbox.ymax() - src_bbox.ymin()) / src_height);
    proj_bbox->difficult = bbox.difficult;
    proj_bbox->orientation = bbox.orientation;

    float area_before_clipping = BBoxSize(*proj_bbox);
    ClipBBox(*proj_bbox, proj_bbox);

    float area_after_clipping = BBoxSize(*proj_bbox);
    ratio = area_after_clipping/area_before_clipping;

    if (BBoxSize(*proj_bbox) > 0) {
        return true;
    } else {
        return false;
    }
}


bool ProjectBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
                 NormalizedBBox* proj_bbox) {
    float ratio = 0;
    return ProjectBBox_GetRatio(src_bbox, bbox, proj_bbox, ratio);
}

template<typename Dtype, typename BBoxType>
Dtype JaccardOverlap_impl(const BBoxType& bbox1, const BBoxType& bbox2,
                          const bool normalized) {
    BBoxType intersect_bbox;
    IntersectBBox(bbox1, bbox2, &intersect_bbox);
    Dtype intersect_width, intersect_height;
    if (normalized) {
        intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
        intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
    } else {
        intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin() + 1;
        intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin() + 1;
    }
    if (intersect_width > 0 && intersect_height > 0) {
        Dtype intersect_size = intersect_width * intersect_height;
        Dtype bbox1_size = BBoxSize(bbox1);
        Dtype bbox2_size = BBoxSize(bbox2);
        return intersect_size / (bbox1_size + bbox2_size - intersect_size);
    } else {
        return 0.;
    }
}

float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                     const bool normalized) {
    return JaccardOverlap_impl<float, NormalizedBBox>(bbox1, bbox2, normalized);
}

template<typename Dtype>
Dtype JaccardOverlap(const BBox<Dtype>& bbox1, const BBox<Dtype>& bbox2,
                     const bool normalized) {
    return JaccardOverlap_impl<Dtype, BBox<Dtype> >(bbox1, bbox2, normalized);
}

// Explicit initialization.
template float JaccardOverlap(const BBox<float>& bbox1, const BBox<float>& bbox2,
                              const bool normalized);
template double JaccardOverlap(const BBox<double>& bbox1, const BBox<double>& bbox2,
                               const bool normalized);

float BBoxCoverage(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
    NormalizedBBox intersect_bbox;
    IntersectBBox(bbox1, bbox2, &intersect_bbox);
    float intersect_size = BBoxSize(intersect_bbox);
    float bbox1_size = BBoxSize(bbox1);
    if (intersect_size > 0 && bbox1_size > 0) {
        return intersect_size / bbox1_size;
    } else {
        return 0.;
    }
}

void DecodeBBox(const NormalizedBBox& prior_bbox, const float* prior_variance,
                const CodeType code_type, const bool variance_encoded_in_target,
                const NormalizedBBox& bbox, NormalizedBBox* decode_bbox) {
    if (code_type == CodeType::CORNER) {
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to add the offset
            // predictions.
            decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin());
            decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin());
            decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax());
            decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax());
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox->set_xmin(
                                  prior_bbox.xmin() + prior_variance[0] * bbox.xmin());
            decode_bbox->set_ymin(
                                  prior_bbox.ymin() + prior_variance[1] * bbox.ymin());
            decode_bbox->set_xmax(
                                  prior_bbox.xmax() + prior_variance[2] * bbox.xmax());
            decode_bbox->set_ymax(
                                  prior_bbox.ymax() + prior_variance[3] * bbox.ymax());
        }
    } else if (code_type == CodeType::CENTER_SIZE) {
        float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
        float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
        float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
        float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;

        float decode_bbox_center_x, decode_bbox_center_y;
        float decode_bbox_width, decode_bbox_height;

        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to retore the offset
            // predictions.
            decode_bbox_center_x = bbox.xmin() * prior_width + prior_center_x;
            decode_bbox_center_y = bbox.ymin() * prior_height + prior_center_y;
            decode_bbox_width = exp(bbox.xmax()) * prior_width;
            decode_bbox_height = exp(bbox.ymax()) * prior_height;
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox_center_x =
            prior_variance[0] * bbox.xmin() * prior_width + prior_center_x;
            decode_bbox_center_y =
            prior_variance[1] * bbox.ymin() * prior_height + prior_center_y;
            decode_bbox_width =
            exp(prior_variance[2] * bbox.xmax()) * prior_width;
            decode_bbox_height =
            exp(prior_variance[3] * bbox.ymax()) * prior_height;
        }

        decode_bbox->set_xmin(decode_bbox_center_x - decode_bbox_width / 2.);
        decode_bbox->set_ymin(decode_bbox_center_y - decode_bbox_height / 2.);
        decode_bbox->set_xmax(decode_bbox_center_x + decode_bbox_width / 2.);
        decode_bbox->set_ymax(decode_bbox_center_y + decode_bbox_height / 2.);
    }

    decode_bbox->set_size(BBoxSize(*decode_bbox));
}

void DecodeBBoxes(const std::vector<NormalizedBBox>& prior_bboxes,
                  const std::vector<float>& prior_variances,
                  const CodeType code_type, const bool variance_encoded_in_target,
                  const std::vector<NormalizedBBox>& bboxes,
                  std::vector<NormalizedBBox>* decode_bboxes) {
    int num_bboxes = prior_bboxes.size();

    decode_bboxes->clear();
    decode_bboxes->resize(num_bboxes);

    for (int i = 0; i < num_bboxes; ++i) {
        NormalizedBBox decode_bbox;
        DecodeBBox(prior_bboxes[i], &prior_variances[i * 4], code_type,
                   variance_encoded_in_target, bboxes[i], &decode_bbox);
        (*decode_bboxes)[i] = decode_bbox;
    }
}

void DecodeBBoxesAll(const std::vector<LabelBBox>& all_loc_preds,
                     const std::vector<NormalizedBBox>& prior_bboxes,
                     const std::vector<float>& prior_variances,
                     const int num, const bool share_location,
                     const int num_loc_classes, const int background_label_id,
                     const CodeType code_type, const bool variance_encoded_in_target,
                     std::vector<LabelBBox>* all_decode_bboxes) {
    all_decode_bboxes->clear();
    all_decode_bboxes->resize(num);

    for (int i = 0; i < num; ++i) {
        // Decode predictions into bboxes.
        LabelBBox& decode_bboxes = (*all_decode_bboxes)[i];
        for (int c = 0; c < num_loc_classes; ++c) {
            int label = share_location ? -1 : c;

            if (label == background_label_id) {
                // Ignore background class.
                continue;
            }

            if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
                continue;
            }

            const std::vector<NormalizedBBox>& label_loc_preds = all_loc_preds[i].find(label)->second;

            DecodeBBoxes(prior_bboxes, prior_variances,
                         code_type, variance_encoded_in_target,
                         label_loc_preds, &(decode_bboxes[label]));
        }
    }
}

template <typename Dtype>
void GetLocPredictions(const Dtype* loc_data, const int num,
                       const int num_preds_per_class, const int num_loc_classes,
                       const bool share_location, std::vector<LabelBBox>* loc_preds) {
    loc_preds->clear();
    loc_preds->resize(num);

    for (int i = 0; i < num; ++i) {
        LabelBBox& label_bbox = (*loc_preds)[i];

        for (int c = 0; c < num_loc_classes; ++c) {
            std::vector<NormalizedBBox>& bbox_vec = (share_location ? label_bbox[-1] : label_bbox[c]);
            bbox_vec.resize(num_preds_per_class);

            for (int p = 0; p < num_preds_per_class; ++p) {
                int start_idx = p * num_loc_classes * 4;
                bbox_vec[p].set_xmin(loc_data[start_idx + c * 4]);
                bbox_vec[p].set_ymin(loc_data[start_idx + c * 4 + 1]);
                bbox_vec[p].set_xmax(loc_data[start_idx + c * 4 + 2]);
                bbox_vec[p].set_ymax(loc_data[start_idx + c * 4 + 3]);
            }
        }
        loc_data += num_preds_per_class * num_loc_classes * 4;
    }
}

template <typename Dtype>
void GetLocPredictions(const Dtype* loc_data, const int num,
                       const int num_preds_per_class, const int num_loc_classes,
                       const bool share_location, std::vector<std::map<int, BBoxArrayWrapper<Dtype> > >* loc_preds) {
    loc_preds->clear();
    loc_preds->resize(num);

    const BBox<Dtype>* bbox_data = reinterpret_cast<const BBox<Dtype>*>(loc_data);

    for (int i = 0; i < num; ++i) {
        std::map<int, BBoxArrayWrapper<Dtype> > & label_bbox = (*loc_preds)[i];

        if (share_location) {
            label_bbox[-1] = BBoxArrayWrapper<Dtype>(&bbox_data[i * num_preds_per_class * num_loc_classes], num_preds_per_class);
        } else {
            for (int c = 0; c < num_loc_classes; ++c) {
                label_bbox[c] = BBoxArrayWrapper<Dtype>(&bbox_data[(i * num_loc_classes + c) * num_preds_per_class], num_preds_per_class);
            }
        }
    }
}

// Explicit initialization.
template void GetLocPredictions(const float* loc_data, const int num,
                                const int num_preds_per_class, const int num_loc_classes,
                                const bool share_location, std::vector<LabelBBox>* loc_preds);
template void GetLocPredictions(const double* loc_data, const int num,
                                const int num_preds_per_class, const int num_loc_classes,
                                const bool share_location, std::vector<LabelBBox>* loc_preds);

template void GetLocPredictions(const float* loc_data, const int num,
                                const int num_preds_per_class, const int num_loc_classes,
                                const bool share_location, std::vector<std::map<int, BBoxArrayWrapper<float>>>* loc_preds);
template void GetLocPredictions(const double* loc_data, const int num,
                                const int num_preds_per_class, const int num_loc_classes,
                                const bool share_location, std::vector<std::map<int, BBoxArrayWrapper<double>>>* loc_preds);

template <typename Dtype>
void GetOrientPredictions(const Dtype* orient_data, const int num, std::vector<float>* orient_preds) {
    orient_preds->clear();
    orient_preds->resize(num);

    for (int i = 0; i < num; ++i) {
        (*orient_preds)[i] = orient_data[i];
    }
}

// Explicit initialization.
template void GetOrientPredictions(const float* orient_data, const int num,
                                   std::vector<float>* orient_preds);
template void GetOrientPredictions(const double* orient_data, const int num,
                                   std::vector<float>* orient_preds);

template <typename Dtype>
void GetOrientationScores(const Dtype* orient_data, const int num,
                          const int num_priors, const int num_orient_classes,
                          std::vector<std::vector<std::vector<float> > >* orient_preds) {
    orient_preds->clear();
    orient_preds->resize(num);
    for (int i = 0; i < num; ++i) {
        std::vector<std::vector<float> >& orient_scores = (*orient_preds)[i];

        orient_scores.resize(num_priors);
        for (int p = 0; p < num_priors; ++p) {
            int start_idx = p * num_orient_classes;
            orient_scores[p].resize(num_orient_classes);
            for (int c = 0; c < num_orient_classes; ++c) {
                orient_scores[p][c] = orient_data[start_idx + c];
            }
        }
        orient_data += num_priors * num_orient_classes;
    }
}

template <typename Dtype>
void GetOrientationScores(const Dtype* orient_data, const int num,
                          const int num_priors, const int num_orient_classes,
                          std::vector<std::vector<ArrayWrapper<Dtype> > >* orient_preds) {
    orient_preds->clear();
    orient_preds->resize(num);
    for (int i = 0; i < num; ++i) {
        std::vector<ArrayWrapper<Dtype> >& orient_scores = (*orient_preds)[i];
        orient_scores.resize(num_priors);
        for (int p = 0; p < num_priors; ++p) {
            int start_idx = p * num_orient_classes;
            orient_scores[p] = ArrayWrapper<Dtype>(&orient_data[start_idx], num_orient_classes);
        }
        orient_data += num_priors * num_orient_classes;
    }
}

// Explicit initialization.
template void GetOrientationScores(const float* orient_data, const int num,
                                   const int num_priors, const int num_orient_classes,
                                   std::vector<std::vector<std::vector<float>>>* orient_preds);
template void GetOrientationScores(const double* orient_data, const int num,
                                   const int num_priors, const int num_orient_classes,
                                   std::vector<std::vector<std::vector<float>>>* orient_preds);

template void GetOrientationScores(const float* orient_data, const int num,
                                   const int num_priors, const int num_orient_classes,
                                   std::vector<std::vector<ArrayWrapper<float>>>* orient_preds);
template void GetOrientationScores(const double* orient_data, const int num,
                                   const int num_priors, const int num_orient_classes,
                                   std::vector<std::vector<ArrayWrapper<double>>>* orient_preds);

template <typename Dtype>
void GetConfidenceScores(const Dtype* conf_data, const int num,
                         const int num_preds_per_class, const int num_classes,
                         std::vector<std::map<int, std::vector<Dtype> > >* conf_preds) {
    bool class_major = false;
    GetConfidenceScores(conf_data, num, num_preds_per_class, num_classes, class_major, conf_preds);
}

template <typename Dtype>
void GetConfidenceScores(const Dtype* conf_data, const int num,
                         const int num_preds_per_class, const int num_classes,
                         const bool class_major, std::vector<std::map<int, std::vector<Dtype>>>* conf_preds) {
    conf_preds->clear();
    conf_preds->resize(num);

    for (int i = 0; i < num; ++i) {
        std::map<int, std::vector<Dtype> >& label_scores = (*conf_preds)[i];

        if (class_major) {
            for (int c = 0; c < num_classes; ++c) {
                label_scores[c].assign(conf_data, conf_data + num_preds_per_class);
                conf_data += num_preds_per_class;
            }
        } else {
            for (int p = 0; p < num_preds_per_class; ++p) {
                int start_idx = p * num_classes;
                for (int c = 0; c < num_classes; ++c) {
                    label_scores[c].push_back(conf_data[start_idx + c]);
                }
            }
            conf_data += num_preds_per_class * num_classes;
        }
    }
}

// Explicit initialization.
template void GetConfidenceScores(const float* conf_data, const int num,
                                  const int num_preds_per_class, const int num_classes,
                                  std::vector<std::map<int, std::vector<float> > >* conf_preds);
template void GetConfidenceScores(const double* conf_data, const int num,
                                  const int num_preds_per_class, const int num_classes,
                                  std::vector<std::map<int, std::vector<double> > >* conf_preds);
template void GetConfidenceScores(const float* conf_data, const int num,
                                  const int num_preds_per_class, const int num_classes,
                                  const bool class_major, std::vector<std::map<int, std::vector<float> > >* conf_preds);
template void GetConfidenceScores(const double* conf_data, const int num,
                                  const int num_preds_per_class, const int num_classes,
                                  const bool class_major, std::vector<std::map<int, std::vector<double>>>* conf_preds);

template <typename Dtype>
void GetMaxConfidenceScores(const Dtype* conf_data, const int num,
                            const int num_preds_per_class, const int num_classes,
                            const int background_label_id, const ConfLossType loss_type,
                            std::vector<std::vector<float> >* all_max_scores) {
    all_max_scores->clear();
    for (int image_index_in_batch = 0; image_index_in_batch < num; ++image_index_in_batch) {
        std::vector<float> max_scores;
        for (int prior_box_index = 0; prior_box_index < num_preds_per_class; ++prior_box_index) {
            int start_idx = prior_box_index * num_classes;
            Dtype maxval = -FLT_MAX;
            Dtype maxval_pos = -FLT_MAX;
            for (int label_index = 0; label_index < num_classes; ++label_index) {
                maxval = std::max<Dtype>(conf_data[start_idx + label_index], maxval);
                if (label_index != background_label_id) {
                    // Find maximum scores for positive classes.
                    maxval_pos = std::max<Dtype>(conf_data[start_idx + label_index], maxval_pos);
                }
            }


            if (loss_type == ConfLossType::SOFTMAX) {
                // Compute softmax probability.
                Dtype sum = 0.;
                for (int label_index = 0; label_index < num_classes; ++label_index) {
                    sum += std::exp(conf_data[start_idx + label_index] - maxval);
                }
                maxval_pos = std::exp(maxval_pos - maxval) / sum;
            } else if (loss_type == ConfLossType::HINGE) {
            } else if (loss_type == ConfLossType::LOGISTIC) {
                maxval_pos = 1. / (1. + exp(-maxval_pos));
            }

            max_scores.push_back(maxval_pos);
        }
        conf_data += num_preds_per_class * num_classes;
        all_max_scores->push_back(max_scores);
    }
}

// Explicit initialization.
template void GetMaxConfidenceScores(const float* conf_data, const int num,
                                     const int num_preds_per_class, const int num_classes,
                                     const int background_label_id, const ConfLossType loss_type,
                                     std::vector<std::vector<float> >* all_max_scores);
template void GetMaxConfidenceScores(const double* conf_data, const int num,
                                     const int num_preds_per_class, const int num_classes,
                                     const int background_label_id, const ConfLossType loss_type,
                                     std::vector<std::vector<float> >* all_max_scores);

template <typename Dtype>
void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
                    std::vector<NormalizedBBox>& prior_bboxes,
                    std::vector<float>& prior_variances) {
    for (int i = 0; i < num_priors; ++i) {
        int start_idx = i * 4;
        NormalizedBBox bbox;
        bbox.set_xmin(prior_data[start_idx]);
        bbox.set_ymin(prior_data[start_idx + 1]);
        bbox.set_xmax(prior_data[start_idx + 2]);
        bbox.set_ymax(prior_data[start_idx + 3]);
        float bbox_size = BBoxSize(bbox);
        bbox.set_size(bbox_size);
        prior_bboxes[i] = bbox;
    }

    for (int i = 0; i < num_priors; ++i) {
        int start_idx = (num_priors + i) * 4;

        prior_variances[i * 4 + 0] = prior_data[start_idx + 0];
        prior_variances[i * 4 + 1] = prior_data[start_idx + 1];
        prior_variances[i * 4 + 2] = prior_data[start_idx + 2];
        prior_variances[i * 4 + 3] = prior_data[start_idx + 3];
    }
}

// Explicit initialization.
template void GetPriorBBoxes(const float* prior_data, const int num_priors,
                             std::vector<NormalizedBBox>& prior_bboxes,
                             std::vector<float >& prior_variances);
template void GetPriorBBoxes(const double* prior_data, const int num_priors,
                             std::vector<NormalizedBBox>& prior_bboxes,
                             std::vector<float >& prior_variances);

template<typename Dtype>
void GetTopKScoreIndex(const std::vector<Dtype>& scores, const std::vector<int>& indices,
                       const int top_k, std::vector<std::pair<Dtype, int>>* score_index_vec) {
    int num_output_scores = (top_k == -1 ? scores.size() : std::min<int>(top_k, scores.size()));

    std::vector<int> sorted_indices(num_output_scores);

    std::partial_sort_copy(indices.begin(), indices.end(),
                           sorted_indices.begin(), sorted_indices.end(),
                           ScoresIndexedComparator<Dtype>(scores));

    score_index_vec->reserve(num_output_scores);

    for (int i = 0; i < num_output_scores; i++) {
        int idx = sorted_indices[i];
        (*score_index_vec).push_back(std::make_pair(scores[idx], idx));
    }
}

// Explicit initialization.
template void GetTopKScoreIndex(const std::vector<float>& scores, const std::vector<int>& indices,
                                const int top_k, std::vector<std::pair<float, int> >* score_index_vec);
template void GetTopKScoreIndex(const std::vector<double>& scores, const std::vector<int>& indices,
                                const int top_k, std::vector<std::pair<double, int> >* score_index_vec);

template<typename Dtype>
void GetMaxScoreIndex(const std::vector<Dtype>& scores, const float threshold,
                      const int top_k, std::vector<std::pair<Dtype, int> >* score_index_vec) {
    std::vector<int> indices(scores.size());
    for (int i = 0; i < scores.size(); ++i) {
        indices[i] = i;
    }

    GetTopKScoreIndex(scores, indices, top_k, score_index_vec);

    // trim output values smaller or equal to the threshold, if exist
    for (int i = 0; i < score_index_vec->size(); i++) {
        Dtype score = (*score_index_vec)[i].first;
        if (score <= threshold) {
            score_index_vec->resize(i);
            break;
        }
    }
}

// Explicit initialization.
template void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold,
                               const int top_k, std::vector<std::pair<float, int> >* score_index_vec);
template void GetMaxScoreIndex(const std::vector<double>& scores, const float threshold,
                               const int top_k, std::vector<std::pair<double, int> >* score_index_vec);

void ApplyNMS(const std::vector<NormalizedBBox>& bboxes, const std::vector<float>& scores,
              const float threshold, const int top_k, const bool reuse_overlaps,
              std::map<int, std::map<int, float> >* overlaps, std::vector<int>* indices) {
    std::vector<int> idx;

    for (int  i = 0; i < scores.size(); i++) {
        idx.push_back(i);
    }

    std::vector<std::pair<float, int> > score_index_vec;

    GetTopKScoreIndex(scores, idx, top_k, &score_index_vec);

    // Do nms.
    indices->clear();

    while (score_index_vec.size() != 0) {
        // Get the current highest score box.
        int best_idx = score_index_vec.front().second;
        const NormalizedBBox& best_bbox = bboxes[best_idx];
        if (BBoxSize(best_bbox) < 1e-5) {
            // Erase small box.
            score_index_vec.erase(score_index_vec.begin());
            continue;
        }

        indices->push_back(best_idx);
        // Erase the best box.
        score_index_vec.erase(score_index_vec.begin());

        if (top_k > -1 && indices->size() >= top_k) {
            // Stop if finding enough bboxes for nms.
            break;
        }

        // Compute overlap between best_bbox and other remaining bboxes.
        // Remove a bbox if the overlap with best_bbox is larger than nms_threshold.
        for (std::vector<std::pair<float, int> >::iterator it = score_index_vec.begin();
             it != score_index_vec.end(); ) {
            int cur_idx = it->second;
            const NormalizedBBox& cur_bbox = bboxes[cur_idx];
            if (BBoxSize(cur_bbox) < 1e-5) {
                // Erase small box.
                it = score_index_vec.erase(it);
                continue;
            }
            float cur_overlap = 0.;
            if (reuse_overlaps) {
                if (overlaps->find(best_idx) != overlaps->end() &&
                    overlaps->find(best_idx)->second.find(cur_idx) !=
                    (*overlaps)[best_idx].end()) {
                    // Use the computed overlap.
                    cur_overlap = (*overlaps)[best_idx][cur_idx];
                } else if (overlaps->find(cur_idx) != overlaps->end() &&
                           overlaps->find(cur_idx)->second.find(best_idx) !=
                           (*overlaps)[cur_idx].end()) {
                    // Use the computed overlap.
                    cur_overlap = (*overlaps)[cur_idx][best_idx];
                } else {
                    cur_overlap = JaccardOverlap(best_bbox, cur_bbox);
                    // Store the overlap for future use.
                    (*overlaps)[best_idx][cur_idx] = cur_overlap;
                }
            } else {
                cur_overlap = JaccardOverlap(best_bbox, cur_bbox);
            }

            // Remove it if necessary
            if (cur_overlap > threshold) {
                it = score_index_vec.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void ApplyNMS(const bool* overlapped, const int num, std::vector<int>* indices) {
    std::vector<int> index_vec;

    for (int i = 0; i < num; i++) {
        index_vec.push_back(i);
    }

    // Do nms.
    indices->clear();

    while (index_vec.size() != 0) {
        // Get the current highest score box.
        int best_idx = index_vec.front();
        indices->push_back(best_idx);
        // Erase the best box.
        index_vec.erase(index_vec.begin());

        for (std::vector<int>::iterator it = index_vec.begin(); it != index_vec.end();) {
            int cur_idx = *it;

            // Remove it if necessary
            if (overlapped[best_idx * num + cur_idx]) {
                it = index_vec.erase(it);
            } else {
                ++it;
            }
        }
    }
}

template<typename Dtype, typename ArrayOfBBox>
void ApplyNMSFast_impl(const ArrayOfBBox& bboxes,
                       const std::vector<Dtype>& scores, const float score_threshold,
                       const float nms_threshold, const int top_k, std::vector<int>* indices) {
    std::vector<std::pair<Dtype, int> > score_index_vec;

    GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);

    indices->clear();

    while (score_index_vec.size() != 0) {
        const int idx = score_index_vec.front().second;
        bool keep = true;

        for (int k = 0; k < indices->size(); ++k) {
            if (keep) {
                const int kept_idx = (*indices)[k];
                float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
                keep = overlap <= nms_threshold;
            } else {
                break;
            }
        }
        if (keep) {
            indices->push_back(idx);
        }

        score_index_vec.erase(score_index_vec.begin());
    }
}

template<typename Dtype>
void ApplyNMSFast(const std::vector<NormalizedBBox>& bboxes,
                  const std::vector<Dtype>& scores, const float score_threshold,
                  const float nms_threshold, const int top_k, std::vector<int>* indices) {
    ApplyNMSFast_impl<Dtype, std::vector<NormalizedBBox>>(bboxes, scores, score_threshold,
                                                           nms_threshold, top_k, indices);
}

template<typename Dtype>
void ApplyNMSFast(const BBoxArrayWrapper<Dtype>& bboxes,
                  const std::vector<Dtype>& scores, const float score_threshold,
                  const float nms_threshold, const int top_k, std::vector<int>* indices) {
    ApplyNMSFast_impl<Dtype, BBoxArrayWrapper<Dtype>>(
                                                      bboxes, scores, score_threshold,
                                                      nms_threshold, top_k, indices);
}

// Explicit initialization.
template void ApplyNMSFast(const std::vector<NormalizedBBox>& bboxes,
                           const std::vector<float>& scores, const float score_threshold,
                           const float nms_threshold, const int top_k, std::vector<int>* indices);

template void ApplyNMSFast(const std::vector<NormalizedBBox>& bboxes,
                           const std::vector<double>& scores, const float score_threshold,
                           const float nms_threshold, const int top_k, std::vector<int>* indices);

template void ApplyNMSFast(const BBoxArrayWrapper<float>& bboxes,
                           const std::vector<float>& scores, const float score_threshold,
                           const float nms_threshold, const int top_k, std::vector<int>* indices);

template void ApplyNMSFast(const BBoxArrayWrapper<double>& bboxes,
                           const std::vector<double>& scores, const float score_threshold,
                           const float nms_threshold, const int top_k, std::vector<int>* indices);

