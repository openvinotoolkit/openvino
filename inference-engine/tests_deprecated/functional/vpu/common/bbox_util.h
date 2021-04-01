// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

enum CodeType {
    CORNER = 1,
    CENTER_SIZE = 2,
};

enum EmitType {
    CENTER = 0,
    MIN_OVERLAP = 1,
};

enum ConfLossType {
    SOFTMAX = 0,
    LOGISTIC = 1,
    HINGE = 2,
    PIECEWISE_LINEAR = 3,
};

struct NormalizedBBox {
public:
    NormalizedBBox() {
        reliable_location = true;
        orientation = -10;
    }

    void set_xmin(const float& v) { _xmin = v; }
    void set_ymin(const float& v) { _ymin = v; }
    void set_xmax(const float& v) { _xmax = v; }
    void set_ymax(const float& v) { _ymax = v; }

    void set_size(const float& v) { _size = v; }

    const float& xmin() const { return _xmin; }
    const float& ymin() const { return _ymin; }
    const float& xmax() const { return _xmax; }
    const float& ymax() const { return _ymax; }

    float _xmin = 0;
    float _ymin = 0;
    float _xmax = 0;
    float _ymax = 0;

    int label = 0;
    bool difficult = false;
    float score = 0;
    float _size = 0;
    bool reliable_location = false;
    float orientation = 0;
};

typedef std::map<int, std::vector<NormalizedBBox> > LabelBBox;

template <typename Dtype>
struct BBox {
  Dtype data[4];
  Dtype& xmin() { return data[0]; }
  Dtype& ymin() { return data[1]; }
  Dtype& xmax() { return data[2]; }
  Dtype& ymax() { return data[3]; }

  const Dtype& xmin() const { return data[0]; }
  const Dtype& ymin() const { return data[1]; }
  const Dtype& xmax() const { return data[2]; }
  const Dtype& ymax() const { return data[3]; }

  void set_xmin(const Dtype& v) { xmin() = v; }
  void set_ymin(const Dtype& v) { ymin() = v; }
  void set_xmax(const Dtype& v) { xmax() = v; }
  void set_ymax(const Dtype& v) { ymax() = v; }
};

template <typename T>
struct ArrayWrapper {
    ArrayWrapper() : _data(NULL), _count(0) {}
    ArrayWrapper(const T* data, int count) : _data(data), _count(count) {}
    ArrayWrapper(const ArrayWrapper& other) : _data(other._data), _count(other._count) {}
    size_t size() const { return _count; }
    size_t sizeb() const { return _count * sizeof(T); }

    inline const T& operator[](int i) const {
        return _data[i];
    }

    const T* _data;
    int _count;
};

template <typename Dtype>
struct BBoxArrayWrapper : public ArrayWrapper<BBox<Dtype> > {
    BBoxArrayWrapper() {}
    BBoxArrayWrapper(const BBox<Dtype>* data, int count) : ArrayWrapper<BBox<Dtype> >(data, count) {}
    BBoxArrayWrapper(const BBoxArrayWrapper& other) : ArrayWrapper<BBox<Dtype> >(other) {}
};

class DetectionResult {
public:
    explicit DetectionResult(float score = 0, float odiff = -10.f) : mScore(score), mOrientationDiff(odiff) {}
    float mScore;
    float mOrientationDiff;
};

// Compute the jaccard (intersection over union IoU) overlap between two bboxes.
float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                     const bool normalized = true);

template<typename Dtype>
Dtype JaccardOverlap(const BBox<Dtype>& bbox1, const BBox<Dtype>& bbox2,
                     const bool normalized = true);

// Compute bbox size.
float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true);

template<typename Dtype>
Dtype BBoxSize(const BBox<Dtype>& bbox, const bool normalized = true);

// Get location predictions from loc_data.
//    loc_data: num x num_preds_per_class * num_loc_classes * 4 blob.
//    num: the number of images.
//    num_preds_per_class: number of predictions per class.
//    num_loc_classes: number of location classes. It is 1 if share_location is
//      true; and is equal to number of classes needed to predict otherwise.
//    share_location: if true, all classes share the same location prediction.
//    loc_preds: stores the location prediction, where each item contains
//      location prediction for an image.
template <typename Dtype>
void GetLocPredictions(const Dtype* loc_data, const int num,
                       const int num_preds_per_class, const int num_loc_classes,
                       const bool share_location, std::vector<LabelBBox>* loc_preds);

template <typename Dtype>
void GetLocPredictions(const Dtype* loc_data, const int num,
                       const int num_preds_per_class, const int num_loc_classes,
                       const bool share_location, std::vector<std::map<int, BBoxArrayWrapper<Dtype>>>* loc_preds);

// Get confidence predictions from conf_data.
//    conf_data: num x num_preds_per_class * num_classes blob.
//    num: the number of images.
//    num_preds_per_class: number of predictions per class.
//    num_classes: number of classes.
//    conf_scores: stores the confidence prediction, where each item contains
//      confidence prediction for an image.
template <typename Dtype>
void GetConfidenceScores(const Dtype* conf_data, const int num,
                         const int num_preds_per_class, const int num_classes,
                         std::vector<std::map<int, std::vector<Dtype> > >* conf_scores);

// Get confidence predictions from conf_data.
//    conf_data: num x num_preds_per_class * num_classes blob.
//    num: the number of images.
//    num_preds_per_class: number of predictions per class.
//    num_classes: number of classes.
//    class_major: if true, data layout is
//      num x num_classes x num_preds_per_class; otherwise, data layerout is
//      num x num_preds_per_class * num_classes.
//    conf_scores: stores the confidence prediction, where each item contains
//      confidence prediction for an image.
template <typename Dtype>
void GetConfidenceScores(const Dtype* conf_data, const int num,
                         const int num_preds_per_class, const int num_classes,
                         const bool class_major, std::vector<std::map<int, std::vector<Dtype>>>* conf_scores);

// Get prior bounding boxes from prior_data.
//    prior_data: 1 x 2 x num_priors * 4 x 1 blob.
//    num_priors: number of priors.
//    prior_bboxes: stores all the prior bboxes in the format of NormalizedBBox.
//    prior_variances: stores all the variances needed by prior bboxes.
template <typename Dtype>
void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
                    std::vector<NormalizedBBox>& prior_bboxes,
                    std::vector<float>& prior_variances);

// Get confidence predictions from conf_data.
//    orient_data: num x num_priors * num_orient_classes blob.
//    num: the number of images.
//    num_priors: number of priors.
//    num_orient_classes: number of orientation classes (bins).
//    orient_preds: stores the orientation pr. prediction, where each item contains
//      predictions for an image.
template <typename Dtype>
void GetOrientationScores(const Dtype* orient_data, const int num,
                          const int num_priors, const int num_orient_classes,
                          std::vector<std::vector<std::vector<float>>>* orient_preds);

template <typename Dtype>
void GetOrientationScores(const Dtype* orient_data, const int num,
                          const int num_priors, const int num_orient_classes,
                          std::vector<std::vector<ArrayWrapper<Dtype>>>* orient_preds);

// Decode all bboxes in a batch.
void DecodeBBoxesAll(const std::vector<LabelBBox>& all_loc_pred,
                     const std::vector<NormalizedBBox>& prior_bboxes,
                     const std::vector<float >& prior_variances,
                     const int num, const bool share_location,
                     const int num_loc_classes, const int background_label_id,
                     const CodeType code_type, const bool variance_encoded_in_target,
                     std::vector<LabelBBox>* all_decode_bboxes);

// Do non maximum suppression given bboxes and scores.
// Inspired by Piotr Dollar's NMS implementation in EdgeBox.
// https://goo.gl/jV3JYS
//    bboxes: a set of bounding boxes.
//    scores: a set of corresponding confidences.
//    score_threshold: a threshold used to filter detection results.
//    nms_threshold: a threshold used in non maximum suppression.
//    top_k: if not -1, keep at most top_k picked indices.
//    indices: the kept indices of bboxes after nms.
template<typename Dtype>
void ApplyNMSFast(const std::vector<NormalizedBBox>& bboxes,
                  const std::vector<Dtype>& scores, const float score_threshold,
                  const float nms_threshold, const int top_k, std::vector<int>* indices);

template<typename Dtype>
void ApplyNMSFast(const BBoxArrayWrapper<Dtype>& bboxes,
                  const std::vector<Dtype>& scores, const float score_threshold,
                  const float nms_threshold, const int top_k, std::vector<int>* indices);

// Function sued to sort pair<float, T>, stored in STL container (e.g. vector)
// in descend order based on the score (first) value.
template <typename T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2);

template <typename T>
bool SortScorePairDescendStable(const std::pair<float, T>& pair1,
                                const std::pair<float, T>& pair2);

// Clip the NormalizedBBox such that the range for each corner is [0, 1].
void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clip_bbox);

template<typename Dtype>
void ClipBBox(const BBox<Dtype>& bbox, BBox<Dtype>* clip_bbox);

float get_orientation(const std::vector<float>& bin_vals, bool interpolate_orientation);
template<typename Dtype>
Dtype get_orientation(const ArrayWrapper<Dtype>& bin_vals, bool interpolate_orientation);
