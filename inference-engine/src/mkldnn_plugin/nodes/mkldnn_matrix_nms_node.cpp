// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include <utility>
#include <queue>
#include "ie_parallel.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "ngraph_ops/nms_static_shape_ie.hpp"
#include "utils/general_utils.h"
#include <ie_ngraph_utils.hpp>
#include "mkldnn_matrix_nms_node.h"
#include <chrono>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using MatrixNmsIEInternal = ngraph::op::internal::NmsStaticShapeIE<ngraph::op::v8::MatrixNms>;

bool MKLDNNMatrixNmsNode::isSupportedOperation(const std::shared_ptr<ngraph::Node> &op, std::string &errorMessage) noexcept {
    try {
        const auto nms = std::dynamic_pointer_cast<const MatrixNmsIEInternal>(op);
        if (!nms) {
            errorMessage = "Only internal MatrixNms operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNMatrixNmsNode::MKLDNNMatrixNmsNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                                                  MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "MatirxNMS layer with name '" + op->get_friendly_name() + "' ";
    const auto matrix_nms = std::dynamic_pointer_cast<const MatrixNmsIEInternal>(
            op);

    if (matrix_nms->get_input_size() != 2)
        IE_THROW() << errorPrefix << "has incorrect number of input edges: "
                   << matrix_nms->get_input_size();

    if (matrix_nms->get_output_size() < 1 || matrix_nms->get_output_size() > 4)
        IE_THROW() << errorPrefix << "has incorrect number of output edges: "
                   << matrix_nms->get_output_size();


    const std::vector<Precision> supportedFloatPrecision = {Precision::FP32};
    const std::vector<Precision> supportedIntOutputPrecision = {Precision::I32, Precision::I64};

    checkPrecision(getOriginalInputPrecisionAtPort(NMS_BOXES), supportedFloatPrecision, "boxes", inType);
    checkPrecision(getOriginalInputPrecisionAtPort(NMS_SCORES), supportedFloatPrecision, "scores", inType);

    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTED_OUTPUTS), supportedFloatPrecision, "selected_outputs", outType);
    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTED_INDICES), supportedIntOutputPrecision, "selected_indices", outType);
    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_VALID_OUTPUTS), supportedIntOutputPrecision, "valid_outputs", outType);

    outputShape_SELECTED_OUTPUTS = op->get_output_shape(NMS_SELECTED_OUTPUTS);
    outputShape_SELECTED_INDICES = op->get_output_shape(NMS_SELECTED_INDICES);
    outputShape_VALID_OUTPUTS = op->get_output_shape(NMS_VALID_OUTPUTS);

    const SizeVector &boxes_dims = op->get_input_shape(NMS_BOXES);
    m_num_batches = boxes_dims[0];
    m_num_boxes = boxes_dims[1];
    if (boxes_dims.size() != 3)
        IE_THROW() << errorPrefix << "has unsupported 'boxes' input rank: " << boxes_dims.size();
    if (boxes_dims[2] != 4)
        IE_THROW() << errorPrefix << "has unsupported 'boxes' input 3rd dimension size: "
                   << boxes_dims[2];
    const SizeVector &scores_dims = op->get_input_shape(NMS_SCORES);
    m_num_classes = scores_dims[1];
    if (scores_dims.size() != 3)
        IE_THROW() << errorPrefix << "has unsupported 'scores' input rank: " << scores_dims.size();

    if (m_num_batches != scores_dims[0])
        IE_THROW() << errorPrefix << " num_batches is different in 'boxes' and 'scores' inputs";
    if (m_num_boxes != scores_dims[2])
        IE_THROW() << errorPrefix << " num_boxes is different in 'boxes' and 'scores' inputs";
    auto& attrs = matrix_nms->get_attrs();
    m_sort_result_type = attrs.sort_result_type;
    m_sort_result_across_batch = attrs.sort_result_across_batch;
    m_output_type = attrs.output_type;
    m_score_threshold = attrs.score_threshold;
    m_nms_top_k = attrs.nms_top_k;
    m_keep_top_k = attrs.keep_top_k;
    m_background_class = attrs.background_class;
    m_decay_function = attrs.decay_function;
    m_gaussian_sigma = attrs.gaussian_sigma;
    m_post_threshold = attrs.post_threshold;
    m_normalized = attrs.normalized;
    int64_t max_output_boxes_per_class = 0;
    size_t  real_num_classes = m_background_class == -1 ? m_num_classes : m_num_classes - 1;
    if (m_nms_top_k >= 0)
        max_output_boxes_per_class = std::min(m_num_boxes, static_cast<size_t>(m_nms_top_k));
    else
        max_output_boxes_per_class = m_num_boxes;

    m_max_boxes_per_batch = max_output_boxes_per_class * real_num_classes;
    if (m_keep_top_k >= 0)
        m_max_boxes_per_batch = std::min(m_max_boxes_per_batch, static_cast<size_t>(m_keep_top_k));
}

void MKLDNNMatrixNmsNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const std::vector<Precision> supportedFloatPrecision = {Precision::FP32};
    const std::vector<Precision> supportedIntOutputPrecision = {Precision::I32, Precision::I64};

    checkPrecision(getOriginalInputPrecisionAtPort(NMS_BOXES), supportedFloatPrecision, "boxes", inType);
    checkPrecision(getOriginalInputPrecisionAtPort(NMS_SCORES), supportedFloatPrecision, "scores", inType);
    checkOutput(outputShape_VALID_OUTPUTS, supportedIntOutputPrecision, "valid_outputs", NMS_VALID_OUTPUTS);
    checkOutput(outputShape_SELECTED_OUTPUTS, supportedFloatPrecision, "selected_outputs", NMS_SELECTED_OUTPUTS);
    checkOutput(outputShape_SELECTED_INDICES, supportedIntOutputPrecision, "selected_indices", NMS_SELECTED_INDICES);

    std::vector<DataConfigurator> inDataConf;
    inDataConf.reserve(getOriginalInputsNumber());
    for (int i = 0; i < getOriginalInputsNumber(); ++i) {
        Precision inPrecision = Precision::FP32;
        inDataConf.emplace_back(TensorDescCreatorTypes::ncsp, inPrecision);
    }

    std::vector<DataConfigurator> outDataConf;
    outDataConf.reserve(getOriginalOutputsNumber());
    for (int i = 0; i < getOriginalOutputsNumber(); ++i) {
        Precision outPrecision = i == NMS_SELECTED_OUTPUTS ? Precision::FP32 : Precision::I32;
        outDataConf.emplace_back(TensorDescCreatorTypes::ncsp, outPrecision);
    }

    addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref_any);
}

bool MKLDNNMatrixNmsNode::created() const {
    return getType() == MatrixNms;
}

namespace {
template<typename T, bool gaussian>
struct decay_score;

template<typename T>
struct decay_score<T, true> {
    T operator()(T iou, T max_iou, T sigma) {
        return std::exp((max_iou * max_iou - iou * iou) * sigma);
    }
};

template<typename T>
struct decay_score<T, false> {
    T operator()(T iou, T max_iou, T sigma) { return (1. - iou) / (1. - max_iou + 1e-10f); }
};

template<class T>
static inline T BBoxArea(const T *box, const bool normalized) {
    if (box[2] < box[0] || box[3] < box[1]) {
        // If coordinate values are is invalid
        // (e.g. xmax < xmin or ymax < ymin), return 0.
        return static_cast<T>(0.);
    } else {
        const T w = box[2] - box[0];
        const T h = box[3] - box[1];
        if (normalized) {
            return w * h;
        } else {
            // If coordinate values are not within range [0, 1].
            return (w + 1) * (h + 1);
        }
    }
}

template<class T>
static inline T
intersectionOverUnion(const T *box1, const T *box2, const bool normalized) {
    if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
        box2[3] < box1[1]) {
        return static_cast<T>(0.);
    } else {
        const T inter_xmin = std::max(box1[0], box2[0]);
        const T inter_ymin = std::max(box1[1], box2[1]);
        const T inter_xmax = std::min(box1[2], box2[2]);
        const T inter_ymax = std::min(box1[3], box2[3]);
        T norm = normalized ? static_cast<T>(0.) : static_cast<T>(1.);
        T inter_w = inter_xmax - inter_xmin + norm;
        T inter_h = inter_ymax - inter_ymin + norm;
        const T inter_area = inter_w * inter_h;
        const T bbox1_area = BBoxArea<T>(box1, normalized);
        const T bbox2_area = BBoxArea<T>(box2, normalized);
        return inter_area / (bbox1_area + bbox2_area - inter_area);
    }
}

struct Rectangle {
    Rectangle(float x_left, float y_left, float x_right, float y_right)
            : x1{x_left}, y1{y_left}, x2{x_right}, y2{y_right} {}

    Rectangle() = default;

    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct BoxInfo {
    BoxInfo(const Rectangle &r,
            int64_t idx,
            float sc,
            int64_t batch_idx,
            int64_t class_idx)
            : box{r}, index{idx}, batch_index{batch_idx}, class_index{class_idx}, score{sc} {
    }

    BoxInfo() = default;

    Rectangle box;
    int64_t index = -1;
    int64_t batch_index = -1;
    int64_t class_index = -1;
    float score = 0.0f;
};

template<typename T, bool gaussian>
size_t nms_matrix(const T *boxes_data,
                  const int64_t boxes_num,
                  const int64_t box_size,
                  const T *scores_data,
                  const T score_threshold,
                  const T post_threshold,
                  const float sigma,
                  const int64_t top_k,
                  const bool normalized,
                  const int64_t batch_idx,
                  const int64_t class_idx,
                  BoxInfo *filtBoxes) {
    std::vector<int32_t> candidate_index(boxes_num);
    std::iota(candidate_index.begin(), candidate_index.end(), 0);
    auto end = std::remove_if(candidate_index.begin(),
                              candidate_index.end(),
                              [&scores_data, score_threshold](int32_t idx) {
                                  return scores_data[idx] <= score_threshold;
                              });
    int64_t num_det = 0;
    int64_t original_size = std::distance(candidate_index.begin(), end);
    if (original_size <= 0) {
        return 0;
    }
    if (top_k > -1 && original_size > top_k) {
        original_size = top_k;
    }

    std::partial_sort(candidate_index.begin(),
                      candidate_index.begin() + original_size,
                      end,
                      [&scores_data](int32_t a, int32_t b) {
                          return scores_data[a] > scores_data[b];
                      });

    std::vector<T> iou_matrix((original_size * (original_size - 1)) >> 1);
    std::vector<T> iou_max(original_size);

    iou_max[0] = 0.;
    InferenceEngine::parallel_for(original_size - 1, [&](size_t i) {
        T max_iou = 0.;
        size_t actual_index = i + 1;
        auto idx_a = candidate_index[actual_index];
        for (int64_t j = 0; j < actual_index; j++) {
            auto idx_b = candidate_index[j];
            auto iou = intersectionOverUnion<T>(boxes_data + idx_a * box_size,
                                                boxes_data + idx_b * box_size,
                                                normalized);
            max_iou = std::max(max_iou, iou);
            iou_matrix[actual_index * (actual_index - 1) / 2 + j] = iou;
        }
        iou_max[actual_index] = max_iou;
    });

    if (scores_data[candidate_index[0]] > post_threshold) {
        auto box_index = candidate_index[0];
        auto box = boxes_data + box_index * box_size;
        filtBoxes[0].box.x1 = box[0];
        filtBoxes[0].box.y1 = box[1];
        filtBoxes[0].box.x2 = box[2];
        filtBoxes[0].box.y2 = box[3];
        filtBoxes[0].index = batch_idx * boxes_num + box_index;
        filtBoxes[0].score = scores_data[candidate_index[0]];
        filtBoxes[0].batch_index = batch_idx;
        filtBoxes[0].class_index = class_idx;
        num_det++;
    }

    decay_score<T, gaussian> decay_fn;
    for (int64_t i = 1; i < original_size; i++) {
        T min_decay = 1.;
        for (int64_t j = 0; j < i; j++) {
            auto max_iou = iou_max[j];
            auto iou = iou_matrix[i * (i - 1) / 2 + j];
            auto decay = decay_fn(iou, max_iou, sigma);
            min_decay = std::min(min_decay, decay);
        }
        auto ds = min_decay * scores_data[candidate_index[i]];
        if (ds <= post_threshold)
            continue;
        auto box_index = candidate_index[i];
        auto box = boxes_data + box_index * box_size;
        filtBoxes[num_det].box.x1 = box[0];
        filtBoxes[num_det].box.y1 = box[1];
        filtBoxes[num_det].box.x2 = box[2];
        filtBoxes[num_det].box.y2 = box[3];
        filtBoxes[num_det].index = batch_idx * boxes_num + box_index;
        filtBoxes[num_det].score = ds;
        filtBoxes[num_det].batch_index = batch_idx;
        filtBoxes[num_det].class_index = class_idx;
        num_det++;
    }
    return num_det;
}
}//  Anonymous matrix_nms namespace

void MKLDNNMatrixNmsNode::execute(mkldnn::stream strm) {
    const float *boxes = reinterpret_cast<const float *>(getParentEdgeAt(NMS_BOXES)->getMemoryPtr()->GetPtr());
    const float *scores = reinterpret_cast<const float *>(getParentEdgeAt(NMS_SCORES)->getMemoryPtr()->GetPtr());

    const int box_shape = 4;
    size_t  real_num_classes = m_background_class == -1 ? m_num_classes : m_num_classes - 1;
    size_t  real_num_boxes = m_nms_top_k == -1 ? m_num_boxes : std::min(m_nms_top_k, static_cast<int>(m_num_boxes));
    std::vector<int64_t> num_per_batch(m_num_batches);
    std::vector<BoxInfo> filtered_boxes(m_num_batches * real_num_classes * real_num_boxes);

    InferenceEngine::parallel_for(m_num_batches, [&](size_t batch) {
        const float *boxes_ptr = boxes + batch * m_num_boxes * 4;
        std::vector<BoxInfo> batch_filtered_box(real_num_classes * real_num_boxes);
        std::vector<int> class_offset(m_num_classes, 0);
        std::vector<int64_t> num_per_class(m_num_classes, 0);
        for (size_t i = 0, count = 0; i < m_num_classes; i++) {
            if (i == m_background_class)
                continue;
            class_offset[i] = (count++) * real_num_boxes;
        }

        int64_t num_det = 0;
        InferenceEngine::parallel_for(m_num_classes, [&](size_t class_idx){
            if (class_idx == m_background_class)
                return;
            const float *scores_ptr =
                    scores + batch * (m_num_classes * m_num_boxes) + class_idx * m_num_boxes;
            size_t class_num_det = 0;
            if (m_decay_function == ngraph::op::v8::MatrixNms::DecayFunction::GAUSSIAN) {
                class_num_det = nms_matrix<float, true>(boxes_ptr,
                                                             m_num_boxes,
                                                             box_shape,
                                                             scores_ptr,
                                                             m_score_threshold,
                                                             m_post_threshold,
                                                             m_gaussian_sigma,
                                                             m_nms_top_k,
                                                             m_normalized,
                                                             batch,
                                                             class_idx,
                                                             batch_filtered_box.data() + class_offset[class_idx]);
            } else {
                class_num_det = nms_matrix<float, false>(boxes_ptr,
                                                         m_num_boxes,
                                                         box_shape,
                                                         scores_ptr,
                                                         m_score_threshold,
                                                         m_post_threshold,
                                                         m_gaussian_sigma,
                                                         m_nms_top_k,
                                                         m_normalized,
                                                         batch,
                                                         class_idx,
                                                         batch_filtered_box.data() + class_offset[class_idx]);
            }
            num_per_class[class_idx] = class_num_det;
        });
        num_det = std::accumulate(num_per_class.begin(), num_per_class.end(), 0);
        if (num_det <= 0) {
            return;
        }

        auto start_offset = num_per_class[0];
        for (size_t i = 1; i < num_per_class.size(); i++) {
            auto offset_class = class_offset[i];
            for (size_t j = 0; j < num_per_class[i]; j++) {
                batch_filtered_box[start_offset + j] = batch_filtered_box[offset_class + j];
            }
            start_offset += num_per_class[i];
        }

        batch_filtered_box.resize(start_offset);

        if (m_keep_top_k > -1) {
            auto k = static_cast<size_t>(m_keep_top_k);
            if (num_det > k)
                num_det = k;
        }

        std::vector<int32_t> perm(batch_filtered_box.size());
        std::iota(perm.begin(), perm.end(), 0);

        std::partial_sort(perm.begin(),
                          perm.begin() + num_det,
                          perm.end(),
                          [&batch_filtered_box](int lhs, int rhs) {
                              return batch_filtered_box[lhs].score > batch_filtered_box[rhs].score ||
                                      (batch_filtered_box[lhs].score == batch_filtered_box[rhs].score &&
                                      batch_filtered_box[lhs].class_index < batch_filtered_box[rhs].class_index) ||
                                      (batch_filtered_box[lhs].score == batch_filtered_box[rhs].score &&
                                       batch_filtered_box[lhs].class_index == batch_filtered_box[rhs].class_index &&
                                       batch_filtered_box[lhs].index < batch_filtered_box[rhs].index);
                          });

        auto offset = batch * real_num_classes * real_num_boxes;
        for (size_t i = 0; i < num_det; i++) {
            filtered_boxes[offset + i] = batch_filtered_box[perm[i]];
        }
        num_per_batch[batch] = num_det;
    });

    auto start_offset = num_per_batch[0];
    for (size_t i = 1; i < num_per_batch.size(); i++) {
        auto offset_batch = i * real_num_classes * real_num_boxes;
        for (size_t j = 0; j < num_per_batch[i]; j++) {
            filtered_boxes[start_offset + j] = filtered_boxes[offset_batch + j];
        }
        start_offset += num_per_batch[i];
    }

    filtered_boxes.resize(start_offset);
    if (m_sort_result_across_batch) { /* sort across batch */
        if (m_sort_result_type == ngraph::op::v8::MatrixNms::SortResultType::SCORE) {
            std::sort(
                    filtered_boxes.begin(),
                    filtered_boxes.end(),
                    [](const BoxInfo &l, const BoxInfo &r) {
                        return (l.score > r.score) ||
                               (l.score == r.score && l.batch_index < r.batch_index) ||
                               (l.score == r.score && l.batch_index == r.batch_index &&
                                l.class_index < r.class_index) ||
                               (l.score == r.score && l.batch_index == r.batch_index &&
                                l.class_index == r.class_index && l.index < r.index);
                    });
        } else if (m_sort_result_type == ngraph::op::v8::MatrixNms::SortResultType::CLASSID) {
            std::sort(filtered_boxes.begin(),
                      filtered_boxes.end(),
                      [](const BoxInfo &l, const BoxInfo &r) {
                          return (l.class_index < r.class_index) ||
                                 (l.class_index == r.class_index &&
                                  l.batch_index < r.batch_index) ||
                                 (l.class_index == r.class_index &&
                                  l.batch_index == r.batch_index &&
                                  l.score > r.score) ||
                                 (l.class_index == r.class_index &&
                                  l.batch_index == r.batch_index &&
                                  l.score == r.score && l.index < r.index);
                      });
        }
    }

    float * selected_outputs = reinterpret_cast<float *>(getChildEdgesAtPort(NMS_SELECTED_OUTPUTS)[0]->getMemoryPtr()->GetPtr());
    int *selected_indices = reinterpret_cast<int *>(getChildEdgesAtPort(NMS_SELECTED_INDICES)[0]->getMemoryPtr()->GetPtr());
    int *valid_outputs = reinterpret_cast<int *>(getChildEdgesAtPort(NMS_VALID_OUTPUTS)[0]->getMemoryPtr()->GetPtr());
    std::copy(num_per_batch.begin(), num_per_batch.end(), valid_outputs);

    int64_t output_offset = 0;
    int64_t original_offset = 0;
    for (size_t i = 0; i < m_num_batches; i++) {
        auto real_boxes = num_per_batch[i];
        valid_outputs[i] = static_cast<int>(real_boxes);

        for (size_t j = 0; j < real_boxes; j++) {
            auto original_index = original_offset + j;
            selected_indices[j + output_offset] = static_cast<int>(filtered_boxes[original_index].index);
            auto selected_base = selected_outputs + (output_offset + j) * 6;
            selected_base[0] = filtered_boxes[original_index].class_index;
            selected_base[1] = filtered_boxes[original_index].score;
            selected_base[2] = filtered_boxes[original_index].box.x1;
            selected_base[3] = filtered_boxes[original_index].box.y1;
            selected_base[4] = filtered_boxes[original_index].box.x2;
            selected_base[5] = filtered_boxes[original_index].box.y2;
        }
        std::fill_n(selected_outputs + (output_offset + real_boxes) * 6, (m_max_boxes_per_batch - real_boxes) * 6, -1);
        std::fill_n(selected_indices + (output_offset + real_boxes), m_max_boxes_per_batch - real_boxes, -1);
        output_offset += m_max_boxes_per_batch;
        original_offset += real_boxes;
    }
}

void MKLDNNMatrixNmsNode::checkPrecision(const Precision prec, const std::vector<Precision> precList,
                                                 const std::string name, const std::string type) {
    if (std::find(precList.begin(), precList.end(), prec) == precList.end())
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' " << type << " precision: " << prec;
}


void MKLDNNMatrixNmsNode::checkOutput(const SizeVector& dims, const std::vector<Precision> precList,
                                              const std::string name, const size_t port) {
    checkPrecision(getOriginalOutputPrecisionAtPort(port), precList, name, outType);
}

REG_MKLDNN_PRIM_FOR(MKLDNNMatrixNmsNode, MatrixNms);
