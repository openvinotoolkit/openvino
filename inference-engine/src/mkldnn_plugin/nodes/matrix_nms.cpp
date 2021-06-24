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
#include <ngraph_ops/matrix_nms_ie_internal.hpp>
#include "utils/general_utils.h"
#include <ie_ngraph_utils.hpp>
#include <chrono>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using namespace MKLDNNPlugin;

class MatrixNmsImpl : public ExtLayerBase {
public:
    bool isSupportedOperation(const std::shared_ptr<ngraph::Node> &op, std::string &errorMessage) noexcept {
        try {
            const auto nms = std::dynamic_pointer_cast<const ngraph::op::internal::MatrixNmsIEInternal>(op);
            if (!nms) {
                errorMessage = "Only internal MatrixNMS operation is supported";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    explicit MatrixNmsImpl(const std::shared_ptr<ngraph::Node> &op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            errorPrefix = "MatirxNMS layer with name '" + op->get_friendly_name() + "' ";
            const auto matrix_nms = std::dynamic_pointer_cast<const ngraph::op::internal::MatrixNmsIEInternal>(
                    op);

            if (matrix_nms->get_input_size() != 2)
                IE_THROW() << errorPrefix << "has incorrect number of input edges: "
                           << matrix_nms->get_input_size();

            if (matrix_nms->get_output_size() < 1 || matrix_nms->get_output_size() > 4)
                IE_THROW() << errorPrefix << "has incorrect number of output edges: "
                           << matrix_nms->get_output_size();


            const std::vector<Precision> supportedFloatPrecision = {Precision::FP32};
            const std::vector<Precision> supportedIntOutputPrecision = {Precision::I32, Precision::I64};

            checkPrecision(op->get_input_element_type(NMS_BOXES), supportedFloatPrecision, "boxes", inType);
            const SizeVector &boxes_dims = op->get_input_shape(NMS_BOXES);
            num_batches = boxes_dims[0];
            num_boxes = boxes_dims[1];
            if (boxes_dims.size() != 3)
                IE_THROW() << errorPrefix << "has unsupported 'boxes' input rank: " << boxes_dims.size();
            if (boxes_dims[2] != 4)
                IE_THROW() << errorPrefix << "has unsupported 'boxes' input 3rd dimension size: "
                           << boxes_dims[2];

            checkPrecision(op->get_input_element_type(NMS_SCORES), supportedFloatPrecision, "scores",
                           inType);
            const SizeVector &scores_dims = op->get_input_shape(NMS_SCORES);
            num_classes = scores_dims[1];
            if (scores_dims.size() != 3)
                IE_THROW() << errorPrefix << "has unsupported 'scores' input rank: " << scores_dims.size();

            if (num_batches != scores_dims[0])
                IE_THROW() << errorPrefix << " num_batches is different in 'boxes' and 'scores' inputs";
            if (num_boxes != scores_dims[2])
                IE_THROW() << errorPrefix << " num_boxes is different in 'boxes' and 'scores' inputs";
            m_sort_result_type = (ngraph::op::util::NmsBase::SortResultType)matrix_nms->m_sort_result_type;
            m_sort_result_across_batch = matrix_nms->m_sort_result_across_batch;
            m_output_type = matrix_nms->m_output_type;
            m_score_threshold = matrix_nms->m_score_threshold;
            m_nms_top_k = matrix_nms->m_nms_top_k;
            m_keep_top_k = matrix_nms->m_keep_top_k;
            m_background_class = matrix_nms->m_background_class;
            m_decay_function = (ngraph::op::v8::MatrixNms::DecayFunction)matrix_nms->m_decay_function;
            m_gaussian_sigma = matrix_nms->m_gaussian_sigma;
            m_post_threshold = matrix_nms->m_post_threshold;
            LayerConfig config;
            for (size_t i = 0; i < op->get_input_size(); i++) {
                DataConfig inConfig;

                Precision inPrecision = Precision::FP32;
                const SizeVector &inDims = op->get_input_shape(i);
                inConfig.desc = TensorDesc(inPrecision, inDims,
                                           InferenceEngine::TensorDesc::getLayoutByDims(inDims));
                config.inConfs.push_back(inConfig);
            }
            for (size_t i = 0; i < op->get_output_size(); i++) {
                DataConfig outConfig;

                Precision outPrecision = i == NMS_SELECTED_OUTPUTS ? Precision::FP32 : Precision::I32;
                const SizeVector &outDims = op->get_output_shape(i);
                outConfig.desc = TensorDesc(outPrecision, outDims,
                                            InferenceEngine::TensorDesc::getLayoutByDims(outDims));
                config.outConfs.push_back(outConfig);
            }

            config.dynBatchSupport = false;
            confs.push_back(config);
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }

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
                      BoxInfo* filtBoxes) {
        std::vector<std::pair<float, int>> sorted_boxes;
        int64_t original_size = 0;

        for (int box_idx = 0; box_idx < boxes_num; box_idx++) {
            if (scores_data[box_idx] > score_threshold) {
                original_size++;
                sorted_boxes.emplace_back(std::make_pair(scores_data[box_idx], box_idx));
            }
        }
        int64_t  num_det = 0;
        if (original_size <= 0) {
            return 0;
        }
        if (top_k > -1 && original_size > top_k) {
            original_size = top_k;
        }

        parallel_sort(sorted_boxes.begin(), sorted_boxes.end(),
                      [](const std::pair<float, int>& l, const std::pair<float, int>& r) {
                          return (l.first > r.first);
                      });
        printf("0Sort %d\n", sorted_boxes[0].second);
        printf("1Sort %d\n", sorted_boxes[1].second);

        std::vector<T> iou_matrix((original_size * (original_size - 1)) >> 1);
        std::vector<T> iou_max(original_size);

        iou_max[0] = 0.;
        InferenceEngine::parallel_for(original_size - 1, [&](size_t i){
            T max_iou = 0.;
            size_t actual_index = i + 1;
            auto idx_a = sorted_boxes[actual_index].second;
            for (int64_t j = 0; j < actual_index; j++) {
                auto idx_b = sorted_boxes[j].second;
                auto iou = intersectionOverUnion<T>(boxes_data + idx_a * box_size,
                                                    boxes_data + idx_b * box_size,
                                                    normalized);
                max_iou = std::max(max_iou, iou);
                iou_matrix[actual_index * (actual_index - 1) / 2 + j] = iou;
            }
            iou_max[actual_index] = max_iou;
        });

        if (scores_data[sorted_boxes[0].second] > post_threshold) {
            auto box_index = sorted_boxes[0].second;
            auto box = boxes_data + box_index * box_size;
            filtBoxes[0].box.x1 = box[0];
            filtBoxes[0].box.y1 = box[1];
            filtBoxes[0].box.x2 = box[2];
            filtBoxes[0].box.y2 = box[3];
            filtBoxes[0].index = batch_idx * num_boxes + box_index;
            filtBoxes[0].score = sorted_boxes[0].first;
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
            auto ds = min_decay * scores_data[sorted_boxes[i].second];
            if (ds <= post_threshold)
                continue;
            auto box_index = sorted_boxes[i].second;
            auto box = boxes_data + box_index * box_size;
            filtBoxes[num_det].box.x1 = box[0];
            filtBoxes[num_det].box.y1 = box[1];
            filtBoxes[num_det].box.x2 = box[2];
            filtBoxes[num_det].box.y2 = box[3];
            filtBoxes[num_det].index = batch_idx * num_boxes + box_index;
            filtBoxes[num_det].score = ds;
            filtBoxes[num_det].batch_index = batch_idx;
            filtBoxes[num_det].class_index = class_idx;
            num_det++;
//            printf("nms_matrix batch %ld class %ld no. %ld index %ld\n", batch_idx, class_idx, i, filtBoxes[num_det].index);
        }
        return num_det;
    }

    StatusCode execute(std::vector<Blob::Ptr> &inputs, std::vector<Blob::Ptr> &outputs,
                       ResponseDesc *resp) noexcept override {
        auto start = std::chrono::high_resolution_clock::now();
        const float *boxes = inputs[NMS_BOXES]->cbuffer().as<const float *>() +
                             inputs[NMS_BOXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const float *scores = inputs[NMS_SCORES]->cbuffer().as<const float *>() +
                              inputs[NMS_SCORES]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const int box_shape = 4;
        size_t  real_num_classes = m_background_class == -1 ? num_classes : num_classes - 1;
        size_t  real_num_boxes = m_nms_top_k == -1 ? num_boxes : std::min(m_nms_top_k, static_cast<int>(num_boxes));
        std::vector<int64_t> num_per_batch(num_batches);
        std::vector<BoxInfo> filtered_boxes(num_batches * real_num_classes * real_num_boxes);

        bool normalized = true;
        InferenceEngine::parallel_for(num_batches, [&](size_t batch) {
            const float *boxes_ptr = boxes + batch * num_boxes * 4;
            std::vector<BoxInfo> batch_filtered_box(real_num_classes * real_num_boxes);
            std::vector<int> class_offset(num_classes, 0);
            std::vector<int64_t> num_per_class(num_classes, 0);
            for (size_t i = 0, count = 0; i < num_classes; i++) {
                if (i == m_background_class)
                    continue;
                class_offset[i] = (count++) * real_num_boxes;
            }

            int64_t num_det = 0;
            InferenceEngine::parallel_for(num_classes, [&](size_t class_idx){
                if (class_idx == m_background_class)
                    return;
                const float *scores_ptr =
                        scores + batch * (num_classes * num_boxes) + class_idx * num_boxes;
                size_t class_num_det = 0;
                if (m_decay_function == ngraph::op::v8::MatrixNms::DecayFunction::GAUSSIAN) {
                    class_num_det = nms_matrix<float, true>(boxes_ptr,
                                                                 num_boxes,
                                                                 box_shape,
                                                                 scores_ptr,
                                                                 m_score_threshold,
                                                                 m_post_threshold,
                                                                 m_gaussian_sigma,
                                                                 m_nms_top_k,
                                                                 normalized,
                                                                 batch,
                                                                 class_idx,
                                                                 batch_filtered_box.data() + class_offset[class_idx]);
                } else {
                    class_num_det = nms_matrix<float, false>(boxes_ptr,
                                                             num_boxes,
                                                             box_shape,
                                                             scores_ptr,
                                                             m_score_threshold,
                                                             m_post_threshold,
                                                             m_gaussian_sigma,
                                                             m_nms_top_k,
                                                             normalized,
                                                             batch,
                                                             class_idx,
                                                             batch_filtered_box.data() + class_offset[class_idx]);
                }
//                printf("batch %ld class_idx %ld %ld\n", batch, class_idx, class_num_det);
                num_per_class[class_idx] = class_num_det;
            });
            num_det = std::accumulate(num_per_class.begin(), num_per_class.end(), 0);
            if (num_det <= 0) {
                return;
            }

            auto start_offset = 0;
            for (size_t i = 0; i < num_per_class.size(); i++) {
                auto offset_class = class_offset[i];
                for (size_t j = 0; j < num_per_class[i]; j++) {
                    batch_filtered_box[start_offset + j] = batch_filtered_box[offset_class + j];
//                    printf("batch %ld class %ld no. %ld total index %ld\n", batch, i, j,
//                            batch_filtered_box[offset_class + j].index);
                }
                start_offset += num_per_class[i];
            }

            batch_filtered_box.resize(start_offset);

            if (m_keep_top_k > -1) {
                auto k = static_cast<size_t>(m_keep_top_k);
                if (num_det > k)
                    num_det = k;
            }
            for (int i = 0; i < 3; i++) {
                printf("before sort index %ld score %f\n", batch_filtered_box[i].index, batch_filtered_box[i].score);
            }
            printf("num_det %ld\n", num_det);
            std::sort(batch_filtered_box.begin(),
                        batch_filtered_box.end(),
                              [](BoxInfo& lhs, BoxInfo& rhs) {
                                  return lhs.score > rhs.score;
                              });
            for (int i = 0; i < 3; i++) {
                printf("after sort index %ld score %f\n", batch_filtered_box[i].index, batch_filtered_box[i].score);
            }
            auto offset = batch * real_num_classes * real_num_boxes;
            for (size_t i = 0; i < num_det; i++) {
                filtered_boxes[offset + i] = batch_filtered_box[i];
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

        float * selected_outputs = outputs[NMS_SELECTED_OUTPUTS]->buffer().as<float *>() +
                                outputs[NMS_SELECTED_OUTPUTS]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        int *selected_indices = outputs[NMS_SELECTED_INDICES]->buffer().as<int *>() +
                                    outputs[NMS_SELECTED_INDICES]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        int *valid_outputs = outputs[NMS_VALID_OUTPUTS]->buffer().as<int *>() +
                                 outputs[NMS_VALID_OUTPUTS]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        std::copy(num_per_batch.begin(), num_per_batch.end(), valid_outputs);

        for (size_t i = 0; i < num_batches; i++) {
            valid_outputs[i] = static_cast<int>(num_per_batch[i]);
        }
        for (size_t i = 0; i < filtered_boxes.size(); i++) {
            selected_indices[i] = static_cast<int>(filtered_boxes[i].index);
            auto selected_base = selected_outputs + i * 6;
            selected_base[0] = filtered_boxes[i].class_index;
            selected_base[1] = filtered_boxes[i].score;
            selected_base[2] = filtered_boxes[i].box.x1;
            selected_base[3] = filtered_boxes[i].box.y1;
            selected_base[4] = filtered_boxes[i].box.x2;
            selected_base[5] = filtered_boxes[i].box.y2;
        }
        printf("first index %d class %f score %f \n", selected_indices[0], selected_outputs[0], selected_outputs[1]);
        printf("second index %d class %f score %f \n", selected_indices[1], selected_outputs[6], selected_outputs[7]);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " NMSMatrix_parallel " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
        return OK;
    }

private:
    // input
    const size_t NMS_BOXES = 0;
    const size_t NMS_SCORES = 1;

    // output
    const size_t NMS_SELECTED_OUTPUTS = 0;
    const size_t NMS_SELECTED_INDICES = 1;
    const size_t NMS_VALID_OUTPUTS = 2;

    enum class boxEncoding {
        CORNER,
        CENTER
    };
    boxEncoding boxEncodingType = boxEncoding::CORNER;
    bool sort_result_descending = true;

    size_t num_batches;
    size_t num_boxes;
    size_t num_classes;

    ngraph::op::util::NmsBase::SortResultType m_sort_result_type;
    bool m_sort_result_across_batch;
    ngraph::element::Type m_output_type;
    float m_score_threshold;
    int m_nms_top_k;
    int m_keep_top_k;
    int m_background_class;
    ngraph::op::v8::MatrixNms::DecayFunction m_decay_function;
    float m_gaussian_sigma;
    float m_post_threshold;

    std::string errorPrefix;
    const std::string inType = "input", outType = "output";

    void checkPrecision(const ngraph::element::Type &ngPrec, const std::vector<Precision> precList,
                        const std::string name, const std::string type) {
        const auto prec = details::convertPrecision(ngPrec);
        if (std::find(precList.begin(), precList.end(), prec) == precList.end())
            IE_THROW()
                    << errorPrefix << "has unsupported '" << name << "' " << type << " precision: " << prec;
    }

    void check1DInput(const std::shared_ptr<ngraph::Node> &op, const std::vector<Precision> precList,
                      const std::string name, const size_t port) {
        checkPrecision(op->get_input_element_type(port), precList, name, inType);

        const SizeVector &dims = op->get_input_shape(port);
        if (dims.size() != 0 && dims.size() != 1)
            IE_THROW() << errorPrefix << "has unsupported '" << name << "' input rank: " << dims.size();
        if (dims.size() == 1)
            if (dims[0] != 1)
                IE_THROW() << errorPrefix << "has unsupported '" << name << "' input 1st dimension size: "
                           << dims[0];
    }

    void checkOutput(const std::shared_ptr<ngraph::Node> &op, const std::vector<Precision> precList,
                     const std::string name, const size_t port) {
        checkPrecision(op->get_output_element_type(port), precList, name, outType);

        const SizeVector &dims = op->get_output_shape(port);
        if (dims.size() != 2)
            IE_THROW() << errorPrefix << "has unsupported '" << name << "' output rank: " << dims.size();
        if (dims[1] != 3)
            IE_THROW() << errorPrefix << "has unsupported '" << name << "' output 2nd dimension size: "
                       << dims[1];
    }
};

REG_FACTORY_FOR(MatrixNmsImpl, MatrixNmsIEInternal);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
