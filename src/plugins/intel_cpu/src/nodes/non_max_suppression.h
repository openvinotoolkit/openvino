// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernels/x64/non_max_suppression.hpp"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

enum NMSCandidateStatus { SUPPRESSED = 0, SELECTED = 1, UPDATED = 2 };

class NonMaxSuppression : public Node {
public:
    NonMaxSuppression(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};

    void initSupportedPrimitiveDescriptors() override;

    void execute(const dnnl::stream& strm) override;

    void executeDynamicImpl(const dnnl::stream& strm) override;

    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    struct FilteredBox {
        float score;
        int batch_index;
        int class_index;
        int box_index;
        FilteredBox() = default;
        FilteredBox(float _score, int _batch_index, int _class_index, int _box_index)
            : score(_score),
              batch_index(_batch_index),
              class_index(_class_index),
              box_index(_box_index) {}
    };

    struct boxInfo {
        float score;
        int idx;
        int suppress_begin_index;
    };

    bool neverExecute() const override;
    bool isExecutable() const override;

    bool needShapeInfer() const override {
        return false;
    }

    void prepareParams() override;

    struct Point2D {
        float x, y;
        Point2D(const float px = 0.f, const float py = 0.f) : x(px), y(py) {}
        Point2D operator+(const Point2D& p) const {
            return Point2D(x + p.x, y + p.y);
        }
        Point2D& operator+=(const Point2D& p) {
            x += p.x;
            y += p.y;
            return *this;
        }
        Point2D operator-(const Point2D& p) const {
            return Point2D(x - p.x, y - p.y);
        }
        Point2D operator*(const float coeff) const {
            return Point2D(x * coeff, y * coeff);
        }
    };

private:
    // input
    enum {
        NMS_BOXES,
        NMS_SCORES,
        NMS_MAX_OUTPUT_BOXES_PER_CLASS,
        NMS_IOU_THRESHOLD,
        NMS_SCORE_THRESHOLD,
        NMS_SOFT_NMS_SIGMA,
    };

    // output
    enum { NMS_SELECTED_INDICES, NMS_SELECTED_SCORES, NMS_VALID_OUTPUTS };

    float intersectionOverUnion(const float* boxesI, const float* boxesJ);

    float rotatedIntersectionOverUnion(const Point2D (&vertices_0)[4], const float area_0, const float* box_1);

    void nmsWithSoftSigma(const float* boxes,
                          const float* scores,
                          const VectorDims& boxesStrides,
                          const VectorDims& scoresStrides,
                          std::vector<FilteredBox>& filtBoxes);

    void nmsWithoutSoftSigma(const float* boxes,
                             const float* scores,
                             const VectorDims& boxesStrides,
                             const VectorDims& scoresStrides,
                             std::vector<FilteredBox>& filtBoxes);

    void nmsRotated(const float* boxes,
                    const float* scores,
                    const VectorDims& boxesStrides,
                    const VectorDims& scoresStrides,
                    std::vector<FilteredBox>& filtBoxes);

    void check1DInput(const Shape& shape, const std::string& name, const size_t port);

    void checkOutput(const Shape& shape, const std::string& name, const size_t port);

    void createJitKernel();

    NMSBoxEncodeType boxEncodingType = NMSBoxEncodeType::CORNER;
    bool m_sort_result_descending = true;
    bool m_clockwise = false;
    bool m_rotated_boxes = false;
    size_t m_coord_num = 1lu;

    size_t m_batches_num = 0lu;
    size_t m_boxes_num = 0lu;
    size_t m_classes_num = 0lu;

    size_t m_max_output_boxes_per_class = 0lu;  // Original value of input NMS_MAX_OUTPUT_BOXES_PER_CLASS
    size_t m_output_boxes_per_class = 0lu;      // Actual number of output boxes
    float m_iou_threshold = 0.f;
    float m_score_threshold = 0.f;
    float m_soft_nms_sigma = 0.f;
    float m_scale = 0.f;
    // control placeholder for NMS in new opset.
    bool m_is_soft_suppressed_by_iou = false;

    bool m_out_static_shape = false;

    std::vector<std::vector<size_t>> m_num_filtered_boxes;
    const std::string inType = "input";
    const std::string outType = "output";
    bool m_defined_outputs[NMS_VALID_OUTPUTS + 1] = {false, false, false};
    std::vector<FilteredBox> m_filtered_boxes;

    std::shared_ptr<kernel::JitKernelBase> m_jit_kernel;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
