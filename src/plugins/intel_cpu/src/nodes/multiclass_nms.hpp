// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov::intel_cpu::node {

enum class MulticlassNmsSortResultType {
    CLASSID,  // sort selected boxes by class id (ascending) in each batch element
    SCORE,    // sort selected boxes by score (descending) in each batch element
    NONE      // do not guarantee the order in each batch element
};

class MultiClassNms : public Node {
public:
    MultiClassNms(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    [[nodiscard]] bool neverExecute() const override;
    [[nodiscard]] bool isExecutable() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    [[nodiscard]] bool needShapeInfer() const override {
        return false;
    }
    void prepareParams() override;

private:
    // input (port Num)
    const size_t NMS_BOXES = 0;
    const size_t NMS_SCORES = 1;
    const size_t NMS_ROISNUM = 2;

    // output (port Num)
    const size_t NMS_SELECTEDOUTPUTS = 0;
    const size_t NMS_SELECTEDINDICES = 1;
    const size_t NMS_SELECTEDNUM = 2;

    bool m_sortResultAcrossBatch = false;
    MulticlassNmsSortResultType m_sortResultType = MulticlassNmsSortResultType::NONE;

    size_t m_numBatches = 0;
    size_t m_numBoxes = 0;
    size_t m_numClasses = 0;
    size_t m_maxBoxesPerBatch = 0;

    int m_nmsRealTopk = 0;
    int m_nmsTopK = 0;
    float m_iouThreshold = 0.0f;
    float m_scoreThreshold = 0.0f;

    int32_t m_backgroundClass = 0;
    int32_t m_keepTopK = 0;
    float m_nmsEta = 0.0f;
    bool m_normalized = true;

    bool m_outStaticShape = false;

    std::vector<std::vector<size_t>> m_numFiltBox;  // number of rois after nms for each class in each image
    std::vector<size_t> m_numBoxOffset;
    const std::string m_inType = "input", m_outType = "output";

    struct filteredBoxes {
        float score;
        int batch_index;
        int class_index;
        int box_index;
        filteredBoxes() = default;
        filteredBoxes(float _score, int _batch_index, int _class_index, int _box_index)
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

    std::vector<filteredBoxes> m_filtBoxes;  // rois after nms for each class in each image

    void checkPrecision(const ov::element::Type prec,
                        const std::vector<ov::element::Type>& precList,
                        const std::string& name,
                        const std::string& type);

    float intersectionOverUnion(const float* boxesI, const float* boxesJ, const bool normalized);

    void nmsWithEta(const float* boxes,
                    const float* scores,
                    const int* roisnum,
                    const VectorDims& boxesStrides,
                    const VectorDims& scoresStrides,
                    const VectorDims& roisnumStrides,
                    const bool shared);

    void nmsWithoutEta(const float* boxes,
                       const float* scores,
                       const int* roisnum,
                       const VectorDims& boxesStrides,
                       const VectorDims& scoresStrides,
                       const VectorDims& roisnumStrides,
                       const bool shared);

    const float* slice_class(const int batch_idx,
                             const int class_idx,
                             const float* dataPtr,
                             const VectorDims& dataStrides,
                             const bool is_boxes,
                             const int* roisnum,
                             const VectorDims& roisnumStrides,
                             const bool shared);
};

}  // namespace ov::intel_cpu::node
