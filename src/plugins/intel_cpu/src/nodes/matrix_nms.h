// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

enum class MatrixNmsSortResultType : uint8_t {
    CLASSID,  // sort selected boxes by class id (ascending) in each batch element
    SCORE,    // sort selected boxes by score (descending) in each batch element
    NONE      // do not guarantee the order in each batch element
};

enum MatrixNmsDecayFunction : uint8_t { GAUSSIAN, LINEAR };

class MatrixNms : public Node {
public:
    MatrixNms(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
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
    // input
    static const size_t NMS_BOXES = 0;
    static const size_t NMS_SCORES = 1;

    // output
    static const size_t NMS_SELECTED_OUTPUTS = 0;
    static const size_t NMS_SELECTED_INDICES = 1;
    static const size_t NMS_VALID_OUTPUTS = 2;

    size_t m_numBatches = 0;
    size_t m_numBoxes = 0;
    size_t m_numClasses = 0;
    size_t m_maxBoxesPerBatch = 0;

    MatrixNmsSortResultType m_sortResultType;
    bool m_sortResultAcrossBatch;
    float m_scoreThreshold;
    int m_nmsTopk;
    int m_keepTopk;
    int m_backgroundClass;
    MatrixNmsDecayFunction m_decayFunction;
    float m_gaussianSigma;
    float m_postThreshold;
    bool m_normalized;

    bool m_outStaticShape = false;

    struct Rectangle {
        Rectangle(float x_left, float y_left, float x_right, float y_right)
            : x1{x_left},
              y1{y_left},
              x2{x_right},
              y2{y_right} {}

        Rectangle() = default;

        float x1 = 0.0F;
        float y1 = 0.0F;
        float x2 = 0.0F;
        float y2 = 0.0F;
    };

    struct BoxInfo {
        BoxInfo(const Rectangle& r, int64_t idx, float sc, int64_t batch_idx, int64_t class_idx)
            : box{r},
              index{idx},
              batchIndex{batch_idx},
              classIndex{class_idx},
              score{sc} {}

        BoxInfo() = default;

        Rectangle box;
        int64_t index = -1;
        int64_t batchIndex = -1;
        int64_t classIndex = -1;
        float score = 0.0F;
    };
    const std::string m_inType = "input", m_outType = "output";
    std::vector<int64_t> m_numPerBatch;
    std::vector<std::vector<int64_t>> m_numPerBatchClass;
    std::vector<BoxInfo> m_filteredBoxes;
    std::vector<int> m_classOffset;
    size_t m_realNumClasses = 0;
    size_t m_realNumBoxes = 0;
    float (*m_decay_fn)(float, float, float) = nullptr;
    void checkPrecision(ov::element::Type prec,
                        const std::vector<ov::element::Type>& precList,
                        const std::string& name,
                        const std::string& type);

    size_t nmsMatrix(const float* boxesData,
                     const float* scoresData,
                     BoxInfo* filterBoxes,
                     int64_t batchIdx,
                     int64_t classIdx);
};

}  // namespace ov::intel_cpu::node
