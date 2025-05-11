// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <string>
#include <vector>

#include "dnnl_extension_utils.h"

namespace ov {
namespace intel_cpu {
namespace node {

class Eye : public Node {
public:
    static constexpr size_t ROWS_NUM = 0lu;
    static constexpr size_t COLS_NUM = 1lu;
    static constexpr size_t DIAGONAL_INDEX = 2lu;
    static constexpr size_t BATCH_SHAPE = 3lu;

public:
    Eye(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    bool needPrepareParams() const override {
        return false;
    };
    bool needShapeInfer() const override {
        return true;
    };
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    ov::element::Type outType = ov::element::Type_t::dynamic;
    template <typename inputType>
    void executeSpecified();
    template <typename T>
    struct EyeExecute;
    inline const size_t getRowNum() const {
        auto rowMem = getSrcMemoryAtPort(ROWS_NUM);
        if (rowMem == nullptr)
            THROW_CPU_NODE_ERR("doesn't contain row_count data");
        const int* rowPtr = rowMem->getDataAs<const int>();

        return rowPtr[0];
    }
    inline const size_t getColNum() const {
        auto colMem = getSrcMemoryAtPort(COLS_NUM);
        if (colMem == nullptr)
            THROW_CPU_NODE_ERR("doesn't contain col_count data");
        const int* colPtr = colMem->getDataAs<const int>();

        return colPtr[0];
    }
    inline const int getDiagIndex() const {
        auto diagIndMem = getSrcMemoryAtPort(DIAGONAL_INDEX);
        if (diagIndMem == nullptr)
            THROW_CPU_NODE_ERR("doesn't contain diag_index data");
        const int* diagIndexPtr = diagIndMem->getDataAs<const int>();

        return diagIndexPtr[0];
    }
    inline const std::vector<int> getBatchShape() const {
        if (withBatchShape) {
            const int batchShapeSize =
                static_cast<const int>(getSrcMemoryAtPort(BATCH_SHAPE)->getShape().getElementsCount());
            std::vector<int> batchShape(batchShapeSize);
            const int* batchShapePtr = getSrcDataAtPortAs<const int>(BATCH_SHAPE);
            batchShape.assign(batchShapePtr, batchShapePtr + batchShapeSize);
            return batchShape;
        } else {
            return std::vector<int>{};
        }
    }

    inline const size_t getBatchVolume(const std::vector<int>& batchShape) {
        return std::accumulate(begin(batchShape), end(batchShape), 1, std::multiplies<>());
    }
    bool withBatchShape = false;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
