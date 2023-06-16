// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <dnnl_extension_utils.h>

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
    Eye(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool needPrepareParams() const override {return false;};
    bool needShapeInfer() const override {return true;};
    void executeDynamicImpl(dnnl::stream strm) override { execute(strm); }

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    std::string errorPrefix = "";
    ov::element::Type outType = ov::element::Type_t::undefined;
    template <typename inputType>
    void executeSpecified();
    template<typename T>
    struct EyeExecute;
    inline const size_t getRowNum() const {
        auto rowMem = getParentEdgeAt(ROWS_NUM)->getMemoryPtr();
        if (rowMem == nullptr)
            IE_THROW() << errorPrefix << " doesn't contain row_count data";
        const int *rowPtr = reinterpret_cast<const int *>(rowMem->getData());

        return rowPtr[0];
    }
    inline const size_t getColNum() const {
        auto colMem = getParentEdgeAt(COLS_NUM)->getMemoryPtr();
        if (colMem == nullptr)
            IE_THROW() << errorPrefix << " doesn't contain col_count data";
        const int *colPtr =  reinterpret_cast<const int *>(colMem->getData());

        return colPtr[0];
    }
    inline const int getDiagIndex() const {
        auto diagIndMem = getParentEdgeAt(DIAGONAL_INDEX)->getMemoryPtr();
        if (diagIndMem == nullptr)
            IE_THROW() << errorPrefix << " doesn't contain diag_index data";
        const int *diagIndexPtr = reinterpret_cast<const int *>(diagIndMem->getData());

        return diagIndexPtr[0];
    }
    inline const std::vector<int> getBatchShape() const {
        if (withBatchShape) {
            const int batchShapeSize = static_cast<const int>(getParentEdgeAt(BATCH_SHAPE)->getMemoryPtr()->getShape().getElementsCount());
            std::vector<int> batchShape(batchShapeSize);
            const int *batchShapePtr = reinterpret_cast<const int *>(getParentEdgeAt(BATCH_SHAPE)->getMemoryPtr()->getData());
            batchShape.assign(batchShapePtr, batchShapePtr + batchShapeSize);
            return batchShape;
        } else {
            return std::vector<int> {};
        }
    }

    inline const size_t getBatchVolume(const std::vector<int> &batchShape) {
        return std::accumulate(begin(batchShape), end(batchShape), 1, std::multiplies<size_t>());
    }
    bool withBatchShape = false;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
