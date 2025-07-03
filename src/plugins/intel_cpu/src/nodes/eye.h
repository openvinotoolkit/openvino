// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class Eye : public Node {
public:
    static constexpr size_t ROWS_NUM = 0lu;
    static constexpr size_t COLS_NUM = 1lu;
    static constexpr size_t DIAGONAL_INDEX = 2lu;
    static constexpr size_t BATCH_SHAPE = 3lu;

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
    const size_t getRowNum() const {
        auto rowMem = getSrcMemoryAtPort(ROWS_NUM);
        if (rowMem == nullptr) {
            THROW_CPU_NODE_ERR("doesn't contain row_count data");
        }
        const auto* rowPtr = rowMem->getDataAs<const int>();

        return rowPtr[0];
    }
    const size_t getColNum() const {
        auto colMem = getSrcMemoryAtPort(COLS_NUM);
        if (colMem == nullptr) {
            THROW_CPU_NODE_ERR("doesn't contain col_count data");
        }
        const auto* colPtr = colMem->getDataAs<const int>();

        return colPtr[0];
    }
    const int getDiagIndex() const {
        auto diagIndMem = getSrcMemoryAtPort(DIAGONAL_INDEX);
        if (diagIndMem == nullptr) {
            THROW_CPU_NODE_ERR("doesn't contain diag_index data");
        }
        const auto* diagIndexPtr = diagIndMem->getDataAs<const int>();

        return diagIndexPtr[0];
    }
    const std::vector<int> getBatchShape() const {
        if (withBatchShape) {
            const auto batchShapeSize =
                static_cast<const int>(getSrcMemoryAtPort(BATCH_SHAPE)->getShape().getElementsCount());
            std::vector<int> batchShape(batchShapeSize);
            const auto* batchShapePtr = getSrcDataAtPortAs<const int>(BATCH_SHAPE);
            batchShape.assign(batchShapePtr, batchShapePtr + batchShapeSize);
            return batchShape;
        }
        return std::vector<int>{};
    }

    static const size_t getBatchVolume(const std::vector<int>& batchShape) {
        return std::accumulate(begin(batchShape), end(batchShape), 1, std::multiplies<>());
    }
    bool withBatchShape = false;
};

}  // namespace ov::intel_cpu::node
