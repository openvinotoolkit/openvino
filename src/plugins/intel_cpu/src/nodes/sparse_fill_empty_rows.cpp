// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_fill_empty_rows.h"

#include "openvino/op/sparse_fill_empty_rows.hpp"
#include "openvino/reference/sparse_fill_empty_rows.hpp"

namespace ov::intel_cpu::node {
SparseFillEmptyRows::SparseFillEmptyRows(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

bool SparseFillEmptyRows::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::v16::SparseFillEmptyRows>(op)) {
            errorMessage = "Only opset16 SparseFillEmptyRows operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void SparseFillEmptyRows::getSupportedDescriptors() {
    // Validation is already done in the ov::opset16::SparseFillEmptyRows
}

void SparseFillEmptyRows::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type valuesPrecision = getOriginalInputPrecisionAtPort(0);
    ov::element::Type indicesPrecision = getOriginalInputPrecisionAtPort(2);

    // Validate tensor indices are int32 or int64
    if (indicesPrecision != ov::element::i32 && indicesPrecision != ov::element::i64) {
        OPENVINO_THROW("SparseFillEmptyRows operation supports only i32 or i64 indices precision");
    }

    addSupportedPrimDesc({{LayoutType::ncsp, valuesPrecision},    // values
                          {LayoutType::ncsp, indicesPrecision},   // dense_shape
                          {LayoutType::ncsp, indicesPrecision},   // indices
                          {LayoutType::ncsp, valuesPrecision}},   // default_value
                         {{LayoutType::ncsp, indicesPrecision},   // output_indices
                          {LayoutType::ncsp, valuesPrecision},    // output_values
                          {LayoutType::ncsp, ov::element::boolean}}, // empty_row_indicator
                         impl_desc_type::ref);
}

bool SparseFillEmptyRows::created() const {
    return getType() == Type::SparseFillEmptyRows;
}

bool SparseFillEmptyRows::needPrepareParams() const {
    return false;
}

void SparseFillEmptyRows::executeDynamicImpl(const dnnl::stream& strm) {
    // Get input shapes and data
    const auto& valuesMemory = getSrcMemoryAtPort(0);
    const auto& denseShapeMemory = getSrcMemoryAtPort(1);
    const auto& indicesMemory = getSrcMemoryAtPort(2);
    
    const auto& valuesShape = valuesMemory->getShape();
    const auto& denseShapeShape = denseShapeMemory->getShape();
    const auto& indicesShape = indicesMemory->getShape();
    
    // Get number of rows from dense_shape
    const auto indicesPrecision = getParentEdgeAt(2)->getMemory().getDesc().getPrecision();
    int64_t numRows = 0;
    
    if (indicesPrecision == ov::element::i32) {
        const auto* denseShapePtr = getSrcDataAtPortAs<const int32_t>(1);
        numRows = static_cast<int64_t>(denseShapePtr[0]);
    } else { // i64
        const auto* denseShapePtr = getSrcDataAtPortAs<const int64_t>(1);
        numRows = denseShapePtr[0];
    }
    
    // Count unique rows to determine empty rows
    std::unordered_set<int64_t> existingRows;
    size_t indicesCount = indicesShape.getElementsCount() / 2; // Divide by 2 because indices is [M,2]
    
    if (indicesPrecision == ov::element::i32) {
        const auto* indicesPtr = getSrcDataAtPortAs<const int32_t>(2);
        for (size_t i = 0; i < indicesCount; i++) {
            existingRows.insert(static_cast<int64_t>(indicesPtr[i * 2])); // Row indices (first column)
        }
    } else { // i64
        const auto* indicesPtr = getSrcDataAtPortAs<const int64_t>(2);
        for (size_t i = 0; i < indicesCount; i++) {
            existingRows.insert(indicesPtr[i * 2]); // Row indices (first column)
        }
    }
    
    // Calculate empty rows count
    size_t emptyRowsCount = numRows - existingRows.size();
    size_t valuesCount = valuesShape.getElementsCount();
    
    // Define output shapes
    ov::Shape outputIndicesShape{valuesCount + emptyRowsCount, 2};
    ov::Shape outputValuesShape{valuesCount + emptyRowsCount};
    ov::Shape emptyRowIndicatorShape{static_cast<size_t>(numRows)};
    
    // Redefine output memory with calculated shapes
    redefineOutputMemory({outputIndicesShape, outputValuesShape, emptyRowIndicatorShape});
    
    // Execute with the new shapes
    execute(strm);
}

namespace {
struct SparseFillEmptyRowsContext {
    SparseFillEmptyRows& node;
};
}  // namespace

template <typename T, typename T_IND>
void SparseFillEmptyRows::executeImpl() {
    const auto valuesShape = getSrcMemoryAtPort(0)->getShape();
    const auto denseShapeShape = getSrcMemoryAtPort(1)->getShape();
    const auto indicesShape = getSrcMemoryAtPort(2)->getShape();
    const auto outputIndicesShape = getDstMemoryAtPort(0)->getShape();
    const auto outputValuesShape = getDstMemoryAtPort(1)->getShape();
    const auto emptyRowIndicatorShape = getDstMemoryAtPort(2)->getShape();
    
    std::cout << "Debug SparseFillEmptyRows shapes:" << std::endl;
    std::cout << "  values shape: " << valuesShape << ", count: " << valuesShape.getElementsCount() << std::endl;
    std::cout << "  dense_shape shape: " << denseShapeShape << std::endl;
    std::cout << "  indices shape: " << indicesShape << std::endl;
    std::cout << "  output_indices shape: " << outputIndicesShape << std::endl;
    std::cout << "  output_values shape: " << outputValuesShape << std::endl;
    std::cout << "  empty_row_indicator shape: " << emptyRowIndicatorShape << std::endl;
    
    ov::reference::sparse_fill_empty_rows(
        getSrcDataAtPortAs<const T>(0),
        getSrcMemoryAtPort(0)->getShape().getElementsCount(),
        getSrcDataAtPortAs<const T_IND>(1),
        getSrcDataAtPortAs<const T_IND>(2),
        *getSrcDataAtPortAs<const T>(3),
        getDstDataAtPortAs<T_IND>(0),
        getDstDataAtPortAs<T>(1),
        getDstDataAtPortAs<bool>(2));
}

template <typename T>
struct SparseFillEmptyRows::SparseFillEmptyRowsExecute {
    void operator()(SparseFillEmptyRowsContext& ctx) {
        auto indicesPrecision = ctx.node.getParentEdgeAt(2)->getMemory().getDesc().getPrecision();
        if (indicesPrecision == ov::element::i32) {
            ctx.node.executeImpl<T, int32_t>();
        } else if (indicesPrecision == ov::element::i64) {
            ctx.node.executeImpl<T, int64_t>();
        } else {
            OPENVINO_THROW("SparseFillEmptyRows operation supports only i32 or i64 indices precision");
        }
    }
};

void SparseFillEmptyRows::execute(const dnnl::stream& strm) {
    auto valuesPrecision = getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
    SparseFillEmptyRowsContext ctx = {*this};
    OV_SWITCH(intel_cpu,
              SparseFillEmptyRowsExecute,
              ctx,
              valuesPrecision,
              OV_CASE(ov::element::f32, float),
              OV_CASE(ov::element::f16, ov::float16),
              OV_CASE(ov::element::bf16, ov::bfloat16),
              OV_CASE(ov::element::i8, int8_t),
              OV_CASE(ov::element::u8, uint8_t),
              OV_CASE(ov::element::i32, int32_t),
              OV_CASE(ov::element::i64, int64_t))
}
}  // namespace ov::intel_cpu::node
