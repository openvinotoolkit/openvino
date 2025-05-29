// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_fill_empty_rows.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <unordered_set>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/sparse_fill_empty_rows.hpp"
#include "openvino/reference/sparse_fill_empty_rows.hpp"
#include "selective_build.h"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {
SparseFillEmptyRows::SparseFillEmptyRows(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

bool SparseFillEmptyRows::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                               std::string& errorMessage) noexcept {
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
    const auto& valuesPrecision = getOriginalInputPrecisionAtPort(0);
    const auto& indicesPrecision = getOriginalInputPrecisionAtPort(2);
    addSupportedPrimDesc({{LayoutType::ncsp, valuesPrecision},        // values
                          {LayoutType::ncsp, indicesPrecision},       // dense_shape
                          {LayoutType::ncsp, indicesPrecision},       // indices
                          {LayoutType::ncsp, valuesPrecision}},       // default_value
                         {{LayoutType::ncsp, indicesPrecision},       // output_indices
                          {LayoutType::ncsp, valuesPrecision},        // output_values
                          {LayoutType::ncsp, ov::element::boolean}},  // empty_row_indicator
                         impl_desc_type::ref);
}

bool SparseFillEmptyRows::created() const {
    return getType() == Type::SparseFillEmptyRows;
}

bool SparseFillEmptyRows::needPrepareParams() const {
    return false;
}

bool SparseFillEmptyRows::isExecutable() const {
    return !isInputTensorAtPortEmpty(1) && !isInputTensorAtPortEmpty(3);
}

void SparseFillEmptyRows::executeDynamicImpl(const dnnl::stream& strm) {
    const auto& valuesMemory = getSrcMemoryAtPort(0);
    const auto& indicesMemory = getSrcMemoryAtPort(2);
    const auto& valuesShape = valuesMemory->getShape();
    const auto& indicesShape = indicesMemory->getShape();

    const auto* denseShapePtr = getSrcDataAtPortAs<const int32_t>(1);
    const auto numRows = static_cast<size_t>(denseShapePtr[0]);

    std::unordered_set<int32_t> existingRows;
    size_t indicesCount = indicesShape.getElementsCount() / 2;  // Divide by 2 because indices is [M, 2]

    const auto* indicesPtr = getSrcDataAtPortAs<const int32_t>(2);
    for (size_t i = 0; i < indicesCount; i++) {
        existingRows.insert(indicesPtr[i * 2]);
    }

    size_t emptyRowsCount = numRows - existingRows.size();
    size_t valuesCount = valuesShape.getElementsCount();
    ov::Shape outputIndicesShape{valuesCount + emptyRowsCount, 2};
    ov::Shape outputValuesShape{valuesCount + emptyRowsCount};
    ov::Shape emptyRowIndicatorShape{numRows};

    redefineOutputMemory({outputIndicesShape, outputValuesShape, emptyRowIndicatorShape});
    execute(strm);
}

namespace {
struct SparseFillEmptyRowsContext {
    SparseFillEmptyRows& node;
};
}  // namespace

template <typename T>
void SparseFillEmptyRows::executeImpl() {
    ov::reference::sparse_fill_empty_rows(getSrcDataAtPortAs<const T>(0),                        // values
                                          getSrcMemoryAtPort(0)->getShape().getElementsCount(),  // values_size
                                          getSrcDataAtPortAs<const int32_t>(1),                  // dense_shape
                                          getSrcDataAtPortAs<const int32_t>(2),                  // indices
                                          *getSrcDataAtPortAs<const T>(3),                       // default_value
                                          getDstDataAtPortAs<int32_t>(0),                        // output_indices
                                          getDstDataAtPortAs<T>(1),                              // output_values
                                          getDstDataAtPortAs<bool>(2));                          // empty_row_indicator
}

template <typename T>
struct SparseFillEmptyRows::SparseFillEmptyRowsExecute {
    void operator()(SparseFillEmptyRowsContext& ctx) {
        auto indicesPrecision = ctx.node.getParentEdgeAt(2)->getMemory().getDesc().getPrecision();
        if (indicesPrecision == ov::element::i32) {
            ctx.node.executeImpl<T>();
        } else {
            OPENVINO_THROW("SparseFillEmptyRows operation supports only i32 indices precision");
        }
    }
};

void SparseFillEmptyRows::execute([[maybe_unused]] const dnnl::stream& strm) {
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
