// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_tree.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "gather_tree.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool GatherTree::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto gatherElementsOp = ov::as_type_ptr<const ov::op::v1::GatherTree>(op);
        if (!gatherElementsOp) {
            errorMessage = "Node is not an instance of the GatherTree operation from operation set v1.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

GatherTree::GatherTree(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    CPU_NODE_ASSERT(inputShapes.size() == 4, "has incorrect number of input edges.");
    CPU_NODE_ASSERT(outputShapes.size() == 1, "has incorrect number of output edges.");

    CPU_NODE_ASSERT(getInputShapeAtPort(GATHER_TREE_STEP_IDX).getRank() == 3, "step_idx vector should be 3 dimension");
    CPU_NODE_ASSERT(getInputShapeAtPort(GATHER_TREE_PARENT_IDX).getRank() == 3,
                    "parent_idx vector should be 3 dimension");
    CPU_NODE_ASSERT(getInputShapeAtPort(GATHER_TREE_MAX_SEQ_LEN).getRank() == 1,
                    "max_seq_len vector should be 1 dimension");
    CPU_NODE_ASSERT(is_scalar(op->get_input_partial_shape(GATHER_TREE_END_TOKEN)), "end_token should be scalar");
}

void GatherTree::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    precision = getOriginalInputPrecisionAtPort(GATHER_TREE_STEP_IDX);
    if (none_of(precision, ov::element::f32, ov::element::i32)) {
        precision = ov::element::f32;
    }

    if (getOriginalInputPrecisionAtPort(GATHER_TREE_PARENT_IDX) != precision ||
        getOriginalInputPrecisionAtPort(GATHER_TREE_MAX_SEQ_LEN) != precision ||
        getOriginalInputPrecisionAtPort(GATHER_TREE_END_TOKEN) != precision ||
        getOriginalOutputPrecisionAtPort(0) != precision) {
        CPU_NODE_THROW("has incorrect input/output data precision. Must be the same.");
    }

    addSupportedPrimDesc({{LayoutType::ncsp, precision},
                          {LayoutType::ncsp, precision},
                          {LayoutType::ncsp, precision},
                          {LayoutType::ncsp, precision}},
                         {{LayoutType::ncsp, precision}},
                         impl_desc_type::ref_any);
}

void GatherTree::execute([[maybe_unused]] const dnnl::stream& strm) {
    CPU_NODE_ASSERT(execPtr, "has not compiled executor.");

    if (precision == ov::element::f32) {
        execPtr->exec<float>(getSrcMemoryAtPort(GATHER_TREE_STEP_IDX),
                             getSrcMemoryAtPort(GATHER_TREE_PARENT_IDX),
                             getSrcMemoryAtPort(GATHER_TREE_MAX_SEQ_LEN),
                             getSrcMemoryAtPort(GATHER_TREE_END_TOKEN),
                             getDstMemoryAtPort(0));
    } else {
        execPtr->exec<int32_t>(getSrcMemoryAtPort(GATHER_TREE_STEP_IDX),
                               getSrcMemoryAtPort(GATHER_TREE_PARENT_IDX),
                               getSrcMemoryAtPort(GATHER_TREE_MAX_SEQ_LEN),
                               getSrcMemoryAtPort(GATHER_TREE_END_TOKEN),
                               getDstMemoryAtPort(0));
    }
}

void GatherTree::prepareParams() {
    const auto& stepIdxMemPtr = getSrcMemoryAtPort(GATHER_TREE_STEP_IDX);
    const auto& parentIdxMemPtr = getSrcMemoryAtPort(GATHER_TREE_PARENT_IDX);
    const auto& maxSeqLenMemPtr = getSrcMemoryAtPort(GATHER_TREE_MAX_SEQ_LEN);
    const auto& dstMemPtr = getDstMemoryAtPort(0);

    CPU_NODE_ASSERT(stepIdxMemPtr && stepIdxMemPtr->isDefined(), "has undefined input memory of 'step_ids'.");
    CPU_NODE_ASSERT(parentIdxMemPtr && parentIdxMemPtr->isDefined(), "has undefined input memory of 'parent_ids'.");
    CPU_NODE_ASSERT(maxSeqLenMemPtr && maxSeqLenMemPtr->isDefined(), "has undefined input memory of 'max_seq_len'.");
    CPU_NODE_ASSERT(dstMemPtr && dstMemPtr->isDefined(), "has undefined output memory.");
    CPU_NODE_ASSERT(getSelectedPrimitiveDescriptor(), "has unidentified preferable primitive descriptor.");

    const VectorDims& stepIdxDims = stepIdxMemPtr->getStaticDims();
    const VectorDims& parentIdxDims = parentIdxMemPtr->getStaticDims();
    const VectorDims& maxSeqLenDims = maxSeqLenMemPtr->getStaticDims();
    const VectorDims& dstDims = dstMemPtr->getStaticDims();

    execPtr = std::make_shared<GatherTreeExecutor>(stepIdxDims, parentIdxDims, maxSeqLenDims, dstDims);
}

void GatherTree::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

GatherTree::GatherTreeExecutor::GatherTreeExecutor(const VectorDims& stepIdxDims,
                                                   const VectorDims& parentIdxDims,
                                                   const VectorDims& maxSeqLenDims,
                                                   const VectorDims& dstDims)
    : maxTime{static_cast<int32_t>(stepIdxDims[0])},
      batchSize{stepIdxDims[1]},
      beamWidth{stepIdxDims[2]},
      bbSize{batchSize * beamWidth},
      parentIdxSize{std::accumulate(parentIdxDims.cbegin(), parentIdxDims.cend(), 1LU, std::multiplies<>())} {
    if (maxTime != static_cast<int32_t>(parentIdxDims[0]) || maxTime != static_cast<int32_t>(dstDims[0]) ||
        batchSize != parentIdxDims[1] || batchSize != dstDims[1] || batchSize != maxSeqLenDims[0] ||
        beamWidth != parentIdxDims[2] || beamWidth != dstDims[2]) {
        std::string errorMsg = "Input/Output tensors dimensions mismatch";
        OPENVINO_THROW(errorMsg);
    }
}

template <typename DATA_T>
void GatherTree::GatherTreeExecutor::exec(const MemoryPtr& stepIdxMemPtr,
                                          const MemoryPtr& parentIdxMemPtr,
                                          const MemoryPtr& maxSeqLenMemPtr,
                                          const MemoryPtr& endTokenMemPtr,
                                          const MemoryPtr& dstMemPtr) {
    const auto* stepIdx = stepIdxMemPtr->getDataAs<DATA_T>();
    const auto* parentIdx = parentIdxMemPtr->getDataAs<DATA_T>();
    const auto* maxSeqLen = maxSeqLenMemPtr->getDataAs<DATA_T>();
    const auto endToken = (endTokenMemPtr->getDataAs<DATA_T>())[0];
    auto* finalIdx = dstMemPtr->getDataAs<DATA_T>();

    bool incorrectResult = false;
    parallel_for2d(batchSize, beamWidth, [&](size_t batch, size_t beam) {
        int32_t maxSequenceInBeam = std::min<int32_t>(maxTime, static_cast<int32_t>(maxSeqLen[batch]));
        if (maxSequenceInBeam > 0) {
            int32_t time = (maxTime - 1);
            int32_t idx = (maxTime - 1) * bbSize + batch * beamWidth;
            for (; time >= maxSequenceInBeam; time--, idx -= bbSize) {
                finalIdx[idx + beam] = endToken;
            }

            for (auto parent = static_cast<int32_t>(beam); time >= 0; time--, idx -= bbSize) {
                if (parent < 0 || parent >= static_cast<int32_t>(beamWidth) ||
                    static_cast<size_t>(idx) + parent >= parentIdxSize) {
                    incorrectResult = true;
                    break;
                }
                finalIdx[idx + beam] = stepIdx[idx + parent];
                parent = static_cast<int32_t>(parentIdx[idx + parent]);
            }

            bool finished = false;
            auto* final = &finalIdx[batch * beamWidth + beam];
            for (time = 0; time < maxSequenceInBeam; time++, final += bbSize) {
                if (finished) {
                    (*final) = endToken;
                } else if ((*final) == endToken) {
                    finished = true;
                }
            }
        }
    });

    if (incorrectResult) {
        std::string errorMsg = "Wrong parent index, result is incorrect";
        OPENVINO_THROW(errorMsg);
    }
}

bool GatherTree::created() const {
    return getType() == Type::GatherTree;
}

}  // namespace ov::intel_cpu::node
