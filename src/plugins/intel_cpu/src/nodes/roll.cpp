// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roll.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <openvino/op/roll.hpp>
#include <string>
#include <vector>

#include "common/cpu_memcpy.h"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

using namespace dnnl;

namespace ov::intel_cpu::node {

bool Roll::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto interp = ov::as_type_ptr<const ov::op::v7::Roll>(op);
        if (!interp) {
            errorMessage = "Only v7 Roll operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Roll::Roll(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        CPU_NODE_ASSERT(inputShapes.size() == 3 && outputShapes.size() == 1,
                        "has incorrect number of input/output edges!");

        const auto& dataPrecision = getOriginalInputPrecisionAtPort(DATA_INDEX);

        if (std::find(supportedPrecisionSizes.begin(), supportedPrecisionSizes.end(), dataPrecision.size()) ==
            supportedPrecisionSizes.end()) {
            CPU_NODE_THROW("as unsupported precision: ", dataPrecision.get_type_name());
        }

        const auto dataRank = getInputShapeAtPort(DATA_INDEX).getRank();
        CPU_NODE_ASSERT(dataRank >= 1, "doesn't support 'data' input tensor with rank: ", dataRank);

        CPU_NODE_ASSERT(dataRank == getOutputShapeAtPort(0).getRank(), "has input/output rank mismatch");

        /* Axes */
        const auto& axesTensorPrec = getOriginalInputPrecisionAtPort(AXES_INDEX);
        if (none_of(axesTensorPrec, ov::element::i32, ov::element::i64)) {
            CPU_NODE_THROW("has unsupported 'axes' input precision: ", axesTensorPrec.get_type_name());
        }

        const auto axesTensorRank = getInputShapeAtPort(AXES_INDEX).getRank();
        CPU_NODE_ASSERT(axesTensorRank <= 1, "doesn't support 'axes' input tensor with rank: ", axesTensorRank);

        /* Shift */
        const auto& shiftTensorPrec = getOriginalInputPrecisionAtPort(SHIFT_INDEX);
        if (none_of(shiftTensorPrec, ov::element::i32, ov::element::i64)) {
            CPU_NODE_THROW("has unsupported 'shift' input precision: ", shiftTensorPrec.get_type_name());
        }

        const auto shiftTensorRank = getInputShapeAtPort(SHIFT_INDEX).getRank();
        CPU_NODE_ASSERT(shiftTensorRank <= 1, "doesn't support 'shift' input tensor with rank: ", shiftTensorRank);
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void Roll::getSupportedDescriptors() {}

void Roll::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type precision = getOriginalInputPrecisionAtPort(0);

    addSupportedPrimDesc(
        {{LayoutType::ncsp, precision}, {LayoutType::ncsp, ov::element::i32}, {LayoutType::ncsp, ov::element::i32}},
        {{LayoutType::ncsp, precision}},
        impl_desc_type::ref);
}

void Roll::prepareParams() {
    const auto& dataMemPtr = getSrcMemoryAtPort(DATA_INDEX);
    const auto& shiftMemPtr = getSrcMemoryAtPort(SHIFT_INDEX);
    const auto& axesMemPtr = getSrcMemoryAtPort(AXES_INDEX);
    const auto& dstMemPtr = getDstMemoryAtPort(0);

    CPU_NODE_ASSERT(dataMemPtr && dataMemPtr->isDefined(), "has undefined input memory of 'data'");
    CPU_NODE_ASSERT(shiftMemPtr && shiftMemPtr->isDefined(), "has undefined input memory of 'shift'");
    CPU_NODE_ASSERT(axesMemPtr && axesMemPtr->isDefined(), "has undefined input memory of 'axes'");
    CPU_NODE_ASSERT(dstMemPtr && dstMemPtr->isDefined(), "has undefined output memory");
    CPU_NODE_ASSERT(getSelectedPrimitiveDescriptor(), "has unidentified preferable primitive descriptor");

    const VectorDims& dataDims = dataMemPtr->getStaticDims();
    const VectorDims& shiftDims = shiftMemPtr->getStaticDims();
    const VectorDims& axesDims = axesMemPtr->getStaticDims();
    const VectorDims& dstDims = dstMemPtr->getStaticDims();

    execPtr = std::make_shared<RollExecutor>(dataDims, shiftDims, axesDims, dstDims);
}

void Roll::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void Roll::execute([[maybe_unused]] const dnnl::stream& strm) {
    CPU_NODE_ASSERT(execPtr, "has no compiled executor");

    const auto dataPrecision = getParentEdgeAt(DATA_INDEX)->getMemory().getDesc().getPrecision();
    const auto& dataTypeSize = dataPrecision.size();
    switch (dataTypeSize) {
    case sizeof(element_type_traits<ov::element::i8>::value_type): {
        execPtr->exec<element_type_traits<ov::element::i8>::value_type>(getSrcMemoryAtPort(DATA_INDEX),
                                                                        getSrcMemoryAtPort(SHIFT_INDEX),
                                                                        getSrcMemoryAtPort(AXES_INDEX),
                                                                        getDstMemoryAtPort(0));
        break;
    }
    case sizeof(element_type_traits<ov::element::i16>::value_type): {
        execPtr->exec<element_type_traits<ov::element::i16>::value_type>(getSrcMemoryAtPort(DATA_INDEX),
                                                                         getSrcMemoryAtPort(SHIFT_INDEX),
                                                                         getSrcMemoryAtPort(AXES_INDEX),
                                                                         getDstMemoryAtPort(0));
        break;
    }
    case sizeof(element_type_traits<ov::element::i32>::value_type): {
        execPtr->exec<element_type_traits<ov::element::i32>::value_type>(getSrcMemoryAtPort(DATA_INDEX),
                                                                         getSrcMemoryAtPort(SHIFT_INDEX),
                                                                         getSrcMemoryAtPort(AXES_INDEX),
                                                                         getDstMemoryAtPort(0));
        break;
    }
    default:
        CPU_NODE_THROW("as unsupported 'data' input precision: ", dataPrecision.get_type_name());
    }
}

Roll::RollExecutor::RollExecutor(const VectorDims& dataDims,
                                 const VectorDims& shiftDims,
                                 const VectorDims& axesDims,
                                 const VectorDims& dstDims)
    : numOfDims{dataDims.size()},
      blockSize{dataDims.back()},
      numOfIterations{std::accumulate(dataDims.cbegin(), dataDims.cend(), 1UL, std::multiplies<>()) / blockSize},
      axesLength{axesDims[0]} {
    for (size_t i = 0; i < dataDims.size(); ++i) {
        OPENVINO_ASSERT(dataDims[i] == dstDims[i], "Input/output tensors dimensions mismatch");
    }

    OPENVINO_ASSERT(shiftDims[0] == axesDims[0], "'shift' and 'axes' dimensions mismatch");
}

template <typename T>
void Roll::RollExecutor::exec(const MemoryPtr& dataMemPtr,
                              const MemoryPtr& shiftMemPtr,
                              const MemoryPtr& axesMemPtr,
                              const MemoryPtr& dstMemPtr) {
    const auto* data = dataMemPtr->getDataAs<const T>();
    const auto* shift = shiftMemPtr->getDataAs<const int32_t>();
    const auto* axes = axesMemPtr->getDataAs<const int32_t>();
    auto* dst = dstMemPtr->getDataAs<T>();

    std::vector<size_t> shiftsVector(numOfDims, 0UL);
    const VectorDims& dataDims = dataMemPtr->getStaticDims();

    for (size_t dim = 0; dim < axesLength; ++dim) {
        int32_t currentAxis = axes[dim] < 0 ? axes[dim] + numOfDims : axes[dim];
        int32_t shiftSum = shiftsVector[currentAxis] + shift[dim];
        int32_t dimSize = dataDims[currentAxis];
        shiftsVector[currentAxis] = (shiftSum % dimSize + dimSize) % dimSize;
    }

    const size_t leftBlockSize = blockSize - shiftsVector.back();
    const size_t rightBlockSize = blockSize - leftBlockSize;
    const size_t elementSize = sizeof(T);

    const auto strides = dataMemPtr->getDescWithType<BlockedMemoryDesc>()->getStrides();
    const auto calculateShiftOffset = [](size_t dataOffset, size_t dimShift, size_t segmentSize, size_t dimSize) {
        size_t pos = dataOffset / segmentSize % dimSize;
        size_t shift = (pos + dimShift) % dimSize - pos;

        return dataOffset + shift * segmentSize;
    };

    parallel_for(numOfIterations, [&, this](size_t iter) {
        size_t start = iter * blockSize;
        size_t leftBlockStartOffset = start;
        size_t rightBlockStartOffset = start + leftBlockSize;

        for (int dim = numOfDims - 1; dim >= 0; --dim) {
            leftBlockStartOffset =
                calculateShiftOffset(leftBlockStartOffset, shiftsVector[dim], strides[dim], dataDims[dim]);
            rightBlockStartOffset =
                calculateShiftOffset(rightBlockStartOffset, shiftsVector[dim], strides[dim], dataDims[dim]);
        }

        if (leftBlockSize > 0) {
            cpu_memcpy(dst + leftBlockStartOffset, data + start, leftBlockSize * elementSize);
        }

        if (rightBlockSize > 0) {
            cpu_memcpy(dst + rightBlockStartOffset, data + (start + leftBlockSize), rightBlockSize * elementSize);
        }
    });
}

bool Roll::created() const {
    return getType() == Type::Roll;
}

}  // namespace ov::intel_cpu::node
