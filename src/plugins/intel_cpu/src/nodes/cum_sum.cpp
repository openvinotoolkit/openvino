// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cum_sum.h"

#include <string>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "utils/bfloat16.hpp"

namespace ov::intel_cpu::node {

bool CumSum::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto cumsum = ov::as_type_ptr<const ov::opset3::CumSum>(op);
        if (!cumsum) {
            errorMessage = "Only opset3 CumSum operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

CumSum::CumSum(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if ((getOriginalInputsNumber() != numOfInputs && getOriginalInputsNumber() != (numOfInputs - 1)) ||
        getOriginalOutputsNumber() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output edges!");
    }

    const auto& dataShape = getInputShapeAtPort(CUM_SUM_DATA);
    numOfDims = dataShape.getRank();
    if (numOfDims < 1) {
        THROW_CPU_NODE_ERR("doesn't support 'data' input tensor with rank: ", numOfDims);
    }

    const auto cumsum = ov::as_type_ptr<const ov::opset3::CumSum>(op);
    if (cumsum == nullptr) {
        THROW_CPU_NODE_ERR("is not an instance of CumSum from opset3.");
    }

    exclusive = cumsum->is_exclusive();
    reverse = cumsum->is_reverse();

    if (getOriginalInputsNumber() == numOfInputs) {
        const auto axis_shape = cumsum->get_input_partial_shape(AXIS);
        if (axis_shape.is_dynamic() || !ov::is_scalar(axis_shape.to_shape())) {
            THROW_CPU_NODE_ERR("doesn't support 'axis' input tensor with non scalar rank");
        }
    }

    if (dataShape != getOutputShapeAtPort(0)) {
        THROW_CPU_NODE_ERR("has different 'data' input and output dimensions");
    }
}

void CumSum::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    dataPrecision = getOriginalInputPrecisionAtPort(CUM_SUM_DATA);
    if (!one_of(dataPrecision,
                ov::element::i8,
                ov::element::u8,
                ov::element::i16,
                ov::element::i32,
                ov::element::i64,
                ov::element::u64,
                ov::element::bf16,
                ov::element::f16,
                ov::element::f32)) {
        THROW_CPU_NODE_ERR("has unsupported 'data' input precision: ", dataPrecision.get_type_name());
    }

    if (inputShapes.size() == numOfInputs) {
        const auto& axisTensorPrec = getOriginalInputPrecisionAtPort(AXIS);
        if (axisTensorPrec != ov::element::i32 && axisTensorPrec != ov::element::i64) {
            THROW_CPU_NODE_ERR("has unsupported 'axis' input precision: ", axisTensorPrec.get_type_name());
        }
    }

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputShapes.size());
    inDataConf.emplace_back(LayoutType::ncsp, dataPrecision);
    for (size_t i = 1; i < inputShapes.size(); ++i) {
        inDataConf.emplace_back(LayoutType::ncsp, ov::element::i32);
    }

    addSupportedPrimDesc(inDataConf, {{LayoutType::ncsp, dataPrecision}}, impl_desc_type::ref_any);
}

void CumSum::execute(const dnnl::stream& strm) {
    if (inputShapes.size() == numOfInputs) {
        axis = getAxis(getParentEdgeAt(AXIS)->getMemory(), getParentEdgeAt(CUM_SUM_DATA)->getMemory());
    }

    OV_SWITCH(intel_cpu,
              CumSumExecute,
              this,
              dataPrecision,
              OV_CASE(ov::element::i8, int8_t),
              OV_CASE(ov::element::u8, uint8_t),
              OV_CASE(ov::element::i16, int16_t),
              OV_CASE(ov::element::bf16, bfloat16_t),
              OV_CASE(ov::element::f16, ov::float16),
              OV_CASE(ov::element::i32, int32_t),
              OV_CASE(ov::element::f32, float),
              OV_CASE(ov::element::i64, int64_t),
              OV_CASE(ov::element::u64, uint64_t))
}

template <typename dataType>
void CumSum::exec() {
    const auto* input = getSrcDataAtPortAs<const dataType>(CUM_SUM_DATA);
    auto* output = getDstDataAtPortAs<dataType>(0);
    const VectorDims strides =
        getParentEdgeAt(CUM_SUM_DATA)->getMemory().getDescWithType<BlockedMemoryDesc>()->getStrides();

    if (reverse) {
        if (exclusive) {
            cumSum<true, true, dataType>(input, output, strides);
        } else {
            cumSum<true, false, dataType>(input, output, strides);
        }
    } else {
        if (exclusive) {
            cumSum<false, true, dataType>(input, output, strides);
        } else {
            cumSum<false, false, dataType>(input, output, strides);
        }
    }
}

template <bool reverse, bool exclusive, typename dataType>
void CumSum::cumSum(const dataType* input, dataType* output, const VectorDims& strides) {
    VectorDims iterationRange(numOfDims - 1);
    size_t j = 0;
    const auto& shape = getParentEdgeAt(CUM_SUM_DATA)->getMemory().getStaticDims();
    for (size_t i = 0; i < shape.size(); i++) {
        if (i == axis) {
            continue;
        }
        iterationRange[j++] = shape[i];
    }
    size_t work_amount_dst =
        std::accumulate(iterationRange.begin(), iterationRange.end(), static_cast<size_t>(1), std::multiplies<>());
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        VectorDims counters(numOfDims - 1, 0);
        splitter(work_amount_dst, nthr, ithr, start, end);

        parallelItInit(start, counters, iterationRange);

        for (size_t iwork = start; iwork < end; ++iwork) {
            std::vector<size_t> forStartOffset(numOfDims);
            forStartOffset[axis] = 0;
            for (size_t offsetIdx = 0, countersIdx = 0; offsetIdx < numOfDims; ++offsetIdx) {
                if (offsetIdx == axis) {
                    continue;
                }
                forStartOffset[offsetIdx] = counters[countersIdx++];
            }

            size_t startOffset = getStartOffset(forStartOffset, strides);

            const dataType* inputStart = input + startOffset;
            dataType* outputStart = output + startOffset;

            size_t offset = strides[axis];
            if (reverse) {
                if (exclusive) {
                    outputStart[offset * (shape[axis] - 1)] = 0;
                    for (int64_t i = shape[axis] - 2; i >= 0; i--) {
                        outputStart[i * offset] = inputStart[(i + 1) * offset] + outputStart[(i + 1) * offset];
                    }
                } else {
                    outputStart[offset * (shape[axis] - 1)] = inputStart[offset * (shape[axis] - 1)];
                    for (int64_t i = shape[axis] - 2; i >= 0; i--) {
                        outputStart[i * offset] = inputStart[i * offset] + outputStart[(i + 1) * offset];
                    }
                }
            } else {
                if (exclusive) {
                    outputStart[0] = 0;
                    for (size_t i = 1; i < shape[axis]; i++) {
                        outputStart[i * offset] = inputStart[(i - 1) * offset] + outputStart[(i - 1) * offset];
                    }
                } else {
                    outputStart[0] = inputStart[0];
                    for (size_t i = 1; i < shape[axis]; i++) {
                        outputStart[i * offset] = inputStart[i * offset] + outputStart[(i - 1) * offset];
                    }
                }
            }

            parallelItStep(counters, iterationRange);
        }
    });
}

void CumSum::parallelItInit(size_t start, std::vector<size_t>& counters, const std::vector<size_t>& iterationRange) {
    auto itCounter = counters.rbegin();
    auto itWork = iterationRange.rbegin();
    while (itCounter != counters.rend() && itWork != iterationRange.rend()) {
        *itCounter = start % *itWork;
        start /= *itWork;
        ++itCounter;
        ++itWork;
    }
}

inline void CumSum::parallelItStep(std::vector<size_t>& counters, const std::vector<size_t>& iterationRange) {
    auto itCounter = counters.rbegin();
    auto itWork = iterationRange.rbegin();

    while (itCounter != counters.rend() && itWork != iterationRange.rend()) {
        *itCounter = (*itCounter + 1) % *itWork;
        if (*itCounter != 0) {
            break;
        }
        ++itCounter;
        ++itWork;
    }
}

inline size_t CumSum::getStartOffset(const std::vector<size_t>& forStartOffset,
                                     const std::vector<size_t>& strides) const {
    size_t startOffset = 0;
    for (size_t idx = 0; idx < forStartOffset.size(); ++idx) {
        startOffset += forStartOffset[idx] * strides[idx];
    }
    return startOffset;
}

size_t CumSum::getAxis(const IMemory& _axis, const IMemory& _data) const {
    const auto& axisPrecision = _axis.getDesc().getPrecision();
    const auto dataShapeSize = static_cast<int64_t>(_data.getShape().getRank());
    int64_t axisValueFromBlob = 0;
    switch (axisPrecision) {
    case ov::element::i32: {
        const auto* axisPtr = _axis.getDataAs<const int32_t>();
        axisValueFromBlob = static_cast<int64_t>(axisPtr[0]);
        break;
    }
    case ov::element::i64: {
        const auto* axisPtr = _axis.getDataAs<const int64_t>();
        axisValueFromBlob = axisPtr[0];
        break;
    }
    default: {
        THROW_CPU_NODE_ERR("doesn't support 'axis' input with precision: ", axisPrecision.get_type_name());
    }
    }
    if (axisValueFromBlob < -dataShapeSize || axisValueFromBlob > dataShapeSize - 1) {
        THROW_CPU_NODE_ERR("has axis with a value out of range: ", axisValueFromBlob);
    }
    return axisValueFromBlob >= 0 ? axisValueFromBlob : (axisValueFromBlob + dataShapeSize);
}

bool CumSum::created() const {
    return getType() == Type::CumSum;
}

bool CumSum::needPrepareParams() const {
    return false;
}

void CumSum::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

}  // namespace ov::intel_cpu::node
