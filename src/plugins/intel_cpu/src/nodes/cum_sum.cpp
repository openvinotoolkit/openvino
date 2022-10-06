// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include "ie_parallel.hpp"
#include "ie_precision.hpp"
#include <ie_ngraph_utils.hpp>
#include "cum_sum.h"
#include "utils/bfloat16.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool CumSum::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto cumsum = std::dynamic_pointer_cast<const ngraph::opset3::CumSum>(op);
        if (!cumsum) {
            errorMessage = "Only opset3 CumSum operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

CumSum::CumSum(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng,
        WeightsSharing::Ptr &cache) : Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "CumSum layer with name '" + op->get_friendly_name() + "' ";

    if ((getOriginalInputsNumber() != numOfInputs && getOriginalInputsNumber() != (numOfInputs - 1)) || getOriginalOutputsNumber() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    const auto &dataShape = getInputShapeAtPort(CUM_SUM_DATA);
    numOfDims = dataShape.getRank();
    if (numOfDims < 1) {
        IE_THROW() << errorPrefix << " doesn't support 'data' input tensor with rank: " << numOfDims;
    }

    const auto cumsum = std::dynamic_pointer_cast<const ngraph::opset3::CumSum>(op);
    if (cumsum == nullptr)
        IE_THROW() << "Operation with name '" << op->get_friendly_name() <<
            "' is not an instance of CumSum from opset3.";

    exclusive = cumsum->is_exclusive();
    reverse = cumsum->is_reverse();

    if (getOriginalInputsNumber() == numOfInputs) {
        const auto axis_shape = cumsum->get_input_partial_shape(AXIS);
        if (axis_shape.is_dynamic() || !ngraph::is_scalar(axis_shape.to_shape()))
            IE_THROW() << errorPrefix << " doesn't support 'axis' input tensor with non scalar rank";
    }

    if (dataShape != getOutputShapeAtPort(0))
        IE_THROW() << errorPrefix << " has different 'data' input and output dimensions";
}

void CumSum::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    dataPrecision = getOriginalInputPrecisionAtPort(CUM_SUM_DATA);
    if (!one_of(dataPrecision, Precision::I8, Precision::U8, Precision::I16, Precision::BF16, Precision::I32, Precision::FP32, Precision::I64, Precision::U64))
        IE_THROW() << errorPrefix << " has unsupported 'data' input precision: " << dataPrecision.name();

    if (inputShapes.size() == numOfInputs) {
        const auto &axisTensorPrec = getOriginalInputPrecisionAtPort(AXIS);
        if (axisTensorPrec != Precision::I32 && axisTensorPrec != Precision::I64)
            IE_THROW() << errorPrefix << " has unsupported 'axis' input precision: " << axisTensorPrec.name();
    }

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputShapes.size());
    inDataConf.emplace_back(LayoutType::ncsp, dataPrecision);
    for (int i = 1; i < inputShapes.size(); ++i)
        inDataConf.emplace_back(LayoutType::ncsp, Precision::I32);

    addSupportedPrimDesc(inDataConf,
                         {{LayoutType::ncsp, dataPrecision}},
                         impl_desc_type::ref_any);
}

void CumSum::execute(dnnl::stream strm) {
    if (inputShapes.size() == numOfInputs)
        axis = getAxis(getParentEdgeAt(AXIS)->getMemory(), getParentEdgeAt(CUM_SUM_DATA)->getMemory());

    OV_SWITCH(intel_cpu, CumSumExecute, this, dataPrecision,
              OV_CASE(Precision::I8, int8_t),
              OV_CASE(Precision::U8, uint8_t),
              OV_CASE(Precision::I16, int16_t),
              OV_CASE(Precision::BF16, bfloat16_t),
              OV_CASE(Precision::I32, int32_t),
              OV_CASE(Precision::FP32, float),
              OV_CASE(Precision::I64, int64_t),
              OV_CASE(Precision::U64, uint64_t))
}

template <typename dataType>
void CumSum::exec() {
    const auto *input = reinterpret_cast<const dataType *>(getParentEdgeAt(CUM_SUM_DATA)->getMemoryPtr()->GetPtr());
    auto *output = reinterpret_cast<dataType *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());
    const VectorDims strides = getParentEdgeAt(CUM_SUM_DATA)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();

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
void CumSum::cumSum(const dataType *input, dataType *output, const VectorDims &strides) {
    SizeVector iterationRange(numOfDims - 1);
    size_t j = 0;
    const auto &shape = getParentEdgesAtPort(CUM_SUM_DATA)[0]->getMemory().getStaticDims();
    for (size_t i = 0; i < shape.size(); i++) {
        if (i == axis)
            continue;
        iterationRange[j++] = shape[i];
    }
    size_t work_amount_dst = std::accumulate(iterationRange.begin(), iterationRange.end(), size_t(1), std::multiplies<size_t>());
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(numOfDims - 1, 0);
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

            const dataType *inputStart = input + startOffset;
            dataType *outputStart = output + startOffset;

            size_t offset = strides[axis];
            if (reverse) {
                if (exclusive) {
                    outputStart[offset*(shape[axis] - 1)] = 0;
                    for (int64_t i = shape[axis] - 2; i >= 0; i--) {
                        outputStart[i*offset] = inputStart[(i+1)*offset] + outputStart[(i+1)*offset];
                    }
                } else {
                    outputStart[offset*(shape[axis] - 1)] = inputStart[offset * (shape[axis] - 1)];
                    for (int64_t i = shape[axis] - 2; i >= 0; i--) {
                        outputStart[i*offset] = inputStart[i*offset] + outputStart[(i+1)*offset];
                    }
                }
            } else {
                if (exclusive) {
                    outputStart[0] = 0;
                    for (size_t i = 1; i < shape[axis]; i++) {
                        outputStart[i*offset] = inputStart[(i-1)*offset] + outputStart[(i-1)*offset];
                    }
                } else {
                    outputStart[0] = inputStart[0];
                    for (size_t i = 1; i < shape[axis]; i++) {
                        outputStart[i*offset] = inputStart[i*offset] + outputStart[(i-1)*offset];
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

inline size_t CumSum::getStartOffset(const std::vector<size_t> &forStartOffset, const std::vector<size_t>& strides) const {
    size_t startOffset = 0;
    for (size_t idx = 0; idx < forStartOffset.size(); ++idx) {
        startOffset += forStartOffset[idx] * strides[idx];
    }
    return startOffset;
}

size_t CumSum::getAxis(const Memory& _axis, const Memory& _data) const {
    const auto& axisPrecision = _axis.getDesc().getPrecision();
    const int64_t dataShapeSize = static_cast<int64_t>(_data.GetShape().getRank());
    int64_t axisValueFromBlob = 0;
    switch (axisPrecision) {
        case Precision::I32 : {
            const auto *axisPtr = reinterpret_cast<const int32_t *>(_axis.GetPtr());
            axisValueFromBlob = static_cast<int64_t>(axisPtr[0]);
            break;
        }
        case Precision::I64 : {
            const auto *axisPtr = reinterpret_cast<const int64_t *>(_axis.GetPtr());
            axisValueFromBlob = axisPtr[0];
            break;
        }
        default : {
            IE_THROW() << errorPrefix << "  doesn't support 'axis' input with precision: " << axisPrecision.name();
        }
    }
    if (axisValueFromBlob < -dataShapeSize || axisValueFromBlob > dataShapeSize - 1)
        IE_THROW() << errorPrefix << "  has axis with a value out of range: " << axisValueFromBlob;
    return axisValueFromBlob >= 0 ? axisValueFromBlob : (axisValueFromBlob + dataShapeSize);
}

bool CumSum::created() const {
    return getType() == Type::CumSum;
}

bool CumSum::needPrepareParams() const {
    return false;
}

void CumSum::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
