// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "list.hpp"

#include <string>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include "ie_parallel.hpp"
#include "ie_precision.hpp"
#include <ie_ngraph_utils.hpp>
#include "mkldnn_cum_sum_node.h"
#include "cpu_memory_desc_utils.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNCumSumNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
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

MKLDNNCumSumNode::MKLDNNCumSumNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "CumSum layer with name '" + op->get_friendly_name() + "' ";

    if ((getOriginalInputsNumber() != numOfInputs && getOriginalInputsNumber() != (numOfInputs - 1)) || getOriginalOutputsNumber() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    const auto &dataShape = op->get_input_shape(CUM_SUM_DATA);
    if (dataShape.size() < 1) {
        IE_THROW() << errorPrefix << " doesn't support 'data' input tensor with rank: " << dataShape.size();
    }
    numOfDims = dataShape.size();

    const auto cumsum = std::dynamic_pointer_cast<const ngraph::opset3::CumSum>(op);
    exclusive = cumsum->is_exclusive();
    reverse = cumsum->is_reverse();

    if (getOriginalInputsNumber() == numOfInputs) {
        if (!ngraph::is_scalar(cumsum->get_input_shape(AXIS)))
            IE_THROW() << errorPrefix << " doesn't support 'axis' input tensor with non scalar rank";
    }

    if (dataShape != cumsum->get_output_shape(0))
        IE_THROW() << errorPrefix << " has different 'data' input and output dimensions";

    shape = dataShape;
}

void MKLDNNCumSumNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    dataPrecision = getOriginalInputPrecisionAtPort(CUM_SUM_DATA);
    if (dataPrecision != Precision::I8 && dataPrecision != Precision::U8 && dataPrecision != Precision::I16 && dataPrecision != Precision::I32 &&
        dataPrecision != Precision::FP32 && dataPrecision != Precision::I64 && dataPrecision != Precision::U64 && dataPrecision != Precision::BF16)
        IE_THROW() << errorPrefix << " has unsupported 'data' input precision: " << dataPrecision.name();

    if (getOriginalInputsNumber() == numOfInputs) {
        const auto &axisTensorPrec = getOriginalInputPrecisionAtPort(AXIS);
        if (axisTensorPrec != Precision::I32 && axisTensorPrec != Precision::I64)
            IE_THROW() << errorPrefix << " has unsupported 'axis' input precision: " << axisTensorPrec.name();
    }

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(getOriginalInputsNumber());
    inDataConf.emplace_back(GeneralLayout::ncsp, dataPrecision);
    for (int i = 1; i < getOriginalInputsNumber(); ++i)
        inDataConf.emplace_back(GeneralLayout::ncsp, Precision::I32);

    addSupportedPrimDesc(inDataConf,
                         {{GeneralLayout::ncsp, dataPrecision}},
                         impl_desc_type::ref_any);
}

void MKLDNNCumSumNode::execute(mkldnn::stream strm) {
    if (inputShapes.size() == numOfInputs)
        axis = getAxis(getParentEdgeAt(AXIS)->getMemory(), getParentEdgeAt(CUM_SUM_DATA)->getMemory());

    switch (dataPrecision) {
        case Precision::I8   : {
            exec<int8_t>();
            break;
        }
        case Precision::U8   : {
            exec<uint8_t>();
            break;
        }
        case Precision::I16  : {
            exec<int16_t>();
            break;
        }
        case Precision::I32  : {
            exec<int32_t>();
            break;
        }
        case Precision::FP32 : {
            exec<float>();
            break;
        }
        case Precision::I64  : {
            exec<int64_t>();
            break;
        }
        case Precision::U64  : {
            exec<uint64_t>();
            break;
        }
        default : {
            std::string errorMsg = errorPrefix + " has unsupported 'data' input precision: " + dataPrecision.name();
            IE_THROW() << errorMsg;
        }
    }
}


template <typename dataType>
void MKLDNNCumSumNode::exec() {
    const auto *input = reinterpret_cast<const dataType *>(getParentEdgeAt(CUM_SUM_DATA)->getMemoryPtr()->GetPtr());
    auto *output = reinterpret_cast<dataType *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());
    const std::vector<size_t> strides = getParentEdgeAt(CUM_SUM_DATA)->getMemory().GetDescWithType<BlockedMemoryDesc>().getStrides();

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
void MKLDNNCumSumNode::cumSum(const dataType *input, dataType *output, const std::vector<size_t> &strides) {
    SizeVector iterationRange(numOfDims - 1);
    size_t j = 0;
    for (size_t i = 0; i < shape.size(); i++) {
        if (i == axis)
            continue;
        iterationRange[j++] = shape[i];
    }
    size_t work_amount_dst = std::accumulate(iterationRange.begin(), iterationRange.end(), 1, std::multiplies<size_t>());
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

void MKLDNNCumSumNode::parallelItInit(size_t start, std::vector<size_t>& counters, const std::vector<size_t>& iterationRange) {
    auto itCounter = counters.rbegin();
    auto itWork = iterationRange.rbegin();
    while (itCounter != counters.rend() && itWork != iterationRange.rend()) {
        *itCounter = start % *itWork;
        start /= *itWork;
        ++itCounter;
        ++itWork;
    }
}

inline void MKLDNNCumSumNode::parallelItStep(std::vector<size_t>& counters, const std::vector<size_t>& iterationRange) {
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

inline size_t MKLDNNCumSumNode::getStartOffset(const std::vector<size_t> &forStartOffset, const std::vector<size_t>& strides) const {
    size_t startOffset = 0;
    for (size_t idx = 0; idx < forStartOffset.size(); ++idx) {
        startOffset += forStartOffset[idx] * strides[idx];
    }
    return startOffset;
}

size_t MKLDNNCumSumNode::getAxis(const MKLDNNMemory& _axis, const MKLDNNMemory& _data) const {
    const auto& axisPrecision = _axis.GetDesc().getPrecision();
    const int64_t dataShapeSize = static_cast<int64_t>(_data.GetDesc().getShape().getRank());
    int64_t axisValueFromBlob;
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

bool MKLDNNCumSumNode::created() const {
    return getType() == CumSum;
}

REG_MKLDNN_PRIM_FOR(MKLDNNCumSumNode, CumSum)
