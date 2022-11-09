// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include "ie_parallel.hpp"
#include "select.h"
#include <nodes/common/blocked_desc_creator.h>
#include <ngraph/opsets/opset1.hpp>
#include <utils/general_utils.h>
#include "common/cpu_memcpy.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool Select::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto select = std::dynamic_pointer_cast<const ngraph::opset1::Select>(op);
        if (!select) {
            errorMessage = "Only opset1 Select operation is supported";
            return false;
        }
        const auto broadcast = select->get_auto_broadcast();
        if (!one_of(broadcast.m_type, ngraph::op::AutoBroadcastType::NONE, ngraph::op::AutoBroadcastType::NUMPY)) {
            errorMessage = "Does not support broadcast type: " + ngraph::as_string(broadcast.m_type);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Select::Select(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng,
        WeightsSharing::Ptr &cache) : Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "Select layer with name '" + op->get_friendly_name() + "'";
    const auto select = std::dynamic_pointer_cast<const ngraph::opset1::Select>(op);

    if (inputShapes.size() != numOfInputs || outputShapes.size() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    const auto broadcast = select->get_auto_broadcast();
    if (broadcast.m_type == ngraph::op::AutoBroadcastType::NONE) {
        broadcastType = SelectBroadcastType::NONE;
    } else if (broadcast.m_type == ngraph::op::AutoBroadcastType::NUMPY) {
        broadcastType = SelectBroadcastType::NUMPY;
    } else {
        IE_THROW() << errorPrefix << " has unsupported broadcast type: " + ngraph::as_string(broadcast.m_type);
    }

    const auto &inCondDims = getInputShapeAtPort(CONDITION).getDims();
    const auto &inThenDims = getInputShapeAtPort(THEN).getDims();
    const auto &inElseDims = getInputShapeAtPort(ELSE).getDims();
    const auto &outputDims = getOutputShapeAtPort(0).getDims();

    if (broadcastType == SelectBroadcastType::NONE && (!dimsEqualWeak(inCondDims, outputDims) || !dimsEqualWeak(inThenDims, outputDims) ||
                                                       !dimsEqualWeak(inElseDims, outputDims))) {
        IE_THROW() << errorPrefix << " and auto_broadcast='none' has input shapes mismatch";
    }

    if (broadcastType == SelectBroadcastType::NUMPY) {
        if (outputDims.size() < inCondDims.size() || outputDims.size() < inThenDims.size() || outputDims.size() < inElseDims.size())
            IE_THROW() << errorPrefix << " and auto_broadcast='numpy' has incompatible input and output shapes";

        for (int condIt = inCondDims.size() - 1, outIt = outputDims.size() - 1; condIt >= 0; condIt--, outIt--)
            if (!dimsEqualWeak(inCondDims[condIt], outputDims[outIt]) && !dimsEqualWeak(inCondDims[condIt], 1))
                IE_THROW() << errorPrefix << " and auto_broadcast='numpy' has incompatible 'Condition' input and output shapes";

        for (int thenIt = inThenDims.size() - 1, outIt = outputDims.size() - 1; thenIt >= 0; thenIt--, outIt--)
            if (!dimsEqualWeak(inThenDims[thenIt], outputDims[outIt]) && !dimsEqualWeak(inThenDims[thenIt], 1))
                IE_THROW() << errorPrefix << " and auto_broadcast='numpy' has incompatible 'Then' input and output shapes";

        for (int elseIt = inElseDims.size() - 1, outIt = outputDims.size() - 1; elseIt >= 0; elseIt--, outIt--)
            if (!dimsEqualWeak(inElseDims[elseIt], outputDims[outIt]) && !dimsEqualWeak(inElseDims[elseIt], 1))
                IE_THROW() << errorPrefix << " and auto_broadcast='numpy' has incompatible 'Else' input and output shapes";
    }

    resDims.resize(numOfDims, 1);
    if (broadcastType == SelectBroadcastType::NUMPY) {
        resOffset.resize(numOfDims);
        condOffset.resize(numOfDims);
        thenOffset.resize(numOfDims);
        elseOffset.resize(numOfDims);

        condDims.resize(numOfDims, 1);
        thenDims.resize(numOfDims, 1);
        elseDims.resize(numOfDims, 1);
    }
}

void Select::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto inputThenPrecision = getOriginalInputPrecisionAtPort(THEN);
    const auto inputElsePrecision = getOriginalInputPrecisionAtPort(ELSE);
    auto inputPrecision = inputThenPrecision;
    if (inputThenPrecision == Precision::BF16 || inputElsePrecision == Precision::BF16) {
        inputPrecision = Precision::BF16;
    } else if (inputThenPrecision != inputElsePrecision) {
        IE_THROW() << errorPrefix << " has different precisions on 'Then' and 'Else' inputs ";
    }

    const auto conditionPrecision = getOriginalInputPrecisionAtPort(CONDITION);
    if (conditionPrecision != Precision::BOOL && conditionPrecision != Precision::I32  && conditionPrecision != Precision::U8)
        IE_THROW() << errorPrefix << " has unsupported precision: " << conditionPrecision << " on 'Condition' input";

    const auto inputPrecisionSize = inputPrecision.size();
    if (inputPrecisionSize != 1 && inputPrecisionSize != 2 && inputPrecisionSize != 4 && inputPrecisionSize != 8)
        IE_THROW() << errorPrefix << " has unsupported precision: " << inputPrecision << " on 'Then' and 'Else' inputs";

    addSupportedPrimDesc({{LayoutType::ncsp, conditionPrecision},
                          {LayoutType::ncsp, inputPrecision},
                          {LayoutType::ncsp, inputPrecision}},
                         {{LayoutType::ncsp, inputPrecision}},
                         impl_desc_type::ref_any);
}

void Select::prepareParams() {
    const auto &_conditionDims = getParentEdgesAtPort(CONDITION)[0]->getMemory().getStaticDims();
    const auto &_thenDims = getParentEdgesAtPort(THEN)[0]->getMemory().getStaticDims();
    const auto &_elseDims = getParentEdgesAtPort(ELSE)[0]->getMemory().getStaticDims();
    const auto &_outputDims = getChildEdgesAtPort(0)[0]->getMemory().getStaticDims();

    std::fill(resDims.begin(), resDims.end(), 1);
    std::copy(std::begin(_outputDims), std::end(_outputDims), std::begin(resDims) + (numOfDims - _outputDims.size()));
    if (broadcastType == SelectBroadcastType::NUMPY) {
        std::fill(resOffset.begin(), resOffset.end(), 1);
        calcOutOffset(resOffset, resDims);

        std::fill(condDims.begin(), condDims.end(), 1);
        std::copy(std::begin(_conditionDims), std::end(_conditionDims), std::begin(condDims) + (numOfDims - _conditionDims.size()));
        std::fill(condOffset.begin(), condOffset.end(), 1);
        calcInOffset(condOffset, condDims, resDims);

        std::fill(thenDims.begin(), thenDims.end(), 1);
        std::copy(std::begin(_thenDims), std::end(_thenDims), std::begin(thenDims) + (numOfDims - _thenDims.size()));
        std::fill(thenOffset.begin(), thenOffset.end(), 1);
        calcInOffset(thenOffset, thenDims, resDims);

        std::fill(elseDims.begin(), elseDims.end(), 1);
        std::copy(std::begin(_elseDims), std::end(_elseDims), std::begin(elseDims) + (numOfDims - _elseDims.size()));
        std::fill(elseOffset.begin(), elseOffset.end(), 1);
        calcInOffset(elseOffset, elseDims, resDims);
    }
}

void Select::calcOutOffset(VectorDims& offset, const VectorDims& dims) {
    int k = 1;
    for (int i = dims.size() - 1; i >= 0; i--) {
        offset[i] = k;
        k *= dims[i];
    }
}

void Select::calcInOffset(VectorDims& offset, const VectorDims& inDims, const VectorDims& outDims) {
    int k = 1;
    for (int i = inDims.size() - 1; i >= 0; i--) {
        offset[i] = (inDims[i] == outDims[i]) ? k : 0;
        k *= inDims[i];
    }
}

template <typename COND_T, typename DATA_T>
void Select::execute_impl() {
    const auto *conditionData = reinterpret_cast<const COND_T *>(getParentEdgeAt(CONDITION)->getMemoryPtr()->GetPtr());
    const auto *thenData = reinterpret_cast<const DATA_T *>(getParentEdgeAt(THEN)->getMemoryPtr()->GetPtr());
    const auto *elseData = reinterpret_cast<const DATA_T *>(getParentEdgeAt(ELSE)->getMemoryPtr()->GetPtr());
    auto *dstData = reinterpret_cast<DATA_T *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    if (broadcastType == SelectBroadcastType::NONE) {
        size_t dstDataSize = std::accumulate(begin(resDims), end(resDims), size_t(1), std::multiplies<size_t>());
        parallel_for(dstDataSize, [&](size_t i) {
            dstData[i] = conditionData[i] ? thenData[i] : elseData[i];
        });
    } else {
        parallel_for4d(resDims[N], resDims[C], resDims[D], resDims[H], [&](int b, int c, int d, int h) {
            for (int w = 0; w < resDims[W]; w++) {
                size_t indexOut = b * resOffset[N] + c * resOffset[C] + d * resOffset[D] + h * resOffset[H] + w * resOffset[W];
                size_t indexCond = b * condOffset[N] + c * condOffset[C] + d * condOffset[D] + h * condOffset[H] + w * condOffset[W];
                size_t indexThen = b * thenOffset[N] + c * thenOffset[C] + d * thenOffset[D] + h * thenOffset[H] + w * thenOffset[W];
                size_t indexElse = b * elseOffset[N] + c * elseOffset[C] + d * elseOffset[D] + h * elseOffset[H] + w * elseOffset[W];
                dstData[indexOut] = conditionData[indexCond] ? thenData[indexThen] : elseData[indexElse];
            }
        });
    }
}

void Select::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Select::execute(dnnl::stream strm) {
    const size_t condPrecSize = getParentEdgeAt(CONDITION)->getMemory().getDesc().getPrecision().size();
    const size_t inputsPrecSize = getParentEdgeAt(THEN)->getMemory().getDesc().getPrecision().size();

    switch (condPrecSize) {
        case 1: {
            switch (inputsPrecSize) {
                case 1: { execute_impl<uint8_t, uint8_t>(); break; }
                case 2: { execute_impl<uint8_t, uint16_t>(); break; }
                case 4: { execute_impl<uint8_t, uint32_t>(); break; }
                case 8: { execute_impl<uint8_t, uint64_t>(); break; }
                default:
                    IE_THROW() << "Select layer doesn't support 'Then' and 'Else' inputs' precision: "
                                   + std::string(getParentEdgeAt(THEN)->getMemory().getDesc().getPrecision().name());
            }
            break;
        }
        case 4: {
            switch (inputsPrecSize) {
                case 1: { execute_impl<int32_t, uint8_t>(); break; }
                case 2: { execute_impl<int32_t, uint16_t>(); break; }
                case 4: { execute_impl<int32_t, uint32_t>(); break; }
                case 8: { execute_impl<int32_t, uint64_t>(); break; }
                default:
                    IE_THROW() << "Select layer doesn't support 'Then' and 'Else' inputs' precision: "
                                  + std::string(getParentEdgeAt(THEN)->getMemory().getDesc().getPrecision().name());
            }
            break;
        }
        default: {
                IE_THROW() << "Select layer doesn't support 'Condition' inputs' precision: "
                              + std::string(getParentEdgeAt(CONDITION)->getMemory().getDesc().getPrecision().name());
        }
    }
}

bool Select::created() const {
    return getType() == Type::Select;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
