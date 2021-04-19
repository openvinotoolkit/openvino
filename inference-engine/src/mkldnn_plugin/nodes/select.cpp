// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <string>
#include <vector>
#include "ie_parallel.hpp"
#include <utils/general_utils.h>
#include <ngraph/opsets/opset1.hpp>

using namespace MKLDNNPlugin;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class SelectImpl: public ExtLayerBase {
    enum { CONDITION, THEN, ELSE, numOfInputs };
    enum { N, C, D, H, W, numOfDims };
    enum class SelectBroadcastType {
        NONE,
        NUMPY
    };

    SelectBroadcastType broadcastType;
    std::vector<size_t> resDims;
    std::vector<size_t> resOffset;
    std::vector<size_t> condOffset;
    std::vector<size_t> thenOffset;
    std::vector<size_t> elseOffset;

    std::string errorPrefix;

    bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            const auto select = std::dynamic_pointer_cast<const ngraph::opset1::Select>(op);
            if (!select) {
                errorMessage = "Only opset1 Select operation is supported";
                return false;
            }
            const auto broadcast = select->get_auto_broadcast();
            if (!one_of(broadcast, ngraph::op::AutoBroadcastSpec::NONE, ngraph::op::AutoBroadcastSpec::NUMPY)) {
                errorMessage = "Does not support broadcast type: " + ngraph::as_string(broadcast.m_type);
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

public:
    explicit SelectImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            errorPrefix = "Select layer with name '" + op->get_friendly_name() + "'";
            const auto select = std::dynamic_pointer_cast<const ngraph::opset1::Select>(op);

            if (op->get_input_size() != numOfInputs || op->get_output_size() != 1)
                IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

            const auto broadcast = select->get_auto_broadcast();
            if (broadcast == ngraph::op::AutoBroadcastSpec::NONE) {
                broadcastType = SelectBroadcastType::NONE;
            } else if (broadcast == ngraph::op::AutoBroadcastSpec::NUMPY) {
                broadcastType = SelectBroadcastType::NUMPY;
            } else {
                IE_THROW() << errorPrefix << " has unsupported broadcast type: " + ngraph::as_string(broadcast.m_type);
            }

            const auto inputThenPrecision = details::convertPrecision(op->get_input_element_type(THEN));
            const auto inputElsePrecision = details::convertPrecision(op->get_input_element_type(ELSE));
            auto inputPrecision = inputThenPrecision;
            if (inputThenPrecision == Precision::BF16 || inputElsePrecision == Precision::BF16) {
                inputPrecision = Precision::BF16;
            } else if (inputThenPrecision != inputElsePrecision) {
                IE_THROW() << errorPrefix << " has different precisions on 'Then' and 'Else' inputs ";
            }

            const auto conditionPrecision = details::convertPrecision(op->get_input_element_type(CONDITION));
            if (conditionPrecision != Precision::BOOL && conditionPrecision != Precision::I32  && conditionPrecision != Precision::U8)
                IE_THROW() << errorPrefix << " has unsupported precision: " << conditionPrecision << " on 'Condition' input";

            const auto inputPrecisionSize = inputPrecision.size();
            if (inputPrecisionSize != 1 && inputPrecisionSize != 2 && inputPrecisionSize != 4 && inputPrecisionSize != 8)
                IE_THROW() << errorPrefix << " has unsupported precision: " << inputPrecision << " on 'Then' and 'Else' inputs";

            auto conditionShapes = op->get_input_shape(CONDITION);
            if (ngraph::is_scalar(conditionShapes))
                conditionShapes = ngraph::Shape{1};
            auto thenShapes = op->get_input_shape(THEN);
            if (ngraph::is_scalar(thenShapes))
                thenShapes = ngraph::Shape{1};
            auto elseShapes = op->get_input_shape(ELSE);
            if (ngraph::is_scalar(elseShapes))
                elseShapes = ngraph::Shape{1};
            auto outputShapes = op->get_output_shape(0);
            if (ngraph::is_scalar(outputShapes))
                outputShapes = ngraph::Shape{1};

            if (broadcastType == SelectBroadcastType::NONE && ((conditionShapes != outputShapes) || (thenShapes != outputShapes) ||
                (elseShapes != outputShapes)))
                IE_THROW() << errorPrefix << " and auto_broadcast='none' has input shapes mismatch";

            if (broadcastType == SelectBroadcastType::NUMPY) {
                if (outputShapes.size() < conditionShapes.size() || outputShapes.size() < thenShapes.size() || outputShapes.size() < elseShapes.size())
                    IE_THROW() << errorPrefix << " and auto_broadcast='numpy' has incompatible input and output shapes";

                for (int condIt = conditionShapes.size() - 1, outIt = outputShapes.size() - 1; condIt >= 0; condIt--, outIt--)
                        if (conditionShapes[condIt] != outputShapes[outIt] && conditionShapes[condIt] != 1)
                            IE_THROW() << errorPrefix << " and auto_broadcast='numpy' has incompatible 'Condition' input and output shapes";

                for (int thenIt = thenShapes.size() - 1, outIt = outputShapes.size() - 1; thenIt >= 0; thenIt--, outIt--)
                        if (thenShapes[thenIt] != outputShapes[outIt] && thenShapes[thenIt] != 1)
                            IE_THROW() << errorPrefix << " and auto_broadcast='numpy' has incompatible 'Then' input and output shapes";

                for (int elseIt = elseShapes.size() - 1, outIt = outputShapes.size() - 1; elseIt >= 0; elseIt--, outIt--)
                        if (elseShapes[elseIt] != outputShapes[outIt] && elseShapes[elseIt] != 1)
                            IE_THROW() << errorPrefix << " and auto_broadcast='numpy' has incompatible 'Else' input and output shapes";
            }

            resDims.resize(numOfDims, 1);
            std::copy(std::begin(outputShapes), std::end(outputShapes), std::begin(resDims) + (numOfDims - outputShapes.size()));
            if (broadcastType == SelectBroadcastType::NUMPY) {
                calcOutOffset(resOffset, resDims);

                std::vector<size_t> condDims(numOfDims, 1);
                std::copy(std::begin(conditionShapes), std::end(conditionShapes), std::begin(condDims) + (numOfDims - conditionShapes.size()));
                calcInOffset(condOffset, condDims, resDims);

                std::vector<size_t> thenDims(numOfDims, 1);
                std::copy(std::begin(thenShapes), std::end(thenShapes), std::begin(thenDims) + (numOfDims - thenShapes.size()));
                calcInOffset(thenOffset, thenDims, resDims);

                std::vector<size_t> elseDims(numOfDims, 1);
                std::copy(std::begin(elseShapes), std::end(elseShapes), std::begin(elseDims) + (numOfDims - elseShapes.size()));
                calcInOffset(elseOffset, elseDims, resDims);
            }

            addConfig(op, {{TensorDescCreatorTypes::ncsp, conditionPrecision},
                           {TensorDescCreatorTypes::ncsp, inputPrecision},
                           {TensorDescCreatorTypes::ncsp, inputPrecision}},
                          {{TensorDescCreatorTypes::ncsp, inputPrecision}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        auto &outputData = outputs[0];
        const size_t condPrecSize = inputs[CONDITION]->getTensorDesc().getPrecision().size();
        const size_t inputsPrecSize = inputs[THEN]->getTensorDesc().getPrecision().size();

        switch (condPrecSize) {
            case 1: {
                switch (inputsPrecSize) {
                    case 1: { execute_impl<uint8_t, uint8_t>(inputs, outputData); break; }
                    case 2: { execute_impl<uint8_t, uint16_t>(inputs, outputData); break; }
                    case 4: { execute_impl<uint8_t, uint32_t>(inputs, outputData); break; }
                    case 8: { execute_impl<uint8_t, uint64_t>(inputs, outputData); break; }
                    default: {
                        if (resp) {
                            std::string errorMsg = "Select layer doesn't support 'Then' and 'Else' inputs' precision: "
                                                                                        + std::string(inputs[THEN]->getTensorDesc().getPrecision().name());
                                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                        }
                        return GENERAL_ERROR;
                    }
                }
                break;
            }
            case 4: {
                switch (inputsPrecSize) {
                    case 1: { execute_impl<int32_t, uint8_t>(inputs, outputData); break; }
                    case 2: { execute_impl<int32_t, uint16_t>(inputs, outputData); break; }
                    case 4: { execute_impl<int32_t, uint32_t>(inputs, outputData); break; }
                    case 8: { execute_impl<int32_t, uint64_t>(inputs, outputData); break; }
                    default: {
                        if (resp) {
                            std::string errorMsg = "Select layer doesn't support 'Then' and 'Else' inputs' precision: "
                                                                                        + std::string(inputs[THEN]->getTensorDesc().getPrecision().name());
                                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                        }
                        return GENERAL_ERROR;
                    }
                }
                break;
            }
            default: {
                if (resp) {
                    std::string errorMsg = "Select layer doesn't support 'Condition' inputs' precision: "
                                                                                    + std::string(inputs[CONDITION]->getTensorDesc().getPrecision().name());
                        errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
            }
        }

        return OK;
    }

private:
    void calcOutOffset(std::vector<size_t>& offset, const std::vector<size_t>& dims) {
        offset.resize(numOfDims);
        int k = 1;
        for (int i = dims.size() - 1; i >= 0; i--) {
            offset[i] = k;
            k *= dims[i];
        }
    }

    void calcInOffset(std::vector<size_t>& offset, const std::vector<size_t>& inDims, const std::vector<size_t>& outDims) {
        offset.resize(numOfDims);
        int k = 1;
        for (int i = inDims.size() - 1; i >= 0; i--) {
            offset[i] = (inDims[i] == outDims[i]) ? k : 0;
            k *= inDims[i];
        }
    }

    template <typename COND_T, typename DATA_T>
    void execute_impl(std::vector<Blob::Ptr>& inputs, Blob::Ptr& output) noexcept {
        auto *conditionData = inputs[CONDITION]->cbuffer().as<const COND_T *>() + inputs[CONDITION]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        auto *thenData = inputs[THEN]->cbuffer().as<const DATA_T *>() + inputs[THEN]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        auto *elseData = inputs[ELSE]->cbuffer().as<const DATA_T *>() + inputs[ELSE]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        auto *dstData = output->buffer().as<DATA_T *>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();

        if (broadcastType == SelectBroadcastType::NONE) {
            size_t dstDataSize = std::accumulate(begin(resDims), end(resDims), 1, std::multiplies<size_t>());
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
};

REG_FACTORY_FOR(SelectImpl, Select);
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
