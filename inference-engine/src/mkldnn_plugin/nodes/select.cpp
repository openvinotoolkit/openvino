// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <string>
#include <vector>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class SelectImpl: public ExtLayerBase {
    enum { CONDITION, THEN, ELSE, numOfInputs };
    enum { N, C, D, H, W, numOfDims };

    std::string broadcast;
    std::vector<size_t> resDims;
    std::vector<size_t> resOffset;
    std::vector<size_t> condOffset;
    std::vector<size_t> thenOffset;
    std::vector<size_t> elseOffset;

public:
    explicit SelectImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != numOfInputs || layer->outData.size() != 1)
                THROW_IE_EXCEPTION << "Select layer with name '" << layer->name << "' has incorrect number of input/output edges!";

            broadcast = layer->GetParamAsString("auto_broadcast", "numpy");

            if (layer->insData[THEN].lock()->getTensorDesc().getPrecision() != layer->insData[ELSE].lock()->getTensorDesc().getPrecision())
                THROW_IE_EXCEPTION << "Select layer with name '" << layer->name << "' has different precisions on 'Then' and 'Else' inputs";

            const auto& conditionPrecision = layer->insData[CONDITION].lock()->getTensorDesc().getPrecision();
            if (conditionPrecision != Precision::BOOL && conditionPrecision != Precision::I32  && conditionPrecision != Precision::U8)
                THROW_IE_EXCEPTION << "Select layer with name '" << layer->name << "' has unsupported precision: " << conditionPrecision
                                                                                                                << " on 'Condition' input";

            const auto& inputPrecisionSize = layer->insData[THEN].lock()->getTensorDesc().getPrecision().size();
            if (inputPrecisionSize != 1 && inputPrecisionSize != 2 && inputPrecisionSize != 4 && inputPrecisionSize != 8)
                THROW_IE_EXCEPTION << "Select layer with name '" << layer->name << "' has unsupported precision: " <<
                                                        layer->insData[THEN].lock()->getTensorDesc().getPrecision() << " on 'Then' and 'Else' inputs";

            const auto &conditionShapes = layer->insData[CONDITION].lock()->getTensorDesc().getDims();
            const auto &thenShapes = layer->insData[THEN].lock()->getTensorDesc().getDims();
            const auto &elseShapes = layer->insData[ELSE].lock()->getTensorDesc().getDims();
            const auto &outputShapes = layer->outData[0]->getTensorDesc().getDims();

            if (broadcast != "none" && broadcast != "numpy")
                THROW_IE_EXCEPTION << "Select layer with name '" << layer->name << "' has unsupported broadcast type: " << broadcast;

            if (broadcast == "none" && ((conditionShapes != outputShapes) || (thenShapes != outputShapes) || (elseShapes != outputShapes)))
                THROW_IE_EXCEPTION << "Select layer with name '" << layer->name << "' and auto_broadcast='none' has input shapes mismatch";

            if (broadcast == "numpy") {
                if (outputShapes.size() < conditionShapes.size() || outputShapes.size() < thenShapes.size() || outputShapes.size() < elseShapes.size())
                    THROW_IE_EXCEPTION << "Select layer with name '" << layer->name << "' and auto_broadcast='numpy' has incompatible input and output shapes";

                for (int condIt = conditionShapes.size() - 1, outIt = outputShapes.size() - 1; condIt >= 0; condIt--, outIt--)
                        if (conditionShapes[condIt] != outputShapes[outIt] && conditionShapes[condIt] != 1)
                            THROW_IE_EXCEPTION << "Select layer with name '" << layer->name
                                                                        << "' and auto_broadcast='numpy' has incompatible 'Condition' input and output shapes";

                for (int thenIt = thenShapes.size() - 1, outIt = outputShapes.size() - 1; thenIt >= 0; thenIt--, outIt--)
                        if (thenShapes[thenIt] != outputShapes[outIt] && thenShapes[thenIt] != 1)
                            THROW_IE_EXCEPTION << "Select layer with name '" << layer->name
                                                                            << "' and auto_broadcast='numpy' has incompatible 'Then' input and output shapes";


                for (int elseIt = elseShapes.size() - 1, outIt = outputShapes.size() - 1; elseIt >= 0; elseIt--, outIt--)
                        if (elseShapes[elseIt] != outputShapes[outIt] && elseShapes[elseIt] != 1)
                            THROW_IE_EXCEPTION << "Select layer with name '" << layer->name
                                                                             << "' and auto_broadcast='numpy' has incompatible 'Else' input and output shapes";
            }

            resDims.resize(numOfDims, 1);
            std::copy(std::begin(outputShapes), std::end(outputShapes), std::begin(resDims) + (numOfDims - outputShapes.size()));
            if (broadcast == "numpy") {
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

            LayerConfig config;
            for (size_t i = 0; i < numOfInputs; i++) {
                DataConfig inConfig;
                inConfig.inPlace = -1;
                inConfig.constant = false;

                Precision inPrecision = layer->insData[i].lock()->getTensorDesc().getPrecision();
                const SizeVector& inDims = layer->insData[i].lock()->getTensorDesc().getDims();
                inConfig.desc = TensorDesc(inPrecision, inDims, InferenceEngine::TensorDesc::getLayoutByDims(inDims));

                config.inConfs.push_back(inConfig);
            }

            DataConfig outConfig;
            outConfig.inPlace = -1;
            outConfig.constant = false;
            Precision outPrecision = layer->insData[1].lock()->getTensorDesc().getPrecision();
            const SizeVector& outDims = layer->outData[0]->getTensorDesc().getDims();
            outConfig.desc = TensorDesc(outPrecision, outDims, InferenceEngine::TensorDesc::getLayoutByDims(outDims));
            config.outConfs.push_back(outConfig);

            config.dynBatchSupport = false;
            confs.push_back(config);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
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

        if (broadcast == "none") {
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
