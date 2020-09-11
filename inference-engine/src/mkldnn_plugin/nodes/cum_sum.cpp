// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "list.hpp"
#include "base.hpp"

#include <string>
#include <vector>
#include "ie_parallel.hpp"
#include "ie_precision.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class CumSumImpl: public ExtLayerBase {
    enum { CUM_SUM_DATA, AXIS, numOfInputs };
    enum { N, C, D, H, W, numOfDims };
    bool exclusive;
    bool reverse;
    size_t axis = 0;
    std::vector<size_t> shape5d;

public:
    explicit CumSumImpl(const CNNLayer* layer) {
        try {
            layerName = layer->name;
            if ((layer->insData.size() != numOfInputs && layer->insData.size() != (numOfInputs - 1)) || layer->outData.size() != 1)
                THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "' has incorrect number of input/output edges!";

            const auto &dataTensor = layer->insData[CUM_SUM_DATA].lock()->getTensorDesc();
            const auto &dataShape = dataTensor.getDims();
            if (dataShape.size() < 1 || dataShape.size() > 5) {
                THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "' doesn't support 'data' input tensor with rank: " << dataShape.size();
            }

            exclusive = layer->GetParamAsBool("exclusive", false);
            reverse = layer->GetParamAsBool("reverse", false);

            const auto& dataPrecision = dataTensor.getPrecision();
            if (dataPrecision != Precision::I8 && dataPrecision != Precision::U8 && dataPrecision != Precision::I16 && dataPrecision != Precision::I32 &&
                dataPrecision != Precision::FP32 && dataPrecision != Precision::I64 && dataPrecision != Precision::U64 && dataPrecision != Precision::BF16)
                THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "' has unsupported 'data' input precision: " << dataPrecision.name();

            if (layer->insData.size() == numOfInputs) {
                const auto& axisTensor = layer->insData[AXIS].lock()->getTensorDesc();
                const auto& axisTensorPrec = layer->insData[AXIS].lock()->getTensorDesc().getPrecision();
                if (axisTensorPrec != Precision::I32 && axisTensorPrec != Precision::I64)
                    THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "' has unsupported 'axis' input precision: " << axisTensorPrec.name();

                const auto axisTensorRank = axisTensor.getDims().size();
                if (axisTensorRank != 0)
                    THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "' doesn't support 'axis' input tensor with rank: " << axisTensorRank;
            }

            if (dataShape != layer->outData[0]->getTensorDesc().getDims())
                THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "' has different 'data' input and output dimensions";

            shape5d = get5dShape(dataShape);

            LayerConfig config;
            for (size_t i = 0; i < layer->insData.size(); i++) {
                DataConfig inConfig;
                inConfig.inPlace = -1;
                inConfig.constant = false;

                Precision inPrecision = layer->insData[i].lock()->getTensorDesc().getPrecision();
                if (inPrecision == Precision::BF16)
                    inPrecision = Precision::FP32;
                const SizeVector& inDims = layer->insData[i].lock()->getTensorDesc().getDims();
                inConfig.desc = TensorDesc(inPrecision, inDims, InferenceEngine::TensorDesc::getLayoutByDims(inDims));

                config.inConfs.push_back(inConfig);
            }
            DataConfig outConfig;
            outConfig.inPlace = -1;
            outConfig.constant = false;
            Precision outPrecision = layer->insData[CUM_SUM_DATA].lock()->getTensorDesc().getPrecision();
            if (outPrecision == Precision::BF16)
                outPrecision = Precision::FP32;
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
        if (inputs.size() == numOfInputs)
            axis = getAxis(inputs[AXIS], inputs[CUM_SUM_DATA]);

        const auto &dataPrecision = inputs[CUM_SUM_DATA]->getTensorDesc().getPrecision();
        switch (dataPrecision) {
            case Precision::I8   : { execImpl<int8_t>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            case Precision::U8   : { execImpl<uint8_t>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            case Precision::I16  : { execImpl<int16_t>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            case Precision::I32  : { execImpl<int32_t>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            case Precision::FP32 : { execImpl<float>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            case Precision::I64  : { execImpl<int64_t>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            case Precision::U64  : { execImpl<uint64_t>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            default : {
                if (resp) {
                    std::string errorMsg = "CumSum layer with name '" + layerName + "' has unsupported 'data' input precision: " + dataPrecision.name();
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
            }
        }
        return OK;
    }

private:
    template <typename dataType>
    void execImpl(const Blob::CPtr& _input, const Blob::Ptr& _output) {
        const auto *input = _input->cbuffer().as<const dataType *>() + _input->getTensorDesc().getBlockingDesc().getOffsetPadding();
        auto *output = _output->buffer().as<dataType *>() + _output->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const size_t offset = _input->getTensorDesc().getBlockingDesc().getStrides()[axis];

        if (reverse) {
            if (exclusive) {
                cumSum<true, true, dataType>(input, output, offset);
            } else {
                cumSum<true, false, dataType>(input, output, offset);
            }
        } else {
            if (exclusive) {
                cumSum<false, true, dataType>(input, output, offset);
            } else {
                cumSum<false, false, dataType>(input, output, offset);
            }
        }
    }

    template <bool reverse, bool exclusive, typename dataType>
    void cumSum(const dataType *input, dataType *output, const size_t &offset) {
        std::vector<size_t> iterationRange(numOfDims - 1);
        size_t j = 0;
        for (size_t i = 0; i < shape5d.size(); i++) {
            if (i == axis)
                continue;
            iterationRange[j++] = shape5d[i];
        }
        parallel_for4d(iterationRange[0], iterationRange[1], iterationRange[2], iterationRange[3], [&](size_t ir0, size_t ir1, size_t ir2, size_t ir3) {
            std::vector<size_t> forStartOffset;
            forStartOffset.push_back(ir0); forStartOffset.push_back(ir1); forStartOffset.push_back(ir2); forStartOffset.push_back(ir3);
            forStartOffset.insert(forStartOffset.begin() + axis, 0);
            size_t startOffset = getStartOffset(forStartOffset);

            const dataType *inputStart = input + startOffset;
            dataType *outputStart = output + startOffset;

            if (reverse) {
                if (exclusive) {
                    outputStart[offset*(shape5d[axis] - 1)] = 0;
                    for (int64_t i = shape5d[axis] - 2; i >= 0; i--) {
                        outputStart[i*offset] = inputStart[(i+1)*offset] + outputStart[(i+1)*offset];
                    }
                } else {
                    outputStart[offset*(shape5d[axis] - 1)] = inputStart[offset*(shape5d[axis] - 1)];
                    for (int64_t i = shape5d[axis] - 2; i >= 0; i--) {
                        outputStart[i*offset] = inputStart[i*offset] + outputStart[(i+1)*offset];
                    }
                }
            } else {
                if (exclusive) {
                    outputStart[0] = 0;
                    for (size_t i = 1; i < shape5d[axis]; i++) {
                        outputStart[i*offset] = inputStart[(i-1)*offset] + outputStart[(i-1)*offset];
                    }
                } else {
                    outputStart[0] = inputStart[0];
                    for (size_t i = 1; i < shape5d[axis]; i++) {
                        outputStart[i*offset] = inputStart[i*offset] + outputStart[(i-1)*offset];
                    }
                }
            }
        });
    }

    size_t getStartOffset(std::vector<size_t> &forStartOffset) {
        return forStartOffset[N]*shape5d[C]*shape5d[D]*shape5d[H]*shape5d[W] + forStartOffset[C]*shape5d[D]*shape5d[H]*shape5d[W] +
               forStartOffset[D]*shape5d[H]*shape5d[W] + forStartOffset[H]*shape5d[W] + forStartOffset[W];
    }

    size_t getAxis(const Blob::CPtr& _axis, const Blob::CPtr& _data) {
        const auto& axisPrecision = _axis->getTensorDesc().getPrecision();
        const int64_t dataShapeSize = static_cast<int64_t>(_data->getTensorDesc().getDims().size());
        int64_t axisValueFromBlob;
        switch (axisPrecision) {
            case Precision::I32 : {
                const auto *axisPtr = _axis->cbuffer().as<const int32_t *>();
                axisValueFromBlob = static_cast<int64_t>(axisPtr[0]);
                break;
            }
            case Precision::I64 : {
                const auto *axisPtr = _axis->cbuffer().as<const int64_t *>();
                axisValueFromBlob = axisPtr[0];
                break;
            }
            default : {
                THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "'  doesn't support 'axis' input with precision: " << axisPrecision.name();
            }
        }
        if (axisValueFromBlob < -dataShapeSize || axisValueFromBlob > dataShapeSize - 1)
            THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "'  has axis with a value out of range: " << axisValueFromBlob;
        return axisValueFromBlob >= 0 ? axisValueFromBlob : (axisValueFromBlob + dataShapeSize);
    }

    std::vector<size_t> get5dShape(const SizeVector& dims) {
        std::vector<size_t> shape5d(numOfDims, 1);
        for (size_t i = 0; i < dims.size(); i++)
            shape5d[i] = dims[i];
        return shape5d;
    }

private:
    std::string layerName;
};

REG_FACTORY_FOR(CumSumImpl, CumSum);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
