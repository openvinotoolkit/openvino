// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <string>
#include <vector>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class GatherElementsImpl: public ExtLayerBase {
public:
    explicit GatherElementsImpl(const CNNLayer* layer) : strideAx1Diff_(0) {
        errorPrefix_ = std::string("Layer GatherElements with name '") + layer->name + "'";

        if (layer->insData.size() != 2 || layer->outData.size() != 1)
            THROW_IE_EXCEPTION << errorPrefix_ << " has invalid number of input/output edges.";

        auto inputData = layer->insData[dataIndex_].lock();
        auto indices = layer->insData[indicesIndex_].lock();
        if (!inputData || !indices)
            THROW_IE_EXCEPTION << errorPrefix_ << " has nullable inputs.";

        const auto& dataDims = inputData->getTensorDesc().getDims();
        const auto& indicesDims = indices->getTensorDesc().getDims();
        if (dataDims.size() != indicesDims.size())
            THROW_IE_EXCEPTION << errorPrefix_ << " has invalid input shapes. Inputs 'Data' and 'Indices' must have equal ranks.";

        Precision dataPrecision = inputData->getTensorDesc().getPrecision();
        if (dataPrecision.size() != sizeof(PrecisionTrait<Precision::I32>::value_type) &&
                dataPrecision.size() != sizeof(PrecisionTrait<Precision::I16>::value_type) &&
                dataPrecision.size() != sizeof(PrecisionTrait<Precision::I8>::value_type)) {
            THROW_IE_EXCEPTION << errorPrefix_ << " has unsupported 'inputData' input precision: " << dataPrecision;
        }

        Precision indicesPrecision = indices->getTensorDesc().getPrecision();
        if (indicesPrecision != Precision::I32) {
            THROW_IE_EXCEPTION << errorPrefix_ << " has unsupported 'indices' input precision: " << indicesPrecision;
        }

        dataTypeSize_ = dataPrecision.size();

        int axis = layer->GetParamAsInt("axis");
        if (axis < 0)
            axis += dataDims.size();
        if (axis < 0 || axis >= static_cast<int>(dataDims.size()))
            THROW_IE_EXCEPTION << errorPrefix_ << " has invalid axis attribute: " << axis;
        axis_ = axis;

        auto& outputData = layer->outData[0];
        strideAxDst_ = outputData->getTensorDesc().getBlockingDesc().getStrides()[axis_];
        dstAxDim_ = outputData->getTensorDesc().getDims()[axis_];
        if (axis_ > 0) {
            strideAx1Diff_ = inputData->getTensorDesc().getBlockingDesc().getStrides()[axis_ - 1] -
                    outputData->getTensorDesc().getBlockingDesc().getStrides()[axis_ - 1];
        }

        LayerConfig config;
        DataConfig dataConfig, indicesConfig, outConfig;
        dataConfig.desc = TensorDesc(dataPrecision, dataDims,
            inputData->getTensorDesc().getLayoutByDims(dataDims));
        config.inConfs.push_back(dataConfig);
        indicesConfig.desc = TensorDesc(Precision::I32, indicesDims,
            indices->getTensorDesc().getLayoutByDims(indicesDims));
        config.inConfs.push_back(indicesConfig);

        const auto& outDims = outputData->getTensorDesc().getDims();
        outConfig.desc = TensorDesc(dataPrecision, outDims,
                outputData->getTensorDesc().getLayoutByDims(outDims));
        config.outConfs.push_back(outConfig);

        config.dynBatchSupport = false;

        confs.push_back(config);
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (dataTypeSize_) {
            case sizeof(PrecisionTrait<Precision::I32>::value_type):
                return directExecution<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
            case sizeof(PrecisionTrait<Precision::I16>::value_type):
                return directExecution<PrecisionTrait<Precision::I16>::value_type>(inputs, outputs, resp);
            case sizeof(PrecisionTrait<Precision::I8>::value_type):
                return directExecution<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs, resp);
            default:
                std::string errMsg = errorPrefix_ + " has inputData input with unsupported precision: " +
                    inputs[dataIndex_]->getTensorDesc().getPrecision().name();
                errMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                return GENERAL_ERROR;
        }
    }

protected:
    template <typename dataType>
    StatusCode directExecution(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept {
        const dataType* srcData = inputs[dataIndex_]->cbuffer().as<const dataType*>() +
            inputs[dataIndex_]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* indices = inputs[indicesIndex_]->cbuffer().as<const int*>() +
            inputs[indicesIndex_]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        dataType* dstData = outputs[0]->buffer().as<dataType*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const int outSize = outputs[0]->size();
        auto threadBody = [&](const int ithr, const int nthr) {
            int start(0lu), end(0lu);
            splitter(outSize, nthr, ithr, start, end);
            if (start >= end)
                return;

            int axStrideIt = start % strideAxDst_;
            int dstAxIdx = (start / strideAxDst_) % dstAxDim_;
            int dstShift0 = (start / strideAxDst_ / dstAxDim_) * strideAx1Diff_;

            for (size_t o = start; o < end; o++, axStrideIt++) {
                if (axStrideIt == strideAxDst_) {
                    axStrideIt = 0;
                    dstAxIdx++;
                    if (dstAxIdx == dstAxDim_) {
                        dstAxIdx = 0;
                        dstShift0 += strideAx1Diff_;
                    }
                }
                dstData[o] = srcData[o + dstShift0 + (indices[o] - dstAxIdx) * strideAxDst_];
            }
        };
        parallel_nt(0, threadBody);

        return OK;
    }

    const size_t dataIndex_ = 0;
    const size_t indicesIndex_ = 1;

    size_t axis_;
    size_t dataTypeSize_;
    int strideAxDst_;
    int dstAxDim_;
    int strideAx1Diff_;
    std::string errorPrefix_;
};

REG_FACTORY_FOR(GatherElementsImpl, GatherElements);
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
