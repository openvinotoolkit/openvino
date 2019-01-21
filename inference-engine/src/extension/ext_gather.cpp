// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include <limits>
#include "ie_parallel.hpp"
#include "simple_copy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

inline void clipping(int *idx, const int min, const int max) {
    (*idx) = ((*idx) > min) ? (*idx) : min;
    (*idx) = ((*idx) < max) ? (*idx) : (max - 1);
    return;
}

class GatherImpl: public ILayerExecImpl {
public:
    StatusCode init(LayerConfig& config, ResponseDesc *resp) noexcept override {
        for (auto& input : config.inConfs) {
            for (auto& offset : input.desc.getBlockingDesc().getOffsetPaddingToData()) {
                if (offset) {
                    return GENERAL_ERROR;
                }
            }
        }
        for (auto& output : config.outConfs) {
            for (auto& offset : output.desc.getBlockingDesc().getOffsetPaddingToData()) {
                if (offset) {
                    return GENERAL_ERROR;
                }
            }
        }

        //  Check for holes in tensors
        SizeVector dictionary_dims = config.inConfs[GATHER_DICTIONARY].desc.getDims();
        SizeVector indexes_dims = config.inConfs[GATHER_INDEXES].desc.getDims();
        SizeVector out_dims = config.outConfs[0].desc.getDims();
        size_t idx_size = 1;
        for (auto dims : indexes_dims)
            idx_size *= dims;

        size_t dct_size = 1;
        for (auto dims : dictionary_dims)
            dct_size *= dims;

        size_t out_size = 1;
        for (auto dims : out_dims)
            out_size *= dims;

        size_t dctSV = config.inConfs[GATHER_DICTIONARY].desc.getBlockingDesc().getStrides()[0];
        size_t dctDV = config.inConfs[GATHER_DICTIONARY].desc.getBlockingDesc().getBlockDims()[0];
        size_t idxSV = config.inConfs[GATHER_INDEXES].desc.getBlockingDesc().getStrides()[0];
        size_t idxDV = config.inConfs[GATHER_INDEXES].desc.getBlockingDesc().getBlockDims()[0];
        size_t outSV = config.outConfs[0].desc.getBlockingDesc().getStrides()[0];
        size_t outDV = config.outConfs[0].desc.getBlockingDesc().getBlockDims()[0];
        if (outSV * outDV == out_size && idxSV * idxDV == idx_size && dctSV * dctDV == dct_size)
            withHoles = NONE;
        else if (outSV * outDV != out_size && idxSV * idxDV == idx_size && dctSV * dctDV == dct_size)
            withHoles = OUTPUT;

        return OK;
    };

    StatusCode getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc *resp) noexcept override {
        if (!errorMsg.empty()) {
            if (resp) {
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
        conf = confs;
        return OK;
    };

    explicit GatherImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 2 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            Precision inIdxPrecision = layer->insData[GATHER_INDEXES].lock()->getTensorDesc().getPrecision();
            if (inIdxPrecision != Precision::FP32 &&
                inIdxPrecision != Precision::I32 &&
                inIdxPrecision != Precision::U16 &&
                inIdxPrecision != Precision::I16 &&
                inIdxPrecision != Precision::U8 &&
                inIdxPrecision != Precision::I8)
                THROW_IE_EXCEPTION << "Incorrect input precision. Only FP32|I32|U16|I16|U8|I8 are supported!";

            //  Remove redundant dimensions
            const SizeVector& dictionary_dims = layer->insData[GATHER_DICTIONARY].lock()->getTensorDesc().getDims();
            size_t actualAxis = 0;
            SizeVector dims_actual;
            for (size_t i = 0; i < dictionary_dims.size(); i++) {
                if (dictionary_dims[i] > 1) {
                    for (size_t j = i; j < dictionary_dims.size(); j++)
                        dims_actual.push_back(dictionary_dims[j]);
                    break;
                }
            }

            if (dims_actual.size() == 0)
                THROW_IE_EXCEPTION << "Incorrect input parameters dimension!";

            axis = static_cast<int>(layer->GetParamAsInt("axis"));
            // Dictionary must be at least rank axis + 1
            if (axis > 0 && (dims_actual.size() - axis) < 1)
                THROW_IE_EXCEPTION << "Incorrect input parameters dimensions and axis number!";
            else if (axis < 0 && (static_cast<int>(dims_actual.size()) + axis) < 0)
                THROW_IE_EXCEPTION << "Incorrect input parameters dimensions and axis number!";

            if (axis < 0)
                axis += dims_actual.size();

            //  Find number of dictionaries, index range and data length
            for (size_t i = 0; i < axis; i++)
                numDictionaries *= dims_actual[i];
            indexRange = dims_actual[axis];
            for (size_t i = axis + 1; i < dims_actual.size(); i++)
                dataLength *= dims_actual[i];

            if (dataLength == 0)
                THROW_IE_EXCEPTION << "Incorrect input parameters dimension!";

            LayerConfig config;
            DataConfig dataConfigIdx, dataConfigDct;
            const SizeVector& indexes_dims = layer->insData[GATHER_INDEXES].lock()->getTensorDesc().getDims();
            dataConfigDct.desc = TensorDesc(InferenceEngine::Precision(InferenceEngine::Precision::FP32), dictionary_dims, InferenceEngine::Layout::ANY);
            dataConfigIdx.desc = TensorDesc(inIdxPrecision, indexes_dims, InferenceEngine::Layout::ANY);
            if (GATHER_DICTIONARY == 0) {
                config.inConfs.push_back(dataConfigDct);
                config.inConfs.push_back(dataConfigIdx);
            } else {
                config.inConfs.push_back(dataConfigIdx);
                config.inConfs.push_back(dataConfigDct);
            }

            DataConfig dataConfigOut;
            const SizeVector& out_dims = layer->outData[0]->getTensorDesc().getDims();
            SizeVector blocks = out_dims;
            SizeVector order(blocks.size());
            SizeVector dimOffsets(blocks.size());
            SizeVector strides(blocks.size());
            size_t offset(std::numeric_limits<size_t>::max());
            for (size_t i = 0; i < order.size(); i++) {
                strides[i] = std::numeric_limits<size_t>::max();
                dimOffsets[i] = 0;
                order[i] = i;
            }
            dataConfigOut.desc = TensorDesc(InferenceEngine::Precision(InferenceEngine::Precision::FP32), out_dims,
                                                                      { blocks, order, offset, dimOffsets, strides });
            config.outConfs.push_back(dataConfigOut);
            config.dynBatchSupport = false;
            confs.push_back(config);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        switch (inputs[GATHER_INDEXES]->precision()) {
            case Precision::FP32:
                gather(inputs[GATHER_INDEXES]->cbuffer().as<const float *>(), inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0], withHoles);
                break;
            case Precision::I32:
                gather(inputs[GATHER_INDEXES]->cbuffer().as<const int32_t *>(), inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0], withHoles);
                break;
            case Precision::U16:
                gather(inputs[GATHER_INDEXES]->cbuffer().as<const uint16_t *>(), inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0], withHoles);
                break;
            case Precision::I16:
                gather(inputs[GATHER_INDEXES]->cbuffer().as<const int16_t *>(), inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0], withHoles);
                break;
            case Precision::U8:
                gather(inputs[GATHER_INDEXES]->cbuffer().as<const uint8_t *>(), inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0], withHoles);
                break;
            case Precision::I8:
                gather(inputs[GATHER_INDEXES]->cbuffer().as<const int8_t *>(), inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0], withHoles);
                break;
            default:
                return GENERAL_ERROR;
        }

        return OK;
    }

protected:
    enum class ConfLayout { ANY, PLN, BLK8, BLK16 };
    std::string errorMsg;
    std::vector<LayerConfig> confs;

private:
    enum HolesMode {
        NONE = 0,
        OUTPUT = 1,
        ALL = 2
    };

    template <typename data_t>
    void gather(data_t *src_dataIdx, Blob::Ptr indexes, Blob::Ptr dictionary, Blob::Ptr output, bool withHoles);

    int axis = 0;
    size_t numDictionaries = 1;
    size_t indexRange = 0;
    size_t dataLength = 1;
    const size_t GATHER_DICTIONARY = 0;
    const size_t GATHER_INDEXES = 1;
    HolesMode withHoles = ALL;
};

template <typename data_t>
void GatherImpl::gather(data_t *src_dataIdx, Blob::Ptr indexes, Blob::Ptr dictionary, Blob::Ptr output, bool withHoles) {
    size_t src_dataIdxSize = indexes->size();
    size_t dataSize = sizeof(float) * dataLength;

    if (withHoles == GatherImpl::NONE) {  //  No holes in tensors
        const float *src_dataDict = dictionary->cbuffer().as<const float *>() + dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* dst_data = output->cbuffer().as<float *>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();
        src_dataIdx += indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();

        if (axis == 0) {
            parallel_for(src_dataIdxSize, [&](size_t i) {
                int idx = static_cast<int>(src_dataIdx[i]);

                //  Index clipping
                clipping(&idx, 0, indexRange);

                //  Copying data to destination from Dictionary
                simple_copy(&dst_data[dataLength * i],
                            output->byteSize() - (dataLength * i),
                            &src_dataDict[dataLength * idx],
                            dataSize);
            });
        } else {
            parallel_for(src_dataIdxSize, [&](size_t i) {
                int idx = static_cast<int>(src_dataIdx[i]);

                //  Index clipping
                clipping(&idx, 0, indexRange);

                //  Copying data to destination from Dictionary
                for (size_t j = 0; j < numDictionaries; j++) {
                    simple_copy(&dst_data[dataLength * (i + j * src_dataIdxSize)],
                                output->byteSize() - (dataLength * (i + j * src_dataIdxSize)),
                                &src_dataDict[dataLength * (idx + j * indexRange)],
                                dataSize);
                }
            });
        }
    } else if (withHoles == GatherImpl::OUTPUT) {  //  If only output tensor have holes
        const float *src_dataDict = dictionary->cbuffer().as<const float *>() + dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* dst_data = output->cbuffer().as<float *>();
        src_dataIdx += indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();

        parallel_for(src_dataIdxSize, [&](size_t i) {
            int idx = static_cast<int>(src_dataIdx[i]);

            //  Index clipping
            clipping(&idx, 0, indexRange);

            //  Copying data to destination from Dictionary
            for (size_t j = 0; j < numDictionaries; j++) {
                for (size_t k = 0; k < dataLength; k++) {
                    dst_data[output->getTensorDesc().offset(k + dataLength * (i + j * src_dataIdxSize))] =
                        src_dataDict[k + dataLength * (idx + j * indexRange)];
                }
            }
        });
    } else {  //  If input and oupput tensors have holes
        const float *src_dataDict = dictionary->cbuffer().as<const float *>();
        float* dst_data = output->cbuffer().as<float *>();

        parallel_for(src_dataIdxSize, [&](size_t i) {
            int idx = static_cast<int>(src_dataIdx[indexes->getTensorDesc().offset(i)]);

            //  Index clipping
            clipping(&idx, 0, indexRange);

            //  Copying data to destination from Dictionary
            for (size_t j = 0; j < numDictionaries; j++) {
                for (size_t k = 0; k < dataLength; k++) {
                    dst_data[output->getTensorDesc().offset(k + dataLength * (i + j * src_dataIdxSize))] =
                        src_dataDict[dictionary->getTensorDesc().offset(k + dataLength * (idx + j * indexRange))];
                }
            }
        });
    }
}

REG_FACTORY_FOR(ImplFactory<GatherImpl>, Gather);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
