// Copyright (C) 2018-2019 Intel Corporation
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

class GatherImpl: public ExtLayerBase {
public:
    explicit GatherImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 2 || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            Precision inIdxPrecision = layer->insData[GATHER_INDEXES].lock()->getTensorDesc().getPrecision();
            if (inIdxPrecision != Precision::FP32 && inIdxPrecision != Precision::I32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision. Only FP32 or I32 are supported!";

            //  Remove redundant dimensions
            const SizeVector& dictionary_dims = layer->insData[GATHER_DICTIONARY].lock()->getTensorDesc().getDims();
            SizeVector dims_actual;
            for (size_t i = 0; i < dictionary_dims.size(); i++) {
                if (dictionary_dims[i] > 1) {
                    for (size_t j = i; j < dictionary_dims.size(); j++)
                        dims_actual.push_back(dictionary_dims[j]);
                    break;
                }
            }

            if (dims_actual.size() == 0)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input parameters dimension!";

            axis = static_cast<int>(layer->GetParamAsInt("axis"));
            // Dictionary must be at least rank axis + 1
            if (axis > 0 && static_cast<int>(dims_actual.size()) < (1 + axis))
                THROW_IE_EXCEPTION << layer->name << " Incorrect input parameters dimensions and axis number!";
            else if (axis < 0 && (static_cast<int>(dims_actual.size()) + axis) < 0)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input parameters dimensions and axis number!";

            if (axis < 0)
                axis += dims_actual.size();

            //  Find number of dictionaries, index range and data length
            for (int i = 0; i < axis; i++)
                numDictionaries *= dims_actual[i];
            indexRange = dims_actual[axis];
            for (size_t i = axis + 1; i < dims_actual.size(); i++)
                dataLength *= dims_actual[i];

            if (dataLength == 0)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input parameters dimension!";

            addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                      { DataConfigurator(ConfLayout::PLN) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[GATHER_INDEXES]->precision()) {
            case Precision::FP32:
                gather(inputs[GATHER_INDEXES]->cbuffer().as<const float *>(), inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                break;
            case Precision::I32:
                gather(inputs[GATHER_INDEXES]->cbuffer().as<const int32_t *>(), inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                break;
            default:
                return GENERAL_ERROR;
        }

        return OK;
    }

private:
    template <typename data_t>
    void gather(data_t *src_dataIdx, Blob::Ptr indexes, Blob::Ptr dictionary, Blob::Ptr output);

    int axis = 0;
    size_t numDictionaries = 1;
    size_t indexRange = 0;
    size_t dataLength = 1;
    const size_t GATHER_DICTIONARY = 0;
    const size_t GATHER_INDEXES = 1;
};

template <typename data_t>
void GatherImpl::gather(data_t *src_dataIdx, Blob::Ptr indexes, Blob::Ptr dictionary, Blob::Ptr output) {
    size_t src_dataIdxSize = indexes->size();
    const float *src_dataDict = dictionary->cbuffer().as<const float *>() + dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
    float* dst_data = output->cbuffer().as<float *>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();
    src_dataIdx += indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();

    if (axis == 0) {
        parallel_for(src_dataIdxSize, [&](size_t i) {
            unsigned int idx = static_cast<unsigned int>(src_dataIdx[i]);

            //  Index clipping
            if (idx < indexRange) {
                //  Copying data to destination from Dictionary
                simple_copy(&dst_data[i * dataLength],
                            output->byteSize() - (dataLength * i),
                            &src_dataDict[dataLength * idx],
                            sizeof(float) * dataLength);
            } else {
                std::fill_n(&dst_data[i * dataLength], dataLength, 0.f);
            }
        });
    } else {
        parallel_for(src_dataIdxSize, [&](size_t i) {
            unsigned int idx = static_cast<unsigned int>(src_dataIdx[i]);

            //  Index clipping
            if (idx < indexRange) {
                //  Copying data to destination from Dictionary
                for (size_t j = 0; j < numDictionaries; j++) {
                    simple_copy(&dst_data[dataLength * (i + j * src_dataIdxSize)],
                                output->byteSize() - (dataLength * (i + j * src_dataIdxSize)),
                                &src_dataDict[dataLength * (idx + j * indexRange)],
                                sizeof(float) * dataLength);
                }
            } else {
                for (size_t j = 0; j < numDictionaries; j++) {
                    std::fill_n(&dst_data[dataLength * (i + j * src_dataIdxSize)], dataLength, 0.f);
                }
            }
        });
    }
}

REG_FACTORY_FOR(ImplFactory<GatherImpl>, Gather);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
