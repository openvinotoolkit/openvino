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
#include "common/simple_copy.h"
#include "common/fp16_utils.h"

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
            if (inIdxPrecision != Precision::FP32 && inIdxPrecision != Precision::I32 && inIdxPrecision != Precision::FP16)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision. Only FP32, FP16 or I32 are supported!";

            Precision inDataPrecision = layer->insData[GATHER_DICTIONARY].lock()->getTensorDesc().getPrecision();
            if (inDataPrecision != Precision::FP32 && inDataPrecision != Precision::FP16)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision. Only FP32 or FP16 are supported!";

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

    struct f32toUi32 {
        inline unsigned int operator()(const float value) {
            return static_cast<unsigned int>(value);
        }
    };

    struct f16toUi32 {
        inline unsigned int operator()(const ie_fp16 value) {
            return static_cast<unsigned int>(f16tof32(value));
        }
    };

    struct i32toUi32 {
        inline unsigned int operator()(const int32_t value) {
            return static_cast<unsigned int>(value);
        }
    };

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[GATHER_INDEXES]->getTensorDesc().getPrecision()) {
            case Precision::FP32:
                gather<float, float, f32toUi32>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                break;
            case Precision::FP16:
                gather<ie_fp16, ie_fp16, f16toUi32>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                break;
            case Precision::I32:
                gather<int32_t, float, i32toUi32>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                break;
            default:
                return GENERAL_ERROR;
        }

        return OK;
    }

private:
    template <typename index_t, typename data_t, class Conversion>
    void gather(Blob::Ptr indexes, Blob::Ptr dictionary, Blob::Ptr output) {
        size_t src_indexSize = indexes->size();
        const index_t *src_index = indexes->cbuffer().as<const index_t *>() + indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const data_t *src_dataDict = dictionary->cbuffer().as<const data_t *>() + dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
        data_t *dst_data = output->cbuffer().as<data_t*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();

        if (axis == 0) {
            parallel_for(src_indexSize, [&](size_t i) {
                unsigned int idx = Conversion()(src_index[i]);

                //  Index clipping
                if (idx < indexRange) {
                    //  Copying data to destination from Dictionary
                    simple_copy(&dst_data[i * dataLength],
                                output->byteSize() - (dataLength * i),
                                &src_dataDict[dataLength * idx],
                                sizeof(data_t) * dataLength);
                } else {
                    memset(&dst_data[i * dataLength], 0, sizeof(data_t) * dataLength);
                }
            });
        } else {
            parallel_for(src_indexSize, [&](size_t i) {
                unsigned int idx = Conversion()(src_index[i]);

                //  Index clipping
                if (idx < indexRange) {
                    //  Copying data to destination from Dictionary
                    for (size_t j = 0; j < numDictionaries; j++) {
                        simple_copy(&dst_data[dataLength * (i + j * src_indexSize)],
                                    output->byteSize() - (dataLength * (i + j * src_indexSize)),
                                    &src_dataDict[dataLength * (idx + j * indexRange)],
                                    sizeof(data_t) * dataLength);
                    }
                } else {
                    for (size_t j = 0; j < numDictionaries; j++) {
                        memset(&dst_data[dataLength * (i + j * src_indexSize)], 0, sizeof(data_t) * dataLength);
                    }
                }
            });
        }
    }

    int axis = 0;
    size_t numDictionaries = 1;
    size_t indexRange = 0;
    size_t dataLength = 1;
    const size_t GATHER_DICTIONARY = 0;
    const size_t GATHER_INDEXES = 1;
};


REG_FACTORY_FOR(ImplFactory<GatherImpl>, Gather);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
