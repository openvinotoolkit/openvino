// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include <limits>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
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
                inIdxPrecision = Precision::I32;

            axis = layer->GetParamAsInt("axis");

            const SizeVector& dictionary_dims = layer->insData[GATHER_DICTIONARY].lock()->getTensorDesc().getDims();
            if (dictionary_dims.size() == 0)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input parameters dimension!";
            // Dictionary must be at least rank axis + 1
            IE_ASSERT(-static_cast<int>(dictionary_dims.size()) <= axis && axis < static_cast<int>(dictionary_dims.size()))
                << layer->name << " Incorrect input parameters dimensions and axis number!";
            if (axis < 0)
                axis += dictionary_dims.size();

            //  Find number of dictionaries, index range and data length
            for (int i = 0; i < axis; i++)
                numDictionaries *= dictionary_dims[i];
            indexRange = dictionary_dims[axis];
            for (size_t i = axis + 1; i < dictionary_dims.size(); i++)
                dataLength *= dictionary_dims[i];

            if (dataLength == 0)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input parameters dimension!";

            LayerConfig config;
            DataConfig dataConfigIdx, dataConfigDct;
            Precision dataPrecision = layer->insData[GATHER_DICTIONARY].lock()->getTensorDesc().getPrecision();
            dataConfigDct.desc = TensorDesc(dataPrecision, dictionary_dims,
                    layer->insData[GATHER_DICTIONARY].lock()->getTensorDesc().getLayoutByDims(dictionary_dims));
            config.inConfs.push_back(dataConfigDct);
            const SizeVector& indexes_dims = layer->insData[GATHER_INDEXES].lock()->getTensorDesc().getDims();
            dataConfigIdx.desc = TensorDesc(inIdxPrecision, indexes_dims,
                    layer->insData[GATHER_INDEXES].lock()->getTensorDesc().getLayout());
            config.inConfs.push_back(dataConfigIdx);

            DataConfig dataConfigOut;
            const SizeVector& out_dims = layer->outData[0]->getTensorDesc().getDims();
            dataConfigOut.desc = TensorDesc(dataPrecision, out_dims,
                    layer->outData[0]->getTensorDesc().getLayoutByDims(out_dims));
            config.outConfs.push_back(dataConfigOut);
            config.dynBatchSupport = false;
            confs.push_back(config);
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
                gather<float, f32toUi32>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                break;
            case Precision::FP16:
                gather<ie_fp16, f16toUi32>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                break;
            case Precision::I32:
                gather<int32_t, i32toUi32>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                break;
            default:
                return GENERAL_ERROR;
        }

        return OK;
    }

private:
    template <typename index_t, class Conversion>
    void gather(Blob::Ptr indexes, Blob::Ptr dictionary, Blob::Ptr output) {
        size_t src_indexSize = indexes->size();
        const index_t *src_index = indexes->cbuffer().as<const index_t *>() + indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const uint8_t *src_dataDict = dictionary->cbuffer().as<const uint8_t *>() + dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
        uint8_t *dst_data = output->cbuffer().as<uint8_t*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();
        size_t len = dataLength * dictionary->getTensorDesc().getPrecision().size();

        parallel_for(src_indexSize, [&](size_t i) {
            unsigned int idx = Conversion()(src_index[i]);

            //  Index clipping
            if (idx < indexRange) {
                //  Copying data to destination from Dictionary
                for (size_t j = 0; j < numDictionaries; j++) {
                    cpu_memcpy_s(&dst_data[len * (i + j * src_indexSize)],
                                output->byteSize() - (len * (i + j * src_indexSize)),
                                &src_dataDict[len * (idx + j * indexRange)],
                                len);
                }
            } else {
                for (size_t j = 0; j < numDictionaries; j++) {
                    memset(&dst_data[len * (i + j * src_indexSize)], 0, len);
                }
            }
        });
    }

    int axis = 0;
    size_t numDictionaries = 1;
    size_t indexRange = 0;
    size_t dataLength = 1;
    const size_t GATHER_DICTIONARY = 0;
    const size_t GATHER_INDEXES = 1;
};


REG_FACTORY_FOR(GatherImpl, Gather);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
