// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <legacy/ie_layers.h>

#include <cmath>
#include <ie_algorithm.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_const_infer_impl.hpp"
#include "ie_parallel.hpp"
#include "precision_utils.h"

namespace InferenceEngine {
namespace ShapeInfer {

struct GatherParams {
    size_t dataLength = 1;
    int axis = 0;
    size_t indexRange = 0;
    size_t numDictionaries = 1;
};

/**
 *@brief Implementation of Const inference for Gather layer
 */
class GatherConstInfer : public ConstInferImpl {
public:
    explicit GatherConstInfer(const std::string& type): ConstInferImpl(type) {}

    struct f32toUi32 {
        inline unsigned int operator()(const float value) {
            return static_cast<unsigned int>(value);
        }
    };

    struct f16toUi32 {
        inline unsigned int operator()(const ie_fp16 value) {
            return static_cast<unsigned int>(PrecisionUtils::f16tof32(value));
        }
    };

    struct i32toUi32 {
        inline unsigned int operator()(const int32_t value) {
            return static_cast<unsigned int>(value);
        }
    };

    template <typename index_t, class Conversion>
    void gather(const Blob::CPtr& indexes, const Blob::CPtr& dictionary, Blob::Ptr output, const GatherParams& p) {
        size_t src_indexSize = indexes->size();
        const index_t* src_index =
            indexes->cbuffer().as<const index_t*>() + indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const uint8_t* src_dataDict = dictionary->cbuffer().as<const uint8_t*>() +
                                      dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
        uint8_t* dst_data =
            output->cbuffer().as<uint8_t*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();

        parallel_for(src_indexSize, [&](size_t i) {
            unsigned int idx = Conversion()(src_index[i]);

            //  Index clipping
            if (idx < p.indexRange) {
                //  Copying data to destination from Dictionary
                for (size_t j = 0; j < p.numDictionaries; j++) {
                    ie_memcpy(&dst_data[p.dataLength * (i + j * src_indexSize)],
                              output->byteSize() - (p.dataLength * (i + j * src_indexSize)),
                              &src_dataDict[p.dataLength * (idx + j * p.indexRange)], p.dataLength);
                }
            } else {
                for (size_t j = 0; j < p.numDictionaries; j++) {
                    memset(&dst_data[p.dataLength * (i + j * src_indexSize)], 0, p.dataLength);
                }
            }
        });
    }

    void inferImpl(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) override {
        LayerParams lp {};
        CNNLayer layer(lp);
        layer.params = params;

        const size_t GATHER_DICTIONARY = 0;
        const size_t GATHER_INDEXES = 1;

        if (inData.size() != 2 || outData.empty()) THROW_IE_EXCEPTION << " Incorrect number of input/output edges!";

        Precision inIdxPrecision = inData[GATHER_INDEXES]->getTensorDesc().getPrecision();
        if (inIdxPrecision != Precision::FP32 && inIdxPrecision != Precision::FP16 && inIdxPrecision != Precision::I32)
            THROW_IE_EXCEPTION << " Incorrect input precision. Only FP32|FP16|I32 are supported!";

        Precision inDataPrecision = inData[GATHER_DICTIONARY]->getTensorDesc().getPrecision();
        if (inDataPrecision != Precision::FP32 && inDataPrecision != Precision::FP16 &&
            inIdxPrecision != Precision::I32)
            THROW_IE_EXCEPTION << " Incorrect input precision. Only FP32|FP16|I32 are supported!";

        //  Remove redundant dimensions
        const SizeVector& dictionary_dims = inData[GATHER_DICTIONARY]->getTensorDesc().getDims();
        if (dictionary_dims.size() == 0) THROW_IE_EXCEPTION << " Incorrect input parameters dimension!";

        GatherParams p;
        p.axis = static_cast<int>(layer.GetParamAsInt("axis"));
        // Dictionary must be at least rank axis + 1
        if (!(-static_cast<int>(dictionary_dims.size()) <= p.axis && p.axis < static_cast<int>(dictionary_dims.size())))
            THROW_IE_EXCEPTION << " Incorrect input parameters dimensions and axis number!";

        if (p.axis < 0) p.axis += dictionary_dims.size();

        //  Find number of dictionaries, index range and data length
        for (size_t i = 0; i < p.axis; i++) p.numDictionaries *= dictionary_dims[i];
        p.indexRange = dictionary_dims[p.axis];
        for (size_t i = p.axis + 1; i < dictionary_dims.size(); i++) p.dataLength *= dictionary_dims[i];

        if (p.dataLength == 0) THROW_IE_EXCEPTION << " Incorrect input parameters dimension!";

        p.dataLength *= inData[GATHER_DICTIONARY]->getTensorDesc().getPrecision().size();

        switch (inData[GATHER_INDEXES]->getTensorDesc().getPrecision()) {
        case Precision::FP32:
            gather<float, f32toUi32>(inData[GATHER_INDEXES], inData[GATHER_DICTIONARY], outData[0], p);
            break;
        case Precision::FP16:
            gather<ie_fp16, f16toUi32>(inData[GATHER_INDEXES], inData[GATHER_DICTIONARY], outData[0], p);
            break;
        case Precision::I32:
            gather<int32_t, i32toUi32>(inData[GATHER_INDEXES], inData[GATHER_DICTIONARY], outData[0], p);
            break;
        default:
            THROW_IE_EXCEPTION << " Unsupported precision!";
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
