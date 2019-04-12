// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <map>
#include <memory>
#include <cmath>
#include <string>
#include <vector>
#include <ie_layers.h>
#include <ie_algorithm.hpp>
#include "ie_const_infer_impl.hpp"
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

struct GatherParams {
    size_t dataLength = 1;
    int axis = 0;
    size_t indexRange = 0;
    size_t numDictionaries = 1;
};

template<typename data_t>
void
gather(data_t* src_dataIdx, const Blob::CPtr& indexes, const Blob::CPtr& dictionary, const Blob::Ptr& output,
       const GatherParams& p) {
    size_t src_dataIdxSize = indexes->size();
    size_t dataSize = sizeof(float) * p.dataLength;

    const float* src_dataDict =
            dictionary->cbuffer().as<const float*>() + dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
    float* dst_data = output->cbuffer().as<float*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();
    src_dataIdx += indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();

    if (p.axis == 0) {
        parallel_for(src_dataIdxSize, [&](size_t i) {
            int idx = static_cast<int>(src_dataIdx[i]);

            //  Index clipping
            details::clipping(&idx, 0, p.indexRange);

            //  Copying data to destination from Dictionary
            ie_memcpy(&dst_data[p.dataLength * i],
                      output->byteSize() - (p.dataLength * i),
                      &src_dataDict[p.dataLength * idx],
                      dataSize);
        });
    } else {
        parallel_for(src_dataIdxSize, [&](size_t i) {
            int idx = static_cast<int>(src_dataIdx[i]);

            //  Index clipping
            details::clipping(&idx, 0, p.indexRange);

            //  Copying data to destination from Dictionary
            for (size_t j = 0; j < p.numDictionaries; j++) {
                ie_memcpy(&dst_data[p.dataLength * (i + j * src_dataIdxSize)],
                          output->byteSize() - (p.dataLength * (i + j * src_dataIdxSize)),
                          &src_dataDict[p.dataLength * (idx + j * p.indexRange)],
                          dataSize);
            }
        });
    }
}

/**
 *@brief Implementation of Const inference for Gather layer
 */
class GatherConstInfer : public ConstInferImpl {
public:
    explicit GatherConstInfer(const std::string& type) : ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        LayerParams lp{};
        CNNLayer layer(lp);
        layer.params = params;


        const size_t GATHER_DICTIONARY = 0;
        const size_t GATHER_INDEXES = 1;

        if (inData.size() != 2 || outData.empty())
            THROW_IE_EXCEPTION << " Incorrect number of input/output edges!";

        Precision inIdxPrecision = inData[GATHER_INDEXES]->getTensorDesc().getPrecision();
        if (inIdxPrecision != Precision::FP32 &&
            inIdxPrecision != Precision::I32 &&
            inIdxPrecision != Precision::U16 &&
            inIdxPrecision != Precision::I16 &&
            inIdxPrecision != Precision::U8 &&
            inIdxPrecision != Precision::I8)
            THROW_IE_EXCEPTION << " Incorrect input precision. Only FP32|I32|U16|I16|U8|I8 are supported!";

        //  Remove redundant dimensions
        const SizeVector& dictionary_dims = inData[GATHER_DICTIONARY]->getTensorDesc().getDims();
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
            THROW_IE_EXCEPTION << " Incorrect input parameters dimension!";

        GatherParams p;
        p.axis = static_cast<int>(layer.GetParamAsInt("axis"));
        // Dictionary must be at least rank axis + 1
        if (p.axis > 0 && dims_actual.size() < (1 + p.axis))
            THROW_IE_EXCEPTION << " Incorrect input parameters dimensions and axis number!";
        else if (p.axis < 0 && (static_cast<int>(dims_actual.size()) + p.axis) < 0)
            THROW_IE_EXCEPTION << " Incorrect input parameters dimensions and axis number!";

        if (p.axis < 0)
            p.axis += dims_actual.size();

        //  Find number of dictionaries, index range and data length
        for (size_t i = 0; i < p.axis; i++)
            p.numDictionaries *= dims_actual[i];
        p.indexRange = dims_actual[p.axis];
        for (size_t i = p.axis + 1; i < dims_actual.size(); i++)
            p.dataLength *= dims_actual[i];

        if (p.dataLength == 0)
            THROW_IE_EXCEPTION << " Incorrect input parameters dimension!";


        switch (inData[GATHER_INDEXES]->precision()) {
            case Precision::FP32:
                gather(inData[GATHER_INDEXES]->cbuffer().as<const float*>(), inData[GATHER_INDEXES],
                       inData[GATHER_DICTIONARY], outData[0], p);
                break;
            case Precision::I32:
                gather(inData[GATHER_INDEXES]->cbuffer().as<const int32_t*>(), inData[GATHER_INDEXES],
                       inData[GATHER_DICTIONARY], outData[0], p);
                break;
            case Precision::U16:
                gather(inData[GATHER_INDEXES]->cbuffer().as<const uint16_t*>(), inData[GATHER_INDEXES],
                       inData[GATHER_DICTIONARY], outData[0], p);
                break;
            case Precision::I16:
                gather(inData[GATHER_INDEXES]->cbuffer().as<const int16_t*>(), inData[GATHER_INDEXES],
                       inData[GATHER_DICTIONARY], outData[0], p);
                break;
            case Precision::U8:
                gather(inData[GATHER_INDEXES]->cbuffer().as<const uint8_t*>(), inData[GATHER_INDEXES],
                       inData[GATHER_DICTIONARY], outData[0], p);
                break;
            case Precision::I8:
                gather(inData[GATHER_INDEXES]->cbuffer().as<const int8_t*>(), inData[GATHER_INDEXES],
                       inData[GATHER_DICTIONARY], outData[0], p);
                break;
            default:
                THROW_IE_EXCEPTION << " Unsupported precision!";
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
