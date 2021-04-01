// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "quantized_layer_params.hpp"
#include "precision_utils.h"

inline InferenceEngine::Blob::Ptr make_fp32_blob(InferenceEngine::Blob::Ptr fp16_blob) {
    auto fp32_blob = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
         fp16_blob->getTensorDesc().getDims(), fp16_blob->getTensorDesc().getLayout() });
    fp32_blob->allocate();

    int i = 0;
    for (auto& f32Value : *fp32_blob) {
        auto f16Value = fp16_blob->buffer().template as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type*>()[i++];
        f32Value = InferenceEngine::PrecisionUtils::f16tof32(f16Value);
    }

    return static_cast<InferenceEngine::Blob::Ptr>(fp32_blob);
}

inline void fp16_to_fp32(InferenceEngine::WeightableLayer *lp) {
    InferenceEngine::BlobMap newBlobs;
    for (auto& blob : lp->blobs) {
        if (blob.second->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP16) {
            THROW_GNA_EXCEPTION << "Unsupported precision. Layer: " << lp->name << " , Blob: " << blob.first;
        }
        auto fp32_blob = make_fp32_blob(blob.second);
        newBlobs[blob.first] = fp32_blob;
    }
    lp->_biases = newBlobs["biases"];
    lp->_weights = newBlobs["weights"];
    lp->blobs = newBlobs;
    lp->precision = InferenceEngine::Precision::FP32;
    for (auto& dataItem : lp->outData) {
        dataItem->setPrecision(InferenceEngine::Precision::FP32);
    }
}

template <class LayerToVisit>
inline bool convertWeights(LayerToVisit*) {
    return false;
}

template <>
inline bool convertWeights(InferenceEngine::CNNLayer* lp) {
    lp->precision = InferenceEngine::Precision::FP32;
    for (auto& dataItem : lp->outData) {
        dataItem->setPrecision(InferenceEngine::Precision::FP32);
    }
    for (auto& blob_pair : lp->blobs) {
        auto &blob_ptr = blob_pair.second;
        if (blob_ptr->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP16) {
            blob_ptr = make_fp32_blob(blob_ptr);
        }
    }

    return true;
}

template<>
inline bool convertWeights(InferenceEngine::WeightableLayer* lp) {
    if (lp->precision == InferenceEngine::Precision::FP16) {
        fp16_to_fp32(lp);
    }
    return true;
}

class WeightsConverter {
public:
    template <class LayerToVisit>
    bool  operator () (LayerToVisit layer) const {
        return convertWeights(layer);
    }
};