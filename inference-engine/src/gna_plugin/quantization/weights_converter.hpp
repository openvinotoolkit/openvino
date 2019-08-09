// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "quantized_layer_params.hpp"
#include "precision_utils.h"

inline void fp16_to_fp32(InferenceEngine::WeightableLayer *lp) {
    InferenceEngine::BlobMap newBlobs;
    for (auto& blob : lp->blobs) {
        if (blob.second->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP16) {
            THROW_GNA_EXCEPTION << "Unsupported precision. Layer: " << lp->name << " , Blob: " << blob.first;
        }
        auto tmp =
                InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                    blob.second->getTensorDesc().getDims(), InferenceEngine::Layout::C });
        tmp->allocate();
        int i = 0;
        for (auto& f32Value : *tmp) {
            auto f16Value = blob.second->buffer().template as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type*>()[i++];
            f32Value = InferenceEngine::PrecisionUtils::f16tof32(f16Value);
        }
        newBlobs[blob.first] = tmp;
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