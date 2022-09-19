// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_converter.hpp"

namespace ov {
namespace intel_gna {
namespace frontend {

static void fp16_to_fp32(InferenceEngine::WeightableLayer& wl) {
    InferenceEngine::BlobMap newBlobs;
    for (auto& blob : wl.blobs) {
        if (blob.second->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP16) {
            THROW_GNA_EXCEPTION << "Unsupported precision. Layer: " << wl.name << " , Blob: " << blob.first;
        }
        auto fp32_blob = make_fp32_blob(blob.second);
        newBlobs[blob.first] = fp32_blob;
    }
    wl._biases = newBlobs["biases"];
    wl._weights = newBlobs["weights"];
    wl.blobs = newBlobs;
    wl.precision = InferenceEngine::Precision::FP32;
    for (auto& dataItem : wl.outData) {
        dataItem->setPrecision(InferenceEngine::Precision::FP32);
    }
}

InferenceEngine::Blob::Ptr make_fp32_blob(InferenceEngine::Blob::Ptr fp16_blob) {
    auto fp32_blob = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32,
                                                               fp16_blob->getTensorDesc().getDims(),
                                                               fp16_blob->getTensorDesc().getLayout()});
    fp32_blob->allocate();

    int i = 0;
    for (auto& f32Value : *fp32_blob) {
        auto f16Value =
            fp16_blob->buffer()
                .template as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type*>()[i++];
        f32Value = InferenceEngine::PrecisionUtils::f16tof32(f16Value);
    }

    return static_cast<InferenceEngine::Blob::Ptr>(fp32_blob);
}

void convertBlobs(InferenceEngine::CNNLayer& layer) {
    auto layer_info = GNAPluginNS::LayerInfo(layer);

    if (layer_info.isWeightable()) {
        InferenceEngine::WeightableLayer& wl = dynamic_cast<InferenceEngine::WeightableLayer&>(layer);
        if (wl.precision == InferenceEngine::Precision::FP16) {
            fp16_to_fp32(wl);
        }
    } else {
        layer.precision = InferenceEngine::Precision::FP32;
        for (auto& dataItem : layer.outData) {
            dataItem->setPrecision(InferenceEngine::Precision::FP32);
        }
        for (auto& blob_pair : layer.blobs) {
            auto& blob_ptr = blob_pair.second;
            if (blob_ptr->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP16) {
                blob_ptr = make_fp32_blob(blob_ptr);
            }
        }
    }
}

}  // namespace frontend
}  // namespace intel_gna
}  // namespace ov
