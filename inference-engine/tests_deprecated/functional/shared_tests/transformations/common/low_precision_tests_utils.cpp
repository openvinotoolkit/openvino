// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_tests_utils.hpp"

#include <details/ie_cnn_network_tools.h>
#include <details/caseless.hpp>
#include <precision_utils.h>
#include <cmath>

using InferenceEngine::CNNLayerPtr;
using InferenceEngine::Blob;
using InferenceEngine::details::CNNNetworkImpl;
using InferenceEngine::CNNNetwork;
using InferenceEngine::DataPtr;
using InferenceEngine::Precision;

// TODO: FP32 detected
void fillDataWithInitValue(float *data, size_t size, float initValue) {
    for (size_t i = 0lu; i < size; i++) {
        data[i] = sin((i + initValue + 1.0f) * 0.03f);
    }
}

void fillDataWithInitValue(std::vector<float>& data, float initValue) {
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = sin((i + initValue + 1.0) * 0.03);
    }
}

void fillDataWithInitValue(Blob::Ptr& blob, float initValue) {
    if (blob == nullptr) {
        THROW_IE_EXCEPTION << "Blob is nullable";
    }

    const Precision& precision = blob->getTensorDesc().getPrecision();
    const size_t dataSize = blob->size();
    if (precision == Precision::FP32) {
        float* buffer = blob->buffer().as<float*>();
        for (size_t i = 0lu; i < dataSize; i++) {
            buffer[i] = sin((float(i) + initValue + 1.f) * 0.03f);
        }
    } else if (precision == Precision::FP16) {
        short* buffer = blob->buffer().as<short*>();
        for (size_t i = 0lu; i < dataSize; i++) {
            buffer[i] = InferenceEngine::PrecisionUtils::f32tof16(sin((float(i) + initValue + 1.f) * 0.03f));
        }
    }
}

void fillDataWithInitValue(CNNLayerPtr layer, const std::string& blobName, float initValue) {
    if (layer == nullptr) {
        THROW_IE_EXCEPTION << "layer is nullable";
    }
    if (blobName.empty() && (layer->blobs.size() != 1)) {
        THROW_IE_EXCEPTION << "several blobs";
    }

    Blob::Ptr blob = blobName.empty() ? layer->blobs.begin()->second : layer->blobs[blobName];
    if (blob == nullptr)
        THROW_IE_EXCEPTION << "Layer '" << layer->name << "' does not have blob '" << blobName << "'";
    fillDataWithInitValue(blob, initValue);
}

void fillData(float *dst, size_t size, float value) {
    std::fill(dst, dst + size, value);
}

void fillData(float* dst, size_t size, const float* src) {
    std::copy(src, src + size, dst);
}

void fillData(float *dst, size_t size, const std::vector<float>& src) {
    if (size != src.size()) {
        THROW_IE_EXCEPTION << "values size is not correct";
    }
    fillData(dst, size, src.data());
}

void fillData(Blob::Ptr& blob, float value) {
    if (blob == nullptr) {
        THROW_IE_EXCEPTION << "Blob is nullable";
    }

    const Precision& precision = blob->getTensorDesc().getPrecision();
    const size_t dataSize = blob->size();
    if (precision == Precision::FP32) {
        fillData(blob->buffer().as<float*>(), dataSize, value);
    } else if (precision == Precision::FP16) {
        short* buffer = blob->buffer().as<short*>();
        for (size_t i = 0lu; i < blob->size(); i++) {
            buffer[i] = InferenceEngine::PrecisionUtils::f32tof16(value);
        }
    }
}

void fillData(Blob::Ptr& blob, const float* src) {
    if (blob == nullptr) {
        THROW_IE_EXCEPTION << "Blob is nullable";
    }

    const Precision& precision = blob->getTensorDesc().getPrecision();
    const size_t dataSize = blob->size();
    if (precision == Precision::FP32) {
        fillData(blob->buffer().as<float*>(), dataSize, src);
    } else if (precision == Precision::FP16) {
        short* dstData = blob->buffer().as<short*>();
        InferenceEngine::PrecisionUtils::f32tof16Arrays(dstData, src, dataSize, 1.f, 0.f);
    } else {
        THROW_IE_EXCEPTION << "Unsupported precision: " << precision;
    }
}

void fillData(Blob::Ptr& blob, const std::vector<float>& src) {
    fillData(blob, src.data());
}

void fillData(CNNLayerPtr layer, float value, const std::string& blobName) {
    if (layer == nullptr) {
        THROW_IE_EXCEPTION << "layer is nullable";
    }
    if (blobName.empty() && (layer->blobs.size() != 1)) {
        THROW_IE_EXCEPTION << "several blobs";
    }

    Blob::Ptr blob = blobName.empty() ? layer->blobs.begin()->second : layer->blobs[blobName];
    fillData(blob, value);
}

void fillData(CNNLayerPtr layer, const std::vector<float>& values, const std::string& blobName) {
    if (layer == nullptr) {
        THROW_IE_EXCEPTION << "layer is nullable";
    }
    if (blobName.empty() && (layer->blobs.size() != 1)) {
        THROW_IE_EXCEPTION << "several blobs";
    }

    Blob::Ptr blob = blobName.empty() ? layer->blobs.begin()->second : layer->blobs[blobName];
    if (blob->size() != values.size()) {
        THROW_IE_EXCEPTION << "values size is not correct";
    }

    fillData(blob, values);
}

CNNLayerPtr getLayer(const CNNNetwork& network, const std::string& layerName) {
    std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
    for (CNNLayerPtr& layer : layers) {
        if (layer->name == layerName) {
            return layer;
        }
    }

    return nullptr;
}

Blob::Ptr getBlob(CNNLayerPtr layer, const std::string& blobName) {
    if (layer == nullptr) {
        THROW_IE_EXCEPTION << "layer is nullable";
    }
    if (blobName.empty() && (layer->blobs.size() != 1)) {
        THROW_IE_EXCEPTION << "several blobs";
    }
    Blob::Ptr blob = blobName.empty() ? layer->blobs.begin()->second : layer->blobs[blobName];
    return blob;
}
