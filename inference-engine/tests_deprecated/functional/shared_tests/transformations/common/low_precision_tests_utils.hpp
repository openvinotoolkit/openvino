// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <cpp/ie_cnn_network.h>
#include <legacy/cnn_network_impl.hpp>

void fillDataWithInitValue(InferenceEngine::Blob::Ptr& blob, float initValue);

void fillDataWithInitValue(float *data, size_t size, float initValue = 0.0);

void fillDataWithInitValue(std::vector<float>& data, float initValue = 0.0);

void fillDataWithInitValue(InferenceEngine::CNNLayerPtr layer, const std::string& blobName = "", float initValue = 0.0);

void fillData(InferenceEngine::CNNLayerPtr layer, float value, const std::string& blobName = "");
void fillData(InferenceEngine::CNNLayerPtr layer, const std::vector<float>& values, const std::string& blobName = "");

inline void fillData(float *dst, size_t size, float value);
inline void fillData(float *dst, size_t size, const float* src);
inline void fillData(float *dst, size_t size, const std::vector<float>& src);

void fillData(InferenceEngine::Blob::Ptr& blob, float value);
void fillData(InferenceEngine::Blob::Ptr& blob, const float* src);
void fillData(InferenceEngine::Blob::Ptr& blob, const std::vector<float>& values);

InferenceEngine::CNNLayerPtr getLayer(const InferenceEngine::CNNNetwork& network, const std::string& layerName);

InferenceEngine::Blob::Ptr getBlob(InferenceEngine::CNNLayerPtr layer, const std::string& blobName);
