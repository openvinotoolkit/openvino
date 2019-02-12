// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides a set of convenience utility functions and the main include file for all other .h
 * files.
 * 
 * @file inference_engine.hpp
 */
#pragma once

#include <cpp/ie_cnn_net_reader.h>
#include <ie_api.h>
#include <ie_blob.h>
#include <ie_layers.h>

#include <algorithm>
#include <cpp/ie_executable_network.hpp>
#include <cpp/ie_plugin_cpp.hpp>
#include <ie_core.hpp>
#include <ie_error.hpp>
#include <ie_icnn_network.hpp>
#include <ie_icnn_network_stats.hpp>
#include <ie_plugin_config.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_version.hpp>
#include <memory>
#include <numeric>
#include <vector>

/**
 * @brief Inference Engine API
 */
namespace InferenceEngine {

/**
 * @deprecated InferenceEngine utility functions are not a part of public API
 * @brief Gets the top n results from a tblob
 *
 * @param n Top n count
 * @param input 1D tblob that contains probabilities
 * @param output Vector of indexes for the top n places
 */
template <class T>
INFERENCE_ENGINE_DEPRECATED(
    "InferenceEngine utility functions are not a part of public API. Will be removed in 2020 R2")
inline void TopResults(unsigned int n, TBlob<T>& input, std::vector<unsigned>& output) {
    SizeVector dims = input.getTensorDesc().getDims();
    size_t input_rank = dims.size();
    if (!input_rank || !dims[0]) THROW_IE_EXCEPTION << "Input blob has incorrect dimensions!";
    size_t batchSize = dims[0];
    std::vector<unsigned> indexes(input.size() / batchSize);

    n = static_cast<unsigned>(std::min<size_t>((size_t)n, input.size()));

    output.resize(n * batchSize);

    for (size_t i = 0; i < batchSize; i++) {
        size_t offset = i * (input.size() / batchSize);
        T* batchData = input.data();
        batchData += offset;

        std::iota(std::begin(indexes), std::end(indexes), 0);
        std::partial_sort(std::begin(indexes), std::begin(indexes) + n, std::end(indexes),
                          [&batchData](unsigned l, unsigned r) {
                              return batchData[l] > batchData[r];
                          });
        for (unsigned j = 0; j < n; j++) {
            output.at(i * n + j) = indexes.at(j);
        }
    }
}

/**
 * @deprecated InferenceEngine utility functions are not a part of public API
 * @brief Gets the top n results from a blob
 *
 * @param n Top n count
 * @param input 1D blob that contains probabilities
 * @param output Vector of indexes for the top n places
 */
void TopResults(unsigned int n, Blob& input, std::vector<unsigned>& output);
 
/**
 * @deprecated InferenceEngine utility functions are not a part of public API
 * @brief Copies a 8-bit RGB image to the blob.
 *
 * Throws an exception in case of dimensions or input size mismatch
 *
 * @tparam data_t Type of the target blob
 * @param RGB8 8-bit RGB image
 * @param RGB8_size Size of the image
 * @param blob Target blob to write image to
 */
template <typename data_t>
INFERENCE_ENGINE_DEPRECATED(
    "InferenceEngine utility functions are not a part of public API. Will be removed in 2020 R2")
void copyFromRGB8(uint8_t* RGB8, size_t RGB8_size, InferenceEngine::TBlob<data_t>* blob) {
    SizeVector dims = blob->getTensorDesc().getDims();
    if (4 != dims.size())
        THROW_IE_EXCEPTION << "Cannot write data to input blob! Blob has incorrect dimensions size " << dims.size();
    size_t num_channels = dims[1];  // because RGB
    size_t num_images = dims[0];
    size_t w = dims[3];
    size_t h = dims[2];
    size_t nPixels = w * h;

    if (RGB8_size != w * h * num_channels * num_images)
        THROW_IE_EXCEPTION << "input pixels mismatch, expecting " << w * h * num_channels * num_images
                           << " bytes, got: " << RGB8_size;

    std::vector<data_t*> dataArray;
    for (unsigned int n = 0; n < num_images; n++) {
        for (unsigned int i = 0; i < num_channels; i++) {
            if (!n && !i && dataArray.empty()) {
                dataArray.push_back(blob->data());
            } else {
                dataArray.push_back(dataArray.at(n * num_channels + i - 1) + nPixels);
            }
        }
    }
    for (size_t n = 0; n < num_images; n++) {
        size_t n_num_channels = n * num_channels;
        size_t n_num_channels_nPixels = n_num_channels * nPixels;
        for (size_t i = 0; i < nPixels; i++) {
            size_t i_num_channels = i * num_channels + n_num_channels_nPixels;
            for (size_t j = 0; j < num_channels; j++) {
                dataArray.at(n_num_channels + j)[i] = RGB8[i_num_channels + j];
            }
        }
    }
}

/**
 * @deprecated InferenceEngine utility functions are not a part of public API
 * @brief Splits the RGB channels to either I16 Blob or float blob.
 *
 * The image buffer is assumed to be packed with no support for strides.
 *
 * @param imgBufRGB8 Packed 24bit RGB image (3 bytes per pixel: R-G-B)
 * @param lengthbytesSize Size in bytes of the RGB image. It is equal to amount of pixels times 3 (number of channels)
 * @param input Blob to contain the split image (to 3 channels)
 */
INFERENCE_ENGINE_DEPRECATED(
    "InferenceEngine utility functions are not a part of public API. Will be removed in 2020 R2")
void ConvertImageToInput(unsigned char* imgBufRGB8, size_t lengthbytesSize, Blob& input);

/**
 * @deprecated InferenceEngine utility functions are not a part of public API
 * @brief Copies data from a certain precision to float
 *
 * @param dst Pointer to an output float buffer, must be allocated before the call
 * @param src Source blob to take data from
 */
template <typename T>
INFERENCE_ENGINE_DEPRECATED(
    "InferenceEngine utility functions are not a part of public API. Will be removed in 2020 R2")
void copyToFloat(float* dst, const InferenceEngine::Blob* src) {
    if (!dst) {
        return;
    }
    const InferenceEngine::TBlob<T>* t_blob = dynamic_cast<const InferenceEngine::TBlob<T>*>(src);
    if (t_blob == nullptr) {
#if defined(__ANDROID__)
	    std::cout << "in t_blob is nullptr" << std::endl;
	    // input type mismatch with actual input
#else
        THROW_IE_EXCEPTION << "input type is " << src->getTensorDesc().getPrecision() << " but input is not "
                           << typeid(T).name();
#endif
    }

    const T* srcPtr = t_blob->readOnly();
    if (srcPtr == nullptr) {
        THROW_IE_EXCEPTION << "Input data was not allocated.";
    }
    for (size_t i = 0; i < t_blob->size(); i++) dst[i] = srcPtr[i];
}

}  // namespace InferenceEngine
