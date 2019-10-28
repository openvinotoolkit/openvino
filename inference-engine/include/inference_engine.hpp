// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides a set of convenience utility functions and the main include file for all other .h files.
 * @file inference_engine.hpp
 */
#pragma once

#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>

#include <ie_blob.h>
#include <ie_api.h>
#include <ie_error.hpp>
#include <ie_layers.h>
#include <ie_device.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_config.hpp>
#include <ie_icnn_network.hpp>
#include <ie_icnn_network_stats.hpp>
#include <ie_core.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include <cpp/ie_plugin_cpp.hpp>
#include <cpp/ie_executable_network.hpp>
#include <ie_version.hpp>

/**
 * @brief Inference Engine API
 */
namespace InferenceEngine {
/**
 * @brief Gets the top n results from a tblob
 * @param n Top n count
 * @param input 1D tblob that contains probabilities
 * @param output Vector of indexes for the top n places
 */
template<class T>
inline void TopResults(unsigned int n, TBlob<T> &input, std::vector<unsigned> &output) {
    SizeVector dims = input.getTensorDesc().getDims();
    size_t input_rank = dims.size();
    if (!input_rank || !dims[0])
        THROW_IE_EXCEPTION << "Input blob has incorrect dimensions!";
    size_t batchSize = dims[0];
    std::vector<unsigned> indexes(input.size() / batchSize);

    n = static_cast<unsigned>(std::min<size_t>((size_t) n, input.size()));

    output.resize(n * batchSize);

    for (size_t i = 0; i < batchSize; i++) {
        size_t offset = i * (input.size() / batchSize);
        T *batchData = input.data();
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

#define TBLOB_TOP_RESULT(precision)\
    case InferenceEngine::Precision::precision  : {\
        using myBlobType = InferenceEngine::PrecisionTrait<Precision::precision>::value_type;\
        TBlob<myBlobType> &tblob = dynamic_cast<TBlob<myBlobType> &>(input);\
        TopResults(n, tblob, output);\
        break;\
    }

/**
 * @brief Gets the top n results from a blob
 * @param n Top n count
 * @param input 1D blob that contains probabilities
 * @param output Vector of indexes for the top n places
 */
inline void TopResults(unsigned int n, Blob &input, std::vector<unsigned> &output) {
    switch (input.getTensorDesc().getPrecision()) {
        TBLOB_TOP_RESULT(FP32);
        TBLOB_TOP_RESULT(FP16);
        TBLOB_TOP_RESULT(Q78);
        TBLOB_TOP_RESULT(I16);
        TBLOB_TOP_RESULT(U8);
        TBLOB_TOP_RESULT(I8);
        TBLOB_TOP_RESULT(U16);
        TBLOB_TOP_RESULT(I32);
        default:
            THROW_IE_EXCEPTION << "cannot locate blob for precision: " << input.getTensorDesc().getPrecision();
    }
}

#undef TBLOB_TOP_RESULT

/**
 * @brief Copies a 8-bit RGB image to the blob.
 * Throws an exception in case of dimensions or input size mismatch
 * @tparam data_t Type of the target blob
 * @param RGB8 8-bit RGB image
 * @param RGB8_size Size of the image
 * @param blob Target blob to write image to
 */
template<typename data_t>
void copyFromRGB8(uint8_t *RGB8, size_t RGB8_size, InferenceEngine::TBlob<data_t> *blob) {
    SizeVector dims = blob->getTensorDesc().getDims();
    if (4 != dims.size())
        THROW_IE_EXCEPTION << "Cannot write data to input blob! Blob has incorrect dimensions size "
                           << dims.size();
    size_t num_channels = dims[1];  // because RGB
    size_t num_images = dims[0];
    size_t w = dims[3];
    size_t h = dims[2];
    size_t nPixels = w * h;

    if (RGB8_size != w * h * num_channels * num_images)
        THROW_IE_EXCEPTION << "input pixels mismatch, expecting " << w * h * num_channels * num_images
                           << " bytes, got: " << RGB8_size;

    std::vector<data_t *> dataArray;
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
 * @brief Splits the RGB channels to either I16 Blob or float blob.
 * The image buffer is assumed to be packed with no support for strides.
 * @param imgBufRGB8 Packed 24bit RGB image (3 bytes per pixel: R-G-B)
 * @param lengthbytesSize Size in bytes of the RGB image. It is equal to amount of pixels times 3 (number of channels)
 * @param input Blob to contain the split image (to 3 channels)
 */
inline void ConvertImageToInput(unsigned char *imgBufRGB8, size_t lengthbytesSize, Blob &input) {
    TBlob<float> *float_input = dynamic_cast<TBlob<float> *>(&input);
    if (float_input != nullptr) copyFromRGB8(imgBufRGB8, lengthbytesSize, float_input);

    TBlob<short> *short_input = dynamic_cast<TBlob<short> *>(&input);
    if (short_input != nullptr) copyFromRGB8(imgBufRGB8, lengthbytesSize, short_input);

    TBlob<uint8_t> *byte_input = dynamic_cast<TBlob<uint8_t> *>(&input);
    if (byte_input != nullptr) copyFromRGB8(imgBufRGB8, lengthbytesSize, byte_input);
}

/**
 * @brief Copies data from a certain precision to float
 * @param dst Pointer to an output float buffer, must be allocated before the call
 * @param src Source blob to take data from
 */
template<typename T>
void copyToFloat(float *dst, const InferenceEngine::Blob *src) {
    if (!dst) {
        return;
    }
    const InferenceEngine::TBlob<T> *t_blob = dynamic_cast<const InferenceEngine::TBlob<T> *>(src);
    if (t_blob == nullptr) {
        THROW_IE_EXCEPTION << "input type is " << src->getTensorDesc().getPrecision() << " but input is not " << typeid(T).name();
    }

    const T *srcPtr = t_blob->readOnly();
    if (srcPtr == nullptr) {
        THROW_IE_EXCEPTION << "Input data was not allocated.";
    }
    for (size_t i = 0; i < t_blob->size(); i++) dst[i] = srcPtr[i];
}

}  // namespace InferenceEngine
