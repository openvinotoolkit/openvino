// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>

using namespace InferenceEngine;

/**
 * @brief Determine if InferenceEngine blob means image or not (OV API 1.0)
 */
template<typename T>
static bool isImage(const T &blob) {
    auto descriptor = blob->getTensorDesc();
    if (descriptor.getLayout() != InferenceEngine::NCHW) {
        return false;
    }
    auto channels = descriptor.getDims()[1];
    return channels == 3;
}

/**
 * @brief Determine if InferenceEngine blob means image or not (OV API 2.0)
 */
static bool isImage(const ov::Output<ov::Node> &input) {
    const auto &layout = ov::layout::get_layout(input);
    if (ov::layout::has_height(layout) && ov::layout::has_width(layout) && ov::layout::has_channels(layout)) {
        return true;
    } else {
        return false;
    }
}


/**
 * @brief Determine if InferenceEngine blob means image information or not (OV API 1.0)
 */
template<typename T>
static bool isImageInfo(const T &blob) {
    auto descriptor = blob->getTensorDesc();
    if (descriptor.getLayout() != InferenceEngine::NC) {
        return false;
    }
    auto channels = descriptor.getDims()[1];
    return (channels >= 2);
}

/**
 * @brief Determine if InferenceEngine blob means image information or not (OV API 2.0)
 */
static bool isImageInfo(const ov::Output<ov::Node> &input) {
    const auto &layout = ov::layout::get_layout(input);
    auto shape = input.get_shape();
    if (ov::layout::has_channels(layout) && shape[ov::layout::channels_idx(layout)] >= 2) {
        return true;
    } else {
        return false;
    }
}


/**
 * @brief Return height and width from provided InferenceEngine tensor description (OV API 1)
 */
inline std::pair<size_t, size_t> getTensorHeightWidth(const InferenceEngine::TensorDesc &desc) {
    const auto &layout = desc.getLayout();
    const auto &dims = desc.getDims();
    const auto &size = dims.size();
    if ((size >= 2) &&
        (layout == InferenceEngine::Layout::NCHW ||
         layout == InferenceEngine::Layout::NHWC ||
         layout == InferenceEngine::Layout::NCDHW ||
         layout == InferenceEngine::Layout::NDHWC ||
         layout == InferenceEngine::Layout::OIHW ||
         layout == InferenceEngine::Layout::GOIHW ||
         layout == InferenceEngine::Layout::OIDHW ||
         layout == InferenceEngine::Layout::GOIDHW ||
         layout == InferenceEngine::Layout::CHW ||
         layout == InferenceEngine::Layout::HW)) {
        // Regardless of layout, dimensions are stored in fixed order
        return std::make_pair(dims.back(), dims.at(size - 2));
    } else {
        throw std::logic_error("Tensor does not have height and width dimensions");
    }
}


/**
 * @brief Return height and width from provided InferenceEngine tensor description (OV API 2)
 */
inline std::pair<size_t, size_t> getTensorHeightWidth(const ov::Output<ov::Node> &input) {
    const auto &layout = ov::layout::get_layout(input);
    auto shape = input.get_shape();
    auto height_index = ov::layout::height_idx(layout);
    auto width_index = ov::layout::width_idx(layout);
    if (ov::layout::has_height(layout) && ov::layout::has_width(layout)) {
        return std::make_pair(shape[height_index], shape[width_index]);
    } else {
        throw std::logic_error("Tensor does not have height and width dimensions");
    }
}

/**
 * @brief Fill InferenceEngine blob with random values
 */
template<typename T>
void fillBlobRandom(Blob::Ptr &inputBlob) {
    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    // locked memory holder should be alive all time while access to its buffer happens
    auto minputHolder = minput->wmap();

    auto inputBlobData = minputHolder.as<T *>();
    for (size_t i = 0; i < inputBlob->size(); i++) {
        auto rand_max = RAND_MAX;
        inputBlobData[i] = (T) rand() / static_cast<T>(rand_max) * 10;
    }
}

/**
 * @brief Fill InferenceEngine tensor with random values
 */
template<typename T>
ov::Tensor fillTensorRandom(const ov::Output<ov::Node> &input) {
    ov::Tensor tensor{input.get_element_type(), input.get_shape()};
    std::vector<T> values(ov::shape_size(input.get_shape()));
    for (size_t i = 0; i < values.size(); ++i) {
        auto rand_max = RAND_MAX;
        values[i] = (T) rand() / static_cast<T>(rand_max) * 10;
    }
    std::memcpy(tensor.data(), values.data(), sizeof(T) * values.size());
    return tensor;
}


/**
 * @brief Fill InferenceEngine blob with image information
 */
template<typename T>
void fillBlobImInfo(Blob::Ptr &inputBlob,
                    const size_t &batchSize,
                    std::pair<size_t, size_t> image_size) {
    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    // locked memory holder should be alive all time while access to its buffer happens
    auto minputHolder = minput->wmap();

    auto inputBlobData = minputHolder.as<T *>();
    for (size_t b = 0; b < batchSize; b++) {
        size_t iminfoSize = inputBlob->size() / batchSize;
        for (size_t i = 0; i < iminfoSize; i++) {
            size_t index = b * iminfoSize + i;
            if (0 == i)
                inputBlobData[index] = static_cast<T>(image_size.first);
            else if (1 == i)
                inputBlobData[index] = static_cast<T>(image_size.second);
            else
                inputBlobData[index] = 1;
        }
    }
}


/**
 * @brief Fill InferenceEngine tensor with image information
 */
template<typename T>
ov::Tensor fillTensorImInfo(const ov::Output<ov::Node> &input,
                            std::pair<size_t, size_t> image_size) {
    ov::Tensor tensor{input.get_element_type(), input.get_shape()};
    std::vector<float> values{static_cast<float>(image_size.first), static_cast<float>(image_size.second)};

    std::memcpy(tensor.data(), values.data(), sizeof(T) * values.size());

    return tensor;
}


/**
 * @brief Fill InferRequest blobs with random values or image information
 */
void fillBlobs(InferenceEngine::InferRequest inferRequest,
               const InferenceEngine::ConstInputsDataMap &inputsInfo,
               const size_t &batchSize);

/**
 * @brief Fill InferRequest tensors with random values or image information
 */
void fillTensors(ov::InferRequest &infer_request,
                 const std::vector<ov::Output<ov::Node>> &inputs);
