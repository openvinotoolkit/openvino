// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>
#include <openvino/openvino.hpp>
#include <limits>


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
 * @brief Fill InferenceEngine blob with random values
 */
template<typename T>
void fillBlobRandom(InferenceEngine::Blob::Ptr &inputBlob) {
    InferenceEngine::MemoryBlob::Ptr minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(inputBlob);
    // locked memory holder should be alive all time while access to its buffer happens
    auto minputHolder = minput->wmap();

    auto inputBlobData = minputHolder.as<T *>();
    for (size_t i = 0; i < inputBlob->size(); i++) {
        auto rand_max = RAND_MAX;
        inputBlobData[i] = (T) rand() / static_cast<T>(rand_max) * 10;
    }
}


/**
 * @brief Fill InferenceEngine tensor with random values (OV API 2.0)
 */
template<typename T, typename U>
ov::Tensor fillTensorRandom(T &input) {
    ov::Tensor tensor{input.get_element_type(), input.get_shape()};
    std::vector<U> values(ov::shape_size(input.get_shape()));

    for (size_t i = 0; i < values.size(); ++i)
        values[i] = 1 + static_cast<U> (rand()) / (static_cast<U> (RAND_MAX / (std::numeric_limits<U>::max() - 1)));

    std::memcpy(tensor.data(), values.data(), sizeof(U) * values.size());

    return tensor;
}


/**
 * @brief Fill InferenceEngine blob with image information (OV API 1)
 */
template<typename T>
void fillBlobImInfo(InferenceEngine::Blob::Ptr &inputBlob,
                    const size_t &batchSize,
                    std::pair<size_t, size_t> image_size) {
    InferenceEngine::MemoryBlob::Ptr minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(inputBlob);
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
 * @brief Fill infer_request tensors with random values (OV API 2)
 */
template<typename T>
void fillTensors(ov::InferRequest &infer_request, std::vector<T> &inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        ov::Tensor input_tensor;

        if (inputs[i].get_element_type() == ov::element::f32) {
            input_tensor = fillTensorRandom<T, float>(inputs[i]);
        } else if (inputs[i].get_element_type() == ov::element::f16) {
            input_tensor = fillTensorRandom<T, short>(inputs[i]);
        } else if (inputs[i].get_element_type() == ov::element::i32) {
            input_tensor = fillTensorRandom<T, int32_t>(inputs[i]);
        } else if (inputs[i].get_element_type() == ov::element::u8) {
            input_tensor = fillTensorRandom<T, uint8_t>(inputs[i]);
        } else if (inputs[i].get_element_type() == ov::element::i8) {
            input_tensor = fillTensorRandom<T, int8_t>(inputs[i]);
        } else if (inputs[i].get_element_type() == ov::element::u16) {
            input_tensor = fillTensorRandom<T, uint16_t>(inputs[i]);
        } else if (inputs[i].get_element_type() == ov::element::i16) {
            input_tensor = fillTensorRandom<T, int16_t>(inputs[i]);
        } else {
            throw std::logic_error(
                    "Input precision is not supported for " + inputs[i].get_element_type().get_type_name());
        }
        infer_request.set_input_tensor(i, input_tensor);
    }
}


/**
 * @brief Fill InferRequest blobs with random values or image information (OV API 1)
 */
void fillBlobs(InferenceEngine::InferRequest inferRequest,
               const InferenceEngine::ConstInputsDataMap &inputsInfo,
               const size_t &batchSize);