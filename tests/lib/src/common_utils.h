// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>
#include <openvino/openvino.hpp>
#include <limits>
#include <random>


template <typename T>
using uniformDistribution = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;


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
template<typename T, typename T2>
void fillTensorRandom(ov::Tensor& tensor,
                      T rand_min = std::numeric_limits<uint8_t>::min(),
                      T rand_max = std::numeric_limits<uint8_t>::max()) {
    std::mt19937 gen(0);
    size_t tensor_size = tensor.get_size();
    if (0 == tensor_size) {
        throw std::runtime_error(
            "Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference");
    }
    T* data = tensor.data<T>();
    uniformDistribution<T2> distribution(rand_min, rand_max);
    for (size_t i = 0; i < tensor_size; i++)
        data[i] = static_cast<T>(distribution(gen));
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
        auto input_tensor = infer_request.get_tensor(inputs[i]);
        auto type = inputs[i].get_element_type();
        if (type == ov::element::f32) {
            fillTensorRandom<float, float>(input_tensor);
        } else if (type == ov::element::f64) {
            fillTensorRandom<double, double>(input_tensor);
        } else if (type == ov::element::f16) {
            fillTensorRandom<ov::float16, float>(input_tensor);
        } else if (type == ov::element::i32) {
            fillTensorRandom<int32_t, int32_t>(input_tensor);
        } else if (type == ov::element::i64) {
            fillTensorRandom<int64_t, int64_t>(input_tensor);
        } else if ((type == ov::element::u8) || (type == ov::element::boolean)) {
            // uniform_int_distribution<uint8_t> is not allowed in the C++17
            // standard and vs2017/19
            fillTensorRandom<uint8_t, uint32_t>(input_tensor);
        } else if (type == ov::element::i8) {
            // uniform_int_distribution<int8_t> is not allowed in the C++17 standard
            // and vs2017/19
            fillTensorRandom<int8_t, int32_t>(input_tensor,
                                              std::numeric_limits<int8_t>::min(),
                                              std::numeric_limits<int8_t>::max());
        } else if (type == ov::element::u16) {
            fillTensorRandom<uint16_t, uint16_t>(input_tensor);
        } else if (type == ov::element::i16) {
            fillTensorRandom<int16_t, int16_t>(input_tensor);
        } else if (type == ov::element::boolean) {
            fillTensorRandom<uint8_t, uint32_t>(input_tensor, 0, 1);
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
