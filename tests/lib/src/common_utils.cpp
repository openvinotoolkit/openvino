// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_utils.h"

#include <inference_engine.hpp>
#include <openvino/openvino.hpp>


/**
 * @brief Fill InferRequest blobs with random values or image information
 */
void fillBlobs(InferenceEngine::InferRequest inferRequest,
               const InferenceEngine::ConstInputsDataMap &inputsInfo,
               const size_t &batchSize) {
    std::vector<std::pair<size_t, size_t>> input_image_sizes;
    for (const InferenceEngine::ConstInputsDataMap::value_type &item: inputsInfo) {
        if (isImage(item.second))
            input_image_sizes.push_back(getTensorHeightWidth(item.second->getTensorDesc()));
    }

    for (const InferenceEngine::ConstInputsDataMap::value_type &item: inputsInfo) {
        InferenceEngine::Blob::Ptr inputBlob = inferRequest.GetBlob(item.first);
        if (isImageInfo(inputBlob) && (input_image_sizes.size() == 1)) {
            // Fill image information
            auto image_size = input_image_sizes.at(0);
            if (item.second->getPrecision() == InferenceEngine::Precision::FP32) {
                fillBlobImInfo<float>(inputBlob, batchSize, image_size);
            } else if (item.second->getPrecision() == InferenceEngine::Precision::FP16) {
                fillBlobImInfo<short>(inputBlob, batchSize, image_size);
            } else if (item.second->getPrecision() == InferenceEngine::Precision::I32) {
                fillBlobImInfo<int32_t>(inputBlob, batchSize, image_size);
            } else if (item.second->getPrecision() == InferenceEngine::Precision::U8) {
                fillBlobImInfo<uint8_t>(inputBlob, batchSize, image_size);
            } else {
                throw std::logic_error("Input precision is not supported for image info!");
            }
            continue;
        }
        // Fill random
        if (item.second->getPrecision() == InferenceEngine::Precision::FP32) {
            fillBlobRandom<float>(inputBlob);
        } else if (item.second->getPrecision() == InferenceEngine::Precision::FP16) {
            fillBlobRandom<short>(inputBlob);
        } else if (item.second->getPrecision() == InferenceEngine::Precision::I32) {
            fillBlobRandom<int32_t>(inputBlob);
        } else if (item.second->getPrecision() == InferenceEngine::Precision::U8) {
            fillBlobRandom<uint8_t>(inputBlob);
        } else if (item.second->getPrecision() == InferenceEngine::Precision::I8) {
            fillBlobRandom<int8_t>(inputBlob);
        } else if (item.second->getPrecision() == InferenceEngine::Precision::U16) {
            fillBlobRandom<uint16_t>(inputBlob);
        } else if (item.second->getPrecision() == InferenceEngine::Precision::I16) {
            fillBlobRandom<int16_t>(inputBlob);
        } else {
            throw std::logic_error("Input precision is not supported for " + item.first);
        }
    }
}

/**
 * @brief Fill infer_request tensors with random values or image information
 */
void fillTensors(ov::InferRequest &infer_request, const std::vector<ov::Output<ov::Node>> &inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        ov::Tensor input_tensor;
        if (inputs[i].get_element_type() == ov::element::f32) {
            input_tensor = fillTensorRandom<float>(inputs[i]);
        } else if (inputs[i].get_element_type() == ov::element::f16) {
            input_tensor = fillTensorRandom<short>(inputs[i]);
        } else if (inputs[i].get_element_type() == ov::element::i32) {
            input_tensor = fillTensorRandom<int32_t>(inputs[i]);
        } else if (inputs[i].get_element_type() == ov::element::u8) {
            input_tensor = fillTensorRandom<uint8_t>(inputs[i]);
        } else if (inputs[i].get_element_type() == ov::element::i8) {
            input_tensor = fillTensorRandom<int8_t>(inputs[i]);
        } else if (inputs[i].get_element_type() == ov::element::u16) {
            input_tensor = fillTensorRandom<uint16_t>(inputs[i]);
        } else if (inputs[i].get_element_type() == ov::element::i16) {
            input_tensor = fillTensorRandom<int16_t>(inputs[i]);
        } else {
            throw std::logic_error(
                    "Input precision is not supported for " + inputs[i].get_element_type().get_type_name());
        }
        infer_request.set_input_tensor(i, input_tensor);
    }
}
