// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_utils.h"


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
            } else if (item.second->getPrecision() == InferenceEngine::Precision::FP64) {
                fillBlobImInfo<double>(inputBlob, batchSize, image_size);
            } else if (item.second->getPrecision() == InferenceEngine::Precision::FP16) {
                fillBlobImInfo<short>(inputBlob, batchSize, image_size);
            } else if (item.second->getPrecision() == InferenceEngine::Precision::I32) {
                fillBlobImInfo<int32_t>(inputBlob, batchSize, image_size);
            } else if (item.second->getPrecision() == InferenceEngine::Precision::I64) {
                fillBlobImInfo<int64_t>(inputBlob, batchSize, image_size);
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
        } else if (item.second->getPrecision() == InferenceEngine::Precision::FP64) {
            fillBlobRandom<double>(inputBlob);
        } else if (item.second->getPrecision() == InferenceEngine::Precision::FP16) {
            fillBlobRandom<short>(inputBlob);
        } else if (item.second->getPrecision() == InferenceEngine::Precision::I32) {
            fillBlobRandom<int32_t>(inputBlob);
        } else if (item.second->getPrecision() == InferenceEngine::Precision::I64) {
            fillBlobRandom<int64_t>(inputBlob);
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
 * @brief Get input/output precision
 */
ov::element::Type getType(std::string value,
                          const std::unordered_map<std::string,
                          ov::element::Type> &supported_precisions) {
    std::transform(value.begin(), value.end(), value.begin(), ::toupper);

    const auto precision = supported_precisions.find(value);
    if (precision == supported_precisions.end()) {
        throw std::logic_error("\"" + value + "\"" + " is not a valid precision");
    }

    return precision->second;
}

ov::element::Type getType(const std::string &value) {
    static const std::unordered_map<std::string, ov::element::Type> supported_precisions = {
        {"FP32", ov::element::f32},
        {"FP16", ov::element::f16},
        {"BF16", ov::element::bf16},
        {"U64", ov::element::u64},
        {"I64", ov::element::i64},
        {"U32", ov::element::u32},
        {"I32", ov::element::i32},
        {"U16", ov::element::u16},
        {"I16", ov::element::i16},
        {"U8", ov::element::u8},
        {"I8", ov::element::i8},
        {"BOOL", ov::element::boolean},
    };

    return getType(value, supported_precisions);
}