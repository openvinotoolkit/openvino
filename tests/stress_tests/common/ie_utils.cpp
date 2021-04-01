// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_utils.h"

#include <inference_engine.hpp>

using namespace InferenceEngine;

/**
 * @brief Fill InferRequest blobs with random values or image information
 */
void fillBlobs(InferenceEngine::InferRequest inferRequest,
        const InferenceEngine::ConstInputsDataMap& inputsInfo,
        const size_t& batchSize) {
  std::vector<std::pair<size_t, size_t>> input_image_sizes;
  for (const ConstInputsDataMap::value_type& item : inputsInfo) {
    if (isImage(item.second))
      input_image_sizes.push_back(getTensorHeightWidth(item.second->getTensorDesc()));
  }

  for (const ConstInputsDataMap::value_type& item : inputsInfo) {
    Blob::Ptr inputBlob = inferRequest.GetBlob(item.first);
    if (isImageInfo(inputBlob) && (input_image_sizes.size() == 1)) {
      // Fill image information
      auto image_size = input_image_sizes.at(0);
      if (item.second->getPrecision() == InferenceEngine::Precision::FP32) {
        fillBlobImInfo<float>(inputBlob, batchSize, image_size);
      } else if (item.second->getPrecision() == InferenceEngine::Precision::FP16) {
        fillBlobImInfo<short>(inputBlob, batchSize, image_size);
      } else if (item.second->getPrecision() == InferenceEngine::Precision::I32) {
        fillBlobImInfo<int32_t>(inputBlob, batchSize, image_size);
      } else {
        THROW_IE_EXCEPTION << "Input precision is not supported for image info!";
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
      THROW_IE_EXCEPTION << "Input precision is not supported for " << item.first;
    }
  }
}