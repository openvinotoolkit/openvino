// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base_matcher.hpp"
#include <precision_utils.h>
#include <format_reader_ptr.h>

namespace Regression { namespace Matchers {

using namespace InferenceEngine;

void loadImage(const std::string &imageFilename, InferenceEngine::Blob::Ptr &blob, bool bgr, int batchNumber) {
    TensorDesc tensDesc = blob->getTensorDesc();
    if (tensDesc.getPrecision() != InferenceEngine::Precision::FP16
        && tensDesc.getPrecision() != InferenceEngine::Precision::FP32
        && tensDesc.getPrecision()!= InferenceEngine::Precision::U8
        && tensDesc.getPrecision()!= InferenceEngine::Precision::I16) {
        THROW_IE_EXCEPTION << "loadImage error: Input must have FP16, FP32 or U8 precision";
    }

    if (tensDesc.getLayout() != NHWC && tensDesc.getLayout() != NCHW) {
        THROW_IE_EXCEPTION << "loadImage error: Input must have NHWC or NHWC layout";
    }

    FormatReader::ReaderPtr reader(imageFilename.c_str());
    if (reader.get() == nullptr) {
        THROW_IE_EXCEPTION << "loadImage error: image " << imageFilename << " cannot be read!";
    }

    size_t w = tensDesc.getDims()[3];
    size_t h = tensDesc.getDims()[2];
    if (reader->width() != w || reader->height() != h) {
        THROW_IE_EXCEPTION << "loadImage error: Input sizes mismatch, got " << reader->width() << "x" << reader->height()
                  << " expecting " << w << "x" << h;
    }

    auto numBlobChannels = tensDesc.getDims()[1];
    size_t numImageChannels = reader->size() / (reader->width() * reader->height());
    if (numBlobChannels != numImageChannels && numBlobChannels != 1) {
        THROW_IE_EXCEPTION << "loadImage error: Input channels mismatch: image channels " << numImageChannels << ", "
                  << "network channels " << numBlobChannels << ", expecting count of image channels are equal "
                  << "to count if network channels or count of network channels are equal to 1";
    }

    auto nPixels = w * h;
    uint8_t *BGR8 = reader->getData().get();
    for (unsigned int i = 0; i < nPixels; i++) {
        for (unsigned int j = 0; j < numBlobChannels; j++) {
            uint8_t val = bgr ? BGR8[i * numImageChannels + j] : BGR8[i * numBlobChannels + (numBlobChannels - j - 1)];
            size_t idx = tensDesc.getLayout() == NHWC ? (i * numBlobChannels + j) : (j * nPixels + i)
                + nPixels * numBlobChannels * batchNumber;
            auto buf = blob->buffer();
            switch (blob->getTensorDesc().getPrecision()) {
            case Precision::U8:
            {
                auto inputDataPtr = buf.as<uint8_t*>();
                inputDataPtr[idx] = val;
                break;
            }
            case Precision::I16:
            {
                auto *inputDataPtr = buf.as<int16_t*>();
                inputDataPtr[idx] = val;
                break;
            }
            case Precision::FP16:
            {
                ie_fp16 *inputDataPtr = buf.as<ie_fp16*>();
                inputDataPtr[idx] = InferenceEngine::PrecisionUtils::f32tof16(static_cast<float>(val));
                break;
            }
            case Precision::FP32:
            {
                auto inputDataPtr = buf.as<float*>();
                inputDataPtr[idx] = static_cast<float>(val);
                break;
            }
            default:
                THROW_IE_EXCEPTION << "Unsupported precision!";
            }
        }
    }
}

void BaseMatcher::checkImgNumber(int dynBatch) {
    InferenceEngine::Core ieCore;
    CNNNetwork net = ieCore.ReadNetwork(config._path_to_models);
    auto numInputs = net.getInputsInfo().size();

    int batch = dynBatch > 0 ? dynBatch : config.batchSize;

    if ((numInputs * batch) > config._paths_to_images.size()) {

        auto readImagesSize = config._paths_to_images.size();
        size_t diff = (numInputs * batch) / readImagesSize;

        for (size_t i = 1; i < diff; i++) {
            for (size_t j = 0; j < readImagesSize; j++) {
                config._paths_to_images.push_back(config._paths_to_images[j]);
            }
        }
        if (readImagesSize * diff != (numInputs * batch)) {
            for (size_t j = 0; j < (numInputs * batch) - readImagesSize * diff; j++) {
                config._paths_to_images.push_back(config._paths_to_images.at(j));
            }
        }
    } else if ((numInputs * batch) < config._paths_to_images.size()) {
        while (config._paths_to_images.size() != batch * numInputs) {
            auto name = config._paths_to_images.back();
            std::cout << "[WARNING]: Image " << name << " skipped!" << std::endl;
            config._paths_to_images.pop_back();
        }
    }
}

}  // namepspace Matchers
}  // namespace Regression
