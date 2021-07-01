// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "inputs_filling.hpp"

#include <format_reader_ptr.h>

#include <algorithm>
#include <memory>
#include <samples/slog.hpp>
#include <string>
#include <utility>
#include <vector>

using namespace InferenceEngine;

#ifdef USE_OPENCV
static const std::vector<std::string> supported_image_extensions = {"bmp", "dib", "jpeg", "jpg", "jpe", "jp2",  "png",
                                                                    "pbm", "pgm", "ppm",  "sr",  "ras", "tiff", "tif"};
#else
static const std::vector<std::string> supported_image_extensions = {"bmp"};
#endif
static const std::vector<std::string> supported_binary_extensions = {"bin"};

std::vector<std::string> filterFilesByExtensions(const std::vector<std::string>& filePaths, const std::vector<std::string>& extensions) {
    std::vector<std::string> filtered;
    auto getExtension = [](const std::string& name) {
        auto extensionPosition = name.rfind('.', name.size());
        return extensionPosition == std::string::npos ? "" : name.substr(extensionPosition + 1, name.size() - 1);
    };
    for (auto& filePath : filePaths) {
        auto extension = getExtension(filePath);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (std::find(extensions.begin(), extensions.end(), extension) != extensions.end()) {
            filtered.push_back(filePath);
        }
    }
    return filtered;
}

template <typename T>
void fillBlobImage(Blob::Ptr& inputBlob, const std::vector<std::string>& filePaths, const size_t& batchSize, const benchmark_app::InputInfo& app_info,
                   const size_t& requestId, const size_t& inputId, const size_t& inputSize) {
    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    if (!minput) {
        IE_THROW() << "We expect inputBlob to be inherited from MemoryBlob in "
                      "fillBlobImage, "
                   << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer
    // happens
    auto minputHolder = minput->wmap();
    auto inputBlobData = minputHolder.as<T*>();

    /** Collect images data ptrs **/
    std::vector<std::shared_ptr<uint8_t>> vreader;
    vreader.reserve(batchSize);

    for (size_t i = 0ULL, inputIndex = requestId * batchSize * inputSize + inputId; i < batchSize; i++, inputIndex += inputSize) {
        inputIndex %= filePaths.size();

        slog::info << "Prepare image " << filePaths[inputIndex] << slog::endl;
        FormatReader::ReaderPtr reader(filePaths[inputIndex].c_str());
        if (reader.get() == nullptr) {
            slog::warn << "Image " << filePaths[inputIndex] << " cannot be read!" << slog::endl << slog::endl;
            continue;
        }

        /** Getting image data **/
        std::shared_ptr<uint8_t> imageData(reader->getData(app_info.width(), app_info.height()));
        if (imageData) {
            vreader.push_back(imageData);
        }
    }

    /** Fill input tensor with images. First b channel, then g and r channels **/
    const size_t numChannels = app_info.channels();
    const size_t width = app_info.width();
    const size_t height = app_info.height();
    /** Iterate over all input images **/
    for (size_t imageId = 0; imageId < vreader.size(); ++imageId) {
        /** Iterate over all width **/
        for (size_t w = 0; w < app_info.width(); ++w) {
            /** Iterate over all height **/
            for (size_t h = 0; h < app_info.height(); ++h) {
                /** Iterate over all channels **/
                for (size_t ch = 0; ch < numChannels; ++ch) {
                    /**          [images stride + channels stride + pixel id ] all in
                     * bytes            **/
                    size_t offset = imageId * numChannels * width * height + (((app_info.layout == "NCHW") || (app_info.layout == "CHW"))
                                                                                  ? (ch * width * height + h * width + w)
                                                                                  : (h * width * numChannels + w * numChannels + ch));
                    inputBlobData[offset] = static_cast<T>(vreader.at(imageId).get()[h * width * numChannels + w * numChannels + ch]);
                }
            }
        }
    }
}

template <typename T>
void fillBlobBinary(Blob::Ptr& inputBlob, const std::vector<std::string>& filePaths, const size_t& batchSize, const size_t& requestId, const size_t& inputId,
                    const size_t& inputSize) {
    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    if (!minput) {
        IE_THROW() << "We expect inputBlob to be inherited from MemoryBlob in "
                      "fillBlobBinary, "
                   << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer
    // happens
    auto minputHolder = minput->wmap();

    auto inputBlobData = minputHolder.as<char*>();
    for (size_t i = 0ULL, inputIndex = requestId * batchSize * inputSize + inputId; i < batchSize; i++, inputIndex += inputSize) {
        inputIndex %= filePaths.size();

        slog::info << "Prepare binary file " << filePaths[inputIndex] << slog::endl;
        std::ifstream binaryFile(filePaths[inputIndex], std::ios_base::binary | std::ios_base::ate);
        if (!binaryFile) {
            IE_THROW() << "Cannot open " << filePaths[inputIndex];
        }

        auto fileSize = static_cast<std::size_t>(binaryFile.tellg());
        binaryFile.seekg(0, std::ios_base::beg);
        if (!binaryFile.good()) {
            IE_THROW() << "Can not read " << filePaths[inputIndex];
        }
        auto inputSize = inputBlob->size() * sizeof(T) / batchSize;
        if (fileSize != inputSize) {
            IE_THROW() << "File " << filePaths[inputIndex] << " contains " << std::to_string(fileSize)
                       << " bytes "
                          "but the network expects "
                       << std::to_string(inputSize);
        }
        binaryFile.read(&inputBlobData[i * inputSize], inputSize);
    }
}

template <typename T>
using uniformDistribution =
    typename std::conditional<std::is_floating_point<T>::value, std::uniform_real_distribution<T>,
                              typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;

template <typename T, typename T2>
void fillBlobRandom(Blob::Ptr& inputBlob, T rand_min = std::numeric_limits<uint8_t>::min(), T rand_max = std::numeric_limits<uint8_t>::max()) {
    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    if (!minput) {
        IE_THROW() << "We expect inputBlob to be inherited from MemoryBlob in "
                      "fillBlobRandom, "
                   << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer
    // happens
    auto minputHolder = minput->wmap();

    auto inputBlobData = minputHolder.as<T*>();
    std::mt19937 gen(0);
    uniformDistribution<T2> distribution(rand_min, rand_max);
    for (size_t i = 0; i < inputBlob->size(); i++) {
        inputBlobData[i] = static_cast<T>(distribution(gen));
    }
}

template <typename T>
void fillBlobImInfo(Blob::Ptr& inputBlob, const size_t& batchSize, std::pair<size_t, size_t> image_size) {
    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    if (!minput) {
        IE_THROW() << "We expect inputBlob to be inherited from MemoryBlob in "
                      "fillBlobImInfo, "
                   << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer
    // happens
    auto minputHolder = minput->wmap();

    auto inputBlobData = minputHolder.as<T*>();
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

void fillBlobs(const std::vector<std::string>& inputFiles, const size_t& batchSize, benchmark_app::InputsInfo& app_inputs_info,
               std::vector<InferReqWrap::Ptr> requests) {
    std::vector<std::pair<size_t, size_t>> input_image_sizes;
    for (auto& item : app_inputs_info) {
        if (item.second.isImage()) {
            input_image_sizes.push_back(std::make_pair(item.second.width(), item.second.height()));
        }
        slog::info << "Network input '" << item.first << "' precision " << item.second.precision << ", dimensions (" << item.second.layout << "): ";
        for (const auto& i : item.second.shape) {
            slog::info << i << " ";
        }
        slog::info << slog::endl;
    }

    size_t imageInputCount = input_image_sizes.size();
    size_t binaryInputCount = app_inputs_info.size() - imageInputCount;

    std::vector<std::string> binaryFiles;
    std::vector<std::string> imageFiles;

    if (inputFiles.empty()) {
        slog::warn << "No input files were given: all inputs will be filled with "
                      "random values!"
                   << slog::endl;
    } else {
        binaryFiles = filterFilesByExtensions(inputFiles, supported_binary_extensions);
        std::sort(std::begin(binaryFiles), std::end(binaryFiles));

        auto binaryToBeUsed = binaryInputCount * batchSize * requests.size();
        if (binaryToBeUsed > 0 && binaryFiles.empty()) {
            std::stringstream ss;
            for (auto& ext : supported_binary_extensions) {
                if (!ss.str().empty()) {
                    ss << ", ";
                }
                ss << ext;
            }
            slog::warn << "No supported binary inputs found! Please check your file "
                          "extensions: "
                       << ss.str() << slog::endl;
        } else if (binaryToBeUsed > binaryFiles.size()) {
            slog::warn << "Some binary input files will be duplicated: " << binaryToBeUsed << " files are required but only " << binaryFiles.size()
                       << " are provided" << slog::endl;
        } else if (binaryToBeUsed < binaryFiles.size()) {
            slog::warn << "Some binary input files will be ignored: only " << binaryToBeUsed << " are required from " << binaryFiles.size() << slog::endl;
        }

        imageFiles = filterFilesByExtensions(inputFiles, supported_image_extensions);
        std::sort(std::begin(imageFiles), std::end(imageFiles));

        auto imagesToBeUsed = imageInputCount * batchSize * requests.size();
        if (imagesToBeUsed > 0 && imageFiles.empty()) {
            std::stringstream ss;
            for (auto& ext : supported_image_extensions) {
                if (!ss.str().empty()) {
                    ss << ", ";
                }
                ss << ext;
            }
            slog::warn << "No supported image inputs found! Please check your file "
                          "extensions: "
                       << ss.str() << slog::endl;
        } else if (imagesToBeUsed > imageFiles.size()) {
            slog::warn << "Some image input files will be duplicated: " << imagesToBeUsed << " files are required but only " << imageFiles.size()
                       << " are provided" << slog::endl;
        } else if (imagesToBeUsed < imageFiles.size()) {
            slog::warn << "Some image input files will be ignored: only " << imagesToBeUsed << " are required from " << imageFiles.size() << slog::endl;
        }
    }

    for (size_t requestId = 0; requestId < requests.size(); requestId++) {
        slog::info << "Infer Request " << requestId << " filling" << slog::endl;

        size_t imageInputId = 0;
        size_t binaryInputId = 0;
        for (auto& item : app_inputs_info) {
            Blob::Ptr inputBlob = requests.at(requestId)->getBlob(item.first);
            auto app_info = app_inputs_info.at(item.first);
            auto precision = app_info.precision;
            if (app_info.isImage()) {
                if (!imageFiles.empty()) {
                    // Fill with Images
                    if (precision == InferenceEngine::Precision::FP32) {
                        fillBlobImage<float>(inputBlob, imageFiles, batchSize, app_info, requestId, imageInputId++, imageInputCount);
                    } else if (precision == InferenceEngine::Precision::FP16) {
                        fillBlobImage<short>(inputBlob, imageFiles, batchSize, app_info, requestId, imageInputId++, imageInputCount);
                    } else if (precision == InferenceEngine::Precision::I32) {
                        fillBlobImage<int32_t>(inputBlob, imageFiles, batchSize, app_info, requestId, imageInputId++, imageInputCount);
                    } else if (precision == InferenceEngine::Precision::I64) {
                        fillBlobImage<int64_t>(inputBlob, imageFiles, batchSize, app_info, requestId, imageInputId++, imageInputCount);
                    } else if (precision == InferenceEngine::Precision::U8) {
                        fillBlobImage<uint8_t>(inputBlob, imageFiles, batchSize, app_info, requestId, imageInputId++, imageInputCount);
                    } else {
                        IE_THROW() << "Input precision is not supported for " << item.first;
                    }
                    continue;
                }
            } else {
                if (!binaryFiles.empty()) {
                    // Fill with binary files
                    if (precision == InferenceEngine::Precision::FP32) {
                        fillBlobBinary<float>(inputBlob, binaryFiles, batchSize, requestId, binaryInputId++, binaryInputCount);
                    } else if (precision == InferenceEngine::Precision::FP16) {
                        fillBlobBinary<short>(inputBlob, binaryFiles, batchSize, requestId, binaryInputId++, binaryInputCount);
                    } else if (precision == InferenceEngine::Precision::I32) {
                        fillBlobBinary<int32_t>(inputBlob, binaryFiles, batchSize, requestId, binaryInputId++, binaryInputCount);
                    } else if (precision == InferenceEngine::Precision::I64) {
                        fillBlobBinary<int64_t>(inputBlob, binaryFiles, batchSize, requestId, binaryInputId++, binaryInputCount);
                    } else if ((precision == InferenceEngine::Precision::U8) || (precision == InferenceEngine::Precision::BOOL)) {
                        fillBlobBinary<uint8_t>(inputBlob, binaryFiles, batchSize, requestId, binaryInputId++, binaryInputCount);
                    } else {
                        IE_THROW() << "Input precision is not supported for " << item.first;
                    }
                    continue;
                }

                if (app_info.isImageInfo() && (input_image_sizes.size() == 1)) {
                    // Most likely it is image info: fill with image information
                    auto image_size = input_image_sizes.at(0);
                    slog::info << "Fill input '" << item.first << "' with image size " << image_size.first << "x" << image_size.second << slog::endl;
                    if (precision == InferenceEngine::Precision::FP32) {
                        fillBlobImInfo<float>(inputBlob, batchSize, image_size);
                    } else if (precision == InferenceEngine::Precision::FP16) {
                        fillBlobImInfo<short>(inputBlob, batchSize, image_size);
                    } else if (precision == InferenceEngine::Precision::I32) {
                        fillBlobImInfo<int32_t>(inputBlob, batchSize, image_size);
                    } else if (precision == InferenceEngine::Precision::I64) {
                        fillBlobImInfo<int64_t>(inputBlob, batchSize, image_size);
                    } else {
                        IE_THROW() << "Input precision is not supported for image info!";
                    }
                    continue;
                }
            }
            // Fill random
            slog::info << "Fill input '" << item.first << "' with random values (" << std::string((app_info.isImage() ? "image" : "some binary data"))
                       << " is expected)" << slog::endl;
            if (precision == InferenceEngine::Precision::FP32) {
                fillBlobRandom<float, float>(inputBlob);
            } else if (precision == InferenceEngine::Precision::FP16) {
                fillBlobRandom<short, short>(inputBlob);
            } else if (precision == InferenceEngine::Precision::I32) {
                fillBlobRandom<int32_t, int32_t>(inputBlob);
            } else if (precision == InferenceEngine::Precision::I64) {
                fillBlobRandom<int64_t, int64_t>(inputBlob);
            } else if (precision == InferenceEngine::Precision::U8) {
                // uniform_int_distribution<uint8_t> is not allowed in the C++17
                // standard and vs2017/19
                fillBlobRandom<uint8_t, uint32_t>(inputBlob);
            } else if (precision == InferenceEngine::Precision::I8) {
                // uniform_int_distribution<int8_t> is not allowed in the C++17 standard
                // and vs2017/19
                fillBlobRandom<int8_t, int32_t>(inputBlob);
            } else if (precision == InferenceEngine::Precision::U16) {
                fillBlobRandom<uint16_t, uint16_t>(inputBlob);
            } else if (precision == InferenceEngine::Precision::I16) {
                fillBlobRandom<int16_t, int16_t>(inputBlob);
            } else if (precision == InferenceEngine::Precision::BOOL) {
                fillBlobRandom<uint8_t, uint32_t>(inputBlob, 0, 1);
            } else {
                IE_THROW() << "Input precision is not supported for " << item.first;
            }
        }
    }
}
