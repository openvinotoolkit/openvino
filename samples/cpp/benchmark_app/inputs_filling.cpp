// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "samples/slog.hpp"
#include "format_reader_ptr.h"

#include "inputs_filling.hpp"
// clang-format on

using namespace InferenceEngine;

#ifdef USE_OPENCV
static const std::vector<std::string> supported_image_extensions =
    {"bmp", "dib", "jpeg", "jpg", "jpe", "jp2", "png", "pbm", "pgm", "ppm", "sr", "ras", "tiff", "tif"};
#else
static const std::vector<std::string> supported_image_extensions = {"bmp"};
#endif
static const std::vector<std::string> supported_binary_extensions = {"bin"};

std::vector<std::string> filterFilesByExtensions(const std::vector<std::string>& filePaths,
                                                 const std::vector<std::string>& extensions) {
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
void fillBlobImage(Blob::Ptr& inputBlob,
                   const std::vector<std::string>& filePaths,
                   const size_t& batchSize,
                   const benchmark_app::InputInfo& app_info,
                   const size_t& requestId,
                   const size_t& inputId,
                   const size_t& inputSize) {
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

    for (size_t i = 0ULL, inputIndex = requestId * batchSize * inputSize + inputId; i < batchSize;
         i++, inputIndex += inputSize) {
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
                    size_t offset = imageId * numChannels * width * height +
                                    (((app_info.layout == "NCHW") || (app_info.layout == "CHW"))
                                         ? (ch * width * height + h * width + w)
                                         : (h * width * numChannels + w * numChannels + ch));
                    inputBlobData[offset] =
                        (static_cast<T>(vreader.at(imageId).get()[h * width * numChannels + w * numChannels + ch]) -
                         static_cast<T>(app_info.mean[ch])) /
                        static_cast<T>(app_info.scale[ch]);
                }
            }
        }
    }
}

template <typename T>
void fillBlobBinary(Blob::Ptr& inputBlob,
                    const std::vector<std::string>& filePaths,
                    const size_t& batchSize,
                    const size_t& requestId,
                    const size_t& inputId,
                    const size_t& inputSize) {
    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    auto adjBatchSize = batchSize;

    // Check layout
    std::stringstream ss;
    auto tensorDesc = inputBlob->getTensorDesc();
    ss << tensorDesc.getLayout();
    auto layout = ss.str();
    std::size_t batchIndex = layout.find("N");
    if (batchIndex == std::string::npos) {
        adjBatchSize = 1;
    } else if (tensorDesc.getDims().at(batchIndex) != batchSize) {
        adjBatchSize = tensorDesc.getDims().at(batchIndex);
    }

    if (!minput) {
        IE_THROW() << "We expect inputBlob to be inherited from MemoryBlob in "
                      "fillBlobBinary, "
                   << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer
    // happens
    auto minputHolder = minput->wmap();

    auto inputBlobData = minputHolder.as<char*>();
    for (size_t i = 0ULL, inputIndex = requestId * adjBatchSize * inputSize + inputId; i < adjBatchSize;
         i++, inputIndex += inputSize) {
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
        auto inputSize = inputBlob->size() * sizeof(T) / adjBatchSize;
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
using uniformDistribution = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;

template <typename T, typename T2>
void fillBlobRandom(Blob::Ptr& inputBlob,
                    T rand_min = std::numeric_limits<uint8_t>::min(),
                    T rand_max = std::numeric_limits<uint8_t>::max()) {
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

void fillBlobs(const std::vector<std::string>& inputFiles,
               const size_t& batchSize,
               benchmark_app::InputsInfo& app_inputs_info,
               std::vector<InferReqWrap::Ptr> requests,
               bool supress) {
    std::vector<std::pair<size_t, size_t>> input_image_sizes;
    for (auto& item : app_inputs_info) {
        if (item.second.partialShape.is_static() && item.second.isImage()) {
            input_image_sizes.push_back(std::make_pair(item.second.width(), item.second.height()));
        }
        if (!supress) {
            slog::info << "Network input '" << item.first << "' precision " << item.second.precision << ", dimensions ("
                       << item.second.layout << "): ";
            for (const auto& i : item.second.tensorShape) {
                slog::info << i << " ";
            }
            slog::info << "(dynamic: " << item.second.partialShape << ")" << slog::endl;
        }
    }

    size_t imageInputCount = input_image_sizes.size();
    size_t binaryInputCount = app_inputs_info.size() - imageInputCount;

    std::vector<std::string> binaryFiles;
    std::vector<std::string> imageFiles;

    if (inputFiles.empty() && !supress) {
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
            if (!supress)
                slog::warn << "No supported binary inputs found! Please check your file "
                              "extensions: "
                           << ss.str() << slog::endl;
        } else if (binaryToBeUsed > binaryFiles.size()) {
            slog::warn << "Some binary input files will be duplicated: " << binaryToBeUsed
                       << " files are required but only " << binaryFiles.size() << " are provided" << slog::endl;
        } else if (binaryToBeUsed < binaryFiles.size()) {
            slog::warn << "Some binary input files will be ignored: only " << binaryToBeUsed << " are required from "
                       << binaryFiles.size() << slog::endl;
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
            if (!supress)
                slog::warn << "No supported image inputs found! Please check your file "
                              "extensions: "
                           << ss.str() << slog::endl;
        } else if (imagesToBeUsed > imageFiles.size()) {
            slog::warn << "Some image input files will be duplicated: " << imagesToBeUsed
                       << " files are required but only " << imageFiles.size() << " are provided" << slog::endl;
        } else if (imagesToBeUsed < imageFiles.size()) {
            slog::warn << "Some image input files will be ignored: only " << imagesToBeUsed << " are required from "
                       << imageFiles.size() << slog::endl;
        }
    }

    for (size_t requestId = 0; requestId < requests.size(); requestId++) {
        if (!supress)
            slog::info << "Infer Request " << requestId << " filling" << slog::endl;

        size_t imageInputId = 0;
        size_t binaryInputId = 0;
        for (auto& item : app_inputs_info) {
            if (item.second.partialShape.is_dynamic())
                requests.at(requestId)->setShape(item.first, item.second.tensorShape);
            Blob::Ptr inputBlob = requests.at(requestId)->getBlob(item.first);
            auto app_info = app_inputs_info.at(item.first);
            auto precision = app_info.precision;
            if (app_info.isImage()) {
                if (!imageFiles.empty()) {
                    // Fill with Images
                    if (precision == InferenceEngine::Precision::FP32) {
                        fillBlobImage<float>(inputBlob,
                                             imageFiles,
                                             batchSize,
                                             app_info,
                                             requestId,
                                             imageInputId++,
                                             imageInputCount);
                    } else if (precision == InferenceEngine::Precision::FP16) {
                        fillBlobImage<short>(inputBlob,
                                             imageFiles,
                                             batchSize,
                                             app_info,
                                             requestId,
                                             imageInputId++,
                                             imageInputCount);
                    } else if (precision == InferenceEngine::Precision::I32) {
                        fillBlobImage<int32_t>(inputBlob,
                                               imageFiles,
                                               batchSize,
                                               app_info,
                                               requestId,
                                               imageInputId++,
                                               imageInputCount);
                    } else if (precision == InferenceEngine::Precision::I64) {
                        fillBlobImage<int64_t>(inputBlob,
                                               imageFiles,
                                               batchSize,
                                               app_info,
                                               requestId,
                                               imageInputId++,
                                               imageInputCount);
                    } else if (precision == InferenceEngine::Precision::U8) {
                        fillBlobImage<uint8_t>(inputBlob,
                                               imageFiles,
                                               batchSize,
                                               app_info,
                                               requestId,
                                               imageInputId++,
                                               imageInputCount);
                    } else {
                        IE_THROW() << "Input precision is not supported for " << item.first;
                    }
                    continue;
                }
            } else {
                if (!binaryFiles.empty()) {
                    // Fill with binary files
                    if (precision == InferenceEngine::Precision::FP32) {
                        fillBlobBinary<float>(inputBlob,
                                              binaryFiles,
                                              batchSize,
                                              requestId,
                                              binaryInputId++,
                                              binaryInputCount);
                    } else if (precision == InferenceEngine::Precision::FP16) {
                        fillBlobBinary<short>(inputBlob,
                                              binaryFiles,
                                              batchSize,
                                              requestId,
                                              binaryInputId++,
                                              binaryInputCount);
                    } else if (precision == InferenceEngine::Precision::I32) {
                        fillBlobBinary<int32_t>(inputBlob,
                                                binaryFiles,
                                                batchSize,
                                                requestId,
                                                binaryInputId++,
                                                binaryInputCount);
                    } else if (precision == InferenceEngine::Precision::I64) {
                        fillBlobBinary<int64_t>(inputBlob,
                                                binaryFiles,
                                                batchSize,
                                                requestId,
                                                binaryInputId++,
                                                binaryInputCount);
                    } else if ((precision == InferenceEngine::Precision::U8) ||
                               (precision == InferenceEngine::Precision::BOOL)) {
                        fillBlobBinary<uint8_t>(inputBlob,
                                                binaryFiles,
                                                batchSize,
                                                requestId,
                                                binaryInputId++,
                                                binaryInputCount);
                    } else {
                        IE_THROW() << "Input precision is not supported for " << item.first;
                    }
                    continue;
                }

                if (app_info.isImageInfo() && (input_image_sizes.size() == 1)) {
                    // Most likely it is image info: fill with image information
                    auto image_size = input_image_sizes.at(0);
                    if (!supress)
                        slog::info << "Fill input '" << item.first << "' with image size " << image_size.first << "x"
                                   << image_size.second << slog::endl;
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
            if (!supress)
                slog::info << "Fill input '" << item.first << "' with random values ("
                           << std::string((app_info.isImage() ? "image" : "some binary data")) << " is expected)"
                           << slog::endl;
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

template <typename T>
Buffer createBlobFromImage(const std::vector<std::string>& files,
                           size_t inputId,
                           const benchmark_app::InputInfo& inputInfo) {
    const size_t batchSize = inputInfo.batch();
    size_t blob_size =
        std::accumulate(inputInfo.tensorShape.begin(), inputInfo.tensorShape.end(), 1, std::multiplies<int>());
    Buffer buff(blob_size, inputInfo.precision);
    auto data = buff.get<T>();

    /** Collect images data ptrs **/
    std::vector<std::shared_ptr<uint8_t>> vreader;
    vreader.reserve(batchSize);

    for (size_t b = 0; b < batchSize; ++b) {
        auto inputIndex = (inputId + b) % files.size();
        slog::info << "Prepare image " << files[inputIndex] << slog::endl;
        FormatReader::ReaderPtr reader(files[inputIndex].c_str());
        if (reader.get() == nullptr) {
            slog::warn << "Image " << files[inputIndex] << " cannot be read!" << slog::endl << slog::endl;
            continue;
        }

        /** Getting image data **/
        std::shared_ptr<uint8_t> imageData(reader->getData(inputInfo.width(), inputInfo.height()));
        if (imageData) {
            vreader.push_back(imageData);
        }
    }

    /** Fill input tensor with image. First b channel, then g and r channels **/
    const size_t numChannels = inputInfo.channels();
    const size_t width = inputInfo.width();
    const size_t height = inputInfo.height();
    /** Iterate over all input images **/
    for (size_t b = 0; b < batchSize; ++b) {
        /** Iterate over all width **/
        for (size_t w = 0; w < width; ++w) {
            /** Iterate over all height **/
            for (size_t h = 0; h < height; ++h) {
                /** Iterate over all channels **/
                for (size_t ch = 0; ch < numChannels; ++ch) {
                    /**          [images stride + channels stride + pixel id ] all in
                     * bytes            **/
                    size_t offset = b * numChannels * width * height +
                                    (((inputInfo.layout == "NCHW") || (inputInfo.layout == "CHW"))
                                         ? (ch * width * height + h * width + w)
                                         : (h * width * numChannels + w * numChannels + ch));
                    data[offset] =
                        (static_cast<T>(vreader.at(b).get()[h * width * numChannels + w * numChannels + ch]) -
                         static_cast<T>(inputInfo.mean[ch])) /
                        static_cast<T>(inputInfo.scale[ch]);
                }
            }
        }
    }

    return buff;
}

template <typename T>
Buffer createBlobImInfo(const std::pair<size_t, size_t>& image_size, const benchmark_app::InputInfo& inputInfo) {
    size_t blob_size =
        std::accumulate(inputInfo.tensorShape.begin(), inputInfo.tensorShape.end(), 1, std::multiplies<int>());
    Buffer buff(blob_size, inputInfo.precision);
    auto data = buff.get<T>();

    const size_t batchSize = inputInfo.batch();  // change
    for (size_t b = 0; b < batchSize; b++) {
        size_t iminfoSize = blob_size / batchSize;
        for (size_t i = 0; i < iminfoSize; i++) {
            size_t index = b * iminfoSize + i;
            if (0 == i)
                data[index] = static_cast<T>(image_size.first);
            else if (1 == i)
                data[index] = static_cast<T>(image_size.second);
            else
                data[index] = 1;
        }
    }

    return buff;
}

template <typename T>
Buffer createBlobFromBinary(const std::vector<std::string>& files,
                            size_t inputId,
                            const benchmark_app::InputInfo& inputInfo) {
    size_t blob_size =
        std::accumulate(inputInfo.tensorShape.begin(), inputInfo.tensorShape.end(), 1, std::multiplies<int>());
    Buffer buff(blob_size, inputInfo.precision);
    auto data = buff.get<char>();

    const size_t batchSize = inputInfo.batch();  // change
    for (size_t b = 0; b < batchSize; ++b) {
        auto inputIndex = (inputId + b) % files.size();
        slog::info << "Prepare binary file " << files[inputIndex] << slog::endl;
        std::ifstream binaryFile(files[inputIndex], std::ios_base::binary | std::ios_base::ate);
        if (!binaryFile) {
            IE_THROW() << "Cannot open " << files[inputIndex];
        }

        auto fileSize = static_cast<std::size_t>(binaryFile.tellg());
        binaryFile.seekg(0, std::ios_base::beg);
        if (!binaryFile.good()) {
            IE_THROW() << "Can not read " << files[inputIndex];
        }
        auto inputSize = blob_size * sizeof(T) / batchSize;
        if (fileSize != inputSize) {
            IE_THROW() << "File " << files[inputIndex] << " contains " << std::to_string(fileSize)
                       << " bytes "
                          "but the network expects "
                       << std::to_string(inputSize);
        }
        binaryFile.read(&data[b * inputSize], inputSize);
    }

    return buff;
}

template <typename T, typename T2>
Buffer createBlobRandom(size_t blobSize,
                        InferenceEngine::Precision precision,
                        T rand_min = std::numeric_limits<uint8_t>::min(),
                        T rand_max = std::numeric_limits<uint8_t>::max()) {
    Buffer buff(blobSize, precision);
    std::mt19937 gen(0);
    auto data = buff.get<T>();
    uniformDistribution<T2> distribution(rand_min, rand_max);
    for (size_t i = 0; i < blobSize; i++) {
        data[i] = static_cast<T>(distribution(gen));
    }

    return buff;
}

Buffer getImageBlob(const std::vector<std::string>& files,
                    size_t inputId,
                    const std::pair<std::string, benchmark_app::InputInfo>& inputInfo) {
    auto precision = inputInfo.second.precision;
    if (precision == InferenceEngine::Precision::FP32) {
        return createBlobFromImage<float>(files, inputId, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::FP16) {
        return createBlobFromImage<short>(files, inputId, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::I32) {
        return createBlobFromImage<int32_t>(files, inputId, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::I64) {
        return createBlobFromImage<int64_t>(files, inputId, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::U8) {
        return createBlobFromImage<uint8_t>(files, inputId, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::I8) {
        return createBlobFromImage<int8_t>(files, inputId, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::U16) {
        return createBlobFromImage<uint16_t>(files, inputId, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::I16) {
        return createBlobFromImage<int16_t>(files, inputId, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::BOOL) {
        return createBlobFromImage<uint8_t>(files, inputId, inputInfo.second);
    } else {
        IE_THROW() << "Input precision is not supported for " << inputInfo.first;
    }
}

Buffer getImInfoBlob(const std::pair<size_t, size_t>& image_size,
                     const std::pair<std::string, benchmark_app::InputInfo>& inputInfo) {
    auto precision = inputInfo.second.precision;
    if (precision == InferenceEngine::Precision::FP32) {
        return createBlobImInfo<float>(image_size, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::FP16) {
        return createBlobImInfo<short>(image_size, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::I32) {
        return createBlobImInfo<int32_t>(image_size, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::I64) {
        return createBlobImInfo<int64_t>(image_size, inputInfo.second);
    } else {
        IE_THROW() << "Input precision is not supported for " << inputInfo.first;
    }
}

Buffer getBinaryBlob(const std::vector<std::string>& files,
                     size_t inputId,
                     const std::pair<std::string, benchmark_app::InputInfo>& inputInfo) {
    auto precision = inputInfo.second.precision;
    if (precision == InferenceEngine::Precision::FP32) {
        return createBlobFromBinary<float>(files, inputId, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::FP16) {
        return createBlobFromBinary<short>(files, inputId, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::I32) {
        return createBlobFromBinary<int32_t>(files, inputId, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::I64) {
        return createBlobFromBinary<int64_t>(files, inputId, inputInfo.second);
    } else if (precision == InferenceEngine::Precision::U8) {
        return createBlobFromBinary<uint8_t>(files, inputId, inputInfo.second);
    } else {
        IE_THROW() << "Input precision is not supported for " << inputInfo.first;
    }
}

Buffer getRandomBlob(const std::pair<std::string, benchmark_app::InputInfo>& inputInfo) {
    auto precision = inputInfo.second.precision;
    size_t blob_size = std::accumulate(inputInfo.second.tensorShape.begin(),
                                       inputInfo.second.tensorShape.end(),
                                       1,
                                       std::multiplies<int>());
    if (precision == InferenceEngine::Precision::FP32) {
        return createBlobRandom<float, float>(blob_size, precision);
    } else if (precision == InferenceEngine::Precision::FP16) {
        return createBlobRandom<short, short>(blob_size, precision);
    } else if (precision == InferenceEngine::Precision::I32) {
        return createBlobRandom<int32_t, int32_t>(blob_size, precision);
    } else if (precision == InferenceEngine::Precision::I64) {
        return createBlobRandom<int64_t, int64_t>(blob_size, precision);
    } else if (precision == InferenceEngine::Precision::U8) {
        // uniform_int_distribution<uint8_t> is not allowed in the C++17
        // standard and vs2017/19
        return createBlobRandom<uint8_t, uint32_t>(blob_size, precision);
    } else if (precision == InferenceEngine::Precision::I8) {
        // uniform_int_distribution<int8_t> is not allowed in the C++17 standard
        // and vs2017/19
        return createBlobRandom<int8_t, int32_t>(blob_size, precision);
    } else if (precision == InferenceEngine::Precision::U16) {
        return createBlobRandom<uint16_t, uint16_t>(blob_size, precision);
    } else if (precision == InferenceEngine::Precision::I16) {
        return createBlobRandom<int16_t, int16_t>(blob_size, precision);
    } else if (precision == InferenceEngine::Precision::BOOL) {
        return createBlobRandom<uint8_t, uint32_t>(blob_size, precision, 0, 1);
    } else {
        IE_THROW() << "Input precision is not supported for " << inputInfo.first;
    }
}

std::map<std::string, std::vector<Buffer>> prepareCachedBlobs(
    std::map<std::string, std::vector<std::string>>& inputFiles,
    std::vector<benchmark_app::InputsInfo>& app_inputs_info) {
    std::map<std::string, std::vector<Buffer>> cachedBlobs;
    if (app_inputs_info.empty()) {
        throw std::logic_error("Inputs Info for network is empty!");
    }

    std::vector<std::pair<size_t, size_t>> input_image_sizes;
    for (auto& inputs_info : app_inputs_info) {
        for (auto& input : inputs_info) {
            if (input.second.isImage()) {
                input_image_sizes.push_back(std::make_pair(input.second.width(), input.second.height()));
            }
            slog::info << "Network input '" << input.first << "' precision " << input.second.precision
                       << ", dimensions (" << input.second.layout << "): ";
            slog::info << getShapeString(input.second.tensorShape);
            slog::info << "(dynamic: " << input.second.partialShape << ")" << slog::endl;
        }
    }

    for (auto& files : inputFiles) {
        if (!files.first.empty() && app_inputs_info[0].find(files.first) == app_inputs_info[0].end()) {
            throw std::logic_error("Input name" + files.first +
                                   "used with files doesn't correspond any network's input");
        }

        std::string input_name = files.first.empty() ? app_inputs_info[0].begin()->first : files.first;
        auto input = app_inputs_info[0].at(input_name);
        if (input.isImage()) {
            files.second = filterFilesByExtensions(files.second, supported_image_extensions);
        } else {
            if (input.isImageInfo() && input_image_sizes.size() == app_inputs_info.size()) {
                slog::info << "Input " << input_name
                           << " probably is image info. All files for this input will"
                              "be ignored."
                           << slog::endl;
                continue;
            } else {
                files.second = filterFilesByExtensions(files.second, supported_binary_extensions);
            }
        }

        if (files.second.empty()) {
            throw std::logic_error("No suitable files for input found!");
        }

        size_t filesToBeUsed = 0;
        size_t shapesToBeUsed = 0;
        if (files.second.size() > app_inputs_info.size()) {
            shapesToBeUsed = app_inputs_info.size();
            filesToBeUsed = files.second.size() - files.second.size() % app_inputs_info.size();
            if (filesToBeUsed != files.second.size()) {
                slog::warn << "Number of files must be a multiple of the number of shapes for certain input. Only " +
                                  std::to_string(filesToBeUsed) + "files will be added."
                           << slog::endl;
            }
            while (files.second.size() != filesToBeUsed) {
                files.second.pop_back();
            }
        } else {
            shapesToBeUsed = app_inputs_info.size() - app_inputs_info.size() % files.second.size();
            filesToBeUsed = files.second.size();
            if (shapesToBeUsed != app_inputs_info.size()) {
                slog::warn
                    << "Number of tensor shapes must be a multiple of the number of files for certain input. Only " +
                           std::to_string(shapesToBeUsed) + "files will be added."
                    << slog::endl;
            }
            while (app_inputs_info.size() != shapesToBeUsed) {
                app_inputs_info.pop_back();
                input_image_sizes.pop_back();
            }
        }

        slog::info << "For input " << files.first << "next file will be used: " << slog::endl;
        for (size_t i = 0; i < filesToBeUsed; ++i) {
            auto inputInfo = app_inputs_info[i % app_inputs_info.size()].at(input_name);

            slog::info << files.second[i] << " with tensor shape " << getShapeString(inputInfo.tensorShape)
                       << slog::endl;
        }
    }

    for (const auto& files : inputFiles) {
        std::string input_name = files.first.empty() ? app_inputs_info[0].begin()->first : files.first;
        size_t n_shape = 0, m_file = 0;
        while (n_shape < app_inputs_info.size() || m_file < files.second.size()) {
            auto app_info = app_inputs_info[n_shape % app_inputs_info.size()].at(input_name);
            auto precision = app_info.precision;
            size_t inputId = m_file % files.second.size();
            if (app_info.isImage()) {
                // Fill with Images
                cachedBlobs[input_name].push_back(getImageBlob(files.second, inputId, {input_name, app_info}));
            } else {
                if (app_info.isImageInfo() && input_image_sizes.size() == app_inputs_info.size()) {
                    // Most likely it is image info: fill with image information
                    auto image_size = input_image_sizes.at(n_shape % app_inputs_info.size());
                    slog::info << "Fill input '" << input_name << "' with image size " << image_size.first << "x"
                               << image_size.second << slog::endl;
                    cachedBlobs[input_name].push_back(getImInfoBlob(image_size, {input_name, app_info}));
                } else {
                    // Fill with binary files
                    cachedBlobs[input_name].push_back(getBinaryBlob(files.second, inputId, {input_name, app_info}));
                }
            }
            ++n_shape;
            ++m_file;
        }
    }

    if (inputFiles.empty()) {
        slog::warn << "No input files were given: all inputs will be filled with "
                      "random values!"
                   << slog::endl;
        size_t i = 0;
        for (auto& input_info : app_inputs_info) {
            for (auto& input : input_info) {
                if (input.second.isImageInfo() && input_image_sizes.size() == app_inputs_info.size()) {
                    // Most likely it is image info: fill with image information
                    auto image_size = input_image_sizes.at(i);
                    slog::info << "Fill input '" << input.first << "' with image size " << image_size.first << "x"
                               << image_size.second << slog::endl;
                    cachedBlobs[input.first].push_back(getImInfoBlob(image_size, input));
                } else {
                    // Fill random
                    slog::info << "Prepare blob for input '" << input.first << "' with random values ("
                               << std::string((input.second.isImage() ? "image" : "some binary data"))
                               << " is expected)" << slog::endl;
                    cachedBlobs[input.first].push_back(getRandomBlob(input));
                }
                ++i;
            }
        }
    }

    return cachedBlobs;
}

void fillBlob(InferenceEngine::Blob::Ptr& inputBlob, Buffer& data) {
    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    if (!minput) {
        IE_THROW() << "We expect inputBlob to be inherited from MemoryBlob in "
                      "fillBlobRandom, "
                   << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer
    // happens
    auto minputHolder = minput->wmap();
    auto precision = data.precision;
    if (precision == InferenceEngine::Precision::FP32) {
        auto inputBlobData = minputHolder.as<float*>();
        memcpy(inputBlobData, data.get<float>(), data.total_size);
    } else if (precision == InferenceEngine::Precision::FP16) {
        auto inputBlobData = minputHolder.as<short*>();
        memcpy(inputBlobData, data.get<short>(), data.total_size);
    } else if (precision == InferenceEngine::Precision::I32) {
        auto inputBlobData = minputHolder.as<int32_t*>();
        memcpy(inputBlobData, data.get<int32_t>(), data.total_size);
    } else if (precision == InferenceEngine::Precision::I64) {
        auto inputBlobData = minputHolder.as<int64_t*>();
        memcpy(inputBlobData, data.get<int64_t>(), data.total_size);
    } else if (precision == InferenceEngine::Precision::U8 || precision == InferenceEngine::Precision::BOOL) {
        auto inputBlobData = minputHolder.as<uint8_t*>();
        memcpy(inputBlobData, data.get<uint8_t>(), data.total_size);
    } else if (precision == InferenceEngine::Precision::I8) {
        auto inputBlobData = minputHolder.as<int8_t*>();
        memcpy(inputBlobData, data.get<int8_t>(), data.total_size);
    } else if (precision == InferenceEngine::Precision::U16) {
        auto inputBlobData = minputHolder.as<uint16_t*>();
        memcpy(inputBlobData, data.get<uint16_t>(), data.total_size);
    } else if (precision == InferenceEngine::Precision::I16) {
        auto inputBlobData = minputHolder.as<int16_t*>();
        memcpy(inputBlobData, data.get<int16_t>(), data.total_size);
    } else {
        IE_THROW() << "Input precision is not supported: " << precision;
    }
}
