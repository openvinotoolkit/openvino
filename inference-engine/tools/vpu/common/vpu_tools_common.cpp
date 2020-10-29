// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/* on windows min and max already defined that makes using numeric_limits impossible */
#if defined(WIN32)
#define NOMINMAX
#endif

#include <sys/stat.h>
#include <os/windows/w_dirent.h>

#include <algorithm>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>
#include <limits>

#include "vpu_tools_common.hpp"
#include <vpu/utils/string.hpp>
#include "samples/common.hpp"

#include "precision_utils.h"

InferenceEngine::CNNNetwork readNetwork(const std::string &xmlFileName) {
    std::string binFileName = fileNameNoExt(xmlFileName) + ".bin";

    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::CNNNetReader reader;
    reader.ReadNetwork(xmlFileName);
    reader.ReadWeights(binFileName);

    return reader.getNetwork();
    IE_SUPPRESS_DEPRECATED_END
}

bool isFP16(InferenceEngine::Precision precision) {
    return precision == InferenceEngine::Precision::FP16;
}

bool isFP32(InferenceEngine::Precision precision) {
    return precision == InferenceEngine::Precision::FP32;
}

bool isU8(InferenceEngine::Precision precision) {
    return precision == InferenceEngine::Precision::U8;
}

bool isFloat(InferenceEngine::Precision precision) {
    return isFP16(precision) || isFP32(precision);
}

void setPrecisions(const InferenceEngine::CNNNetwork &network) {
    for (auto &&layer : network.getInputsInfo()) {
        if (isFP32(layer.second->getPrecision())) {
            layer.second->setPrecision(InferenceEngine::Precision::FP16);
        }
    }

    for (auto &&layer : network.getOutputsInfo()) {
        if (isFP32(layer.second->getPrecision())) {
            layer.second->setPrecision(InferenceEngine::Precision::FP16);
        }
    }
}

std::map<std::string, std::string> parseConfig(const std::string &configName, char comment) {
    std::map<std::string, std::string> config = {};

    std::ifstream file(configName);
    if (!file.is_open()) {
        return config;
    }

    std::string key, value;
    while (file >> key >> value) {
        if (key.empty() || key[0] == comment) {
            continue;
        }
        config[key] = value;
    }

    return config;
}

BitMap::BitMap(const std::string &filename) {
    BmpHeader header;
    BmpInfoHeader infoHeader;

    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        return;
    }

    input.read(reinterpret_cast<char *>(&header.type), 2);

    if (header.type != 'M'*256+'B') {
        std::cerr << "[BMP] file is not bmp type\n";
        return;
    }

    input.read(reinterpret_cast<char *>(&header.size), 4);
    input.read(reinterpret_cast<char *>(&header.reserved), 4);
    input.read(reinterpret_cast<char *>(&header.offset), 4);

    input.read(reinterpret_cast<char *>(&infoHeader), sizeof(BmpInfoHeader));

    bool rowsReversed = infoHeader.height < 0;
    _width  = static_cast<std::size_t>(infoHeader.width);
    _height = static_cast<std::size_t>(std::abs(infoHeader.height));

    if (infoHeader.bits != 24) {
        std::cerr << "[BMP] 24bpp only supported. But input has:" << infoHeader.bits << "\n";
        return;
    }

    if (infoHeader.compression != 0) {
        std::cerr << "[BMP] compression not supported\n";
    }

    auto padSize = _width & 3;
    char pad[3];
    size_t size = _width * _height * 3;

    _data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());

    input.seekg(header.offset, std::ios::beg);

    // reading by rows in invert vertically
    for (uint32_t i = 0; i < _height; i++) {
        uint32_t storeAt = rowsReversed ? i : (uint32_t)_height - 1 - i;
        input.read(reinterpret_cast<char *>(_data.get()) + _width * 3 * storeAt, _width * 3);
        input.read(pad, padSize);
    }
}

void loadImage(const std::string &imageFilename, InferenceEngine::Blob::Ptr &blob) {
    InferenceEngine::TensorDesc tensDesc = blob->getTensorDesc();
    if (tensDesc.getPrecision() != InferenceEngine::Precision::FP16) {
        throw std::invalid_argument("Input must have FP16 precision");
    }

    BitMap reader(imageFilename);

    const auto dims = tensDesc.getDims();
    auto numBlobChannels = dims[1];
    size_t batch = dims[0];
    size_t w = dims[3];
    size_t h = dims[2];
    size_t img_w = reader.width();
    size_t img_h = reader.height();

    size_t numImageChannels = reader.size() / (reader.width() * reader.height());
    if (numBlobChannels != numImageChannels && numBlobChannels != 1) {
        throw std::invalid_argument("Input channels mismatch: image channels " + std::to_string(numImageChannels) +
                                    ", network channels " + std::to_string(numBlobChannels) +
                                    ", expecting count of image channels are equal to count if network channels"
                                    "or count of network channels are equal to 1");
    }

    int16_t *blobDataPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<int16_t>>(blob)->data();
    auto nPixels = w * h;
    unsigned char *RGB8 = reader.getData().get();
    float xscale = 1.0f * img_w / w;
    float yscale = 1.0f * img_h / h;

    for (std::size_t n = 0; n != batch; n++) {
        for (std::size_t i = 0; i < h; ++i) {
            int y = static_cast<int>(std::floor((i + 0.5f) * yscale));
            for (std::size_t j = 0; j < w; ++j) {
                int x = static_cast<int>(std::floor((j + 0.5f) * xscale));
                for (std::size_t k = 0; k < numBlobChannels; k++) {
                    float src = 1.0f * RGB8[(y * img_w + x) * numImageChannels + k];
                    if (tensDesc.getLayout() == InferenceEngine::NHWC) {
                        blobDataPtr[n * h * w * numBlobChannels + (i * w + j) * numBlobChannels + k] =
                            InferenceEngine::PrecisionUtils::f32tof16(src);
                    } else {
                        blobDataPtr[n * h * w * numBlobChannels + (i * w + j) + k * nPixels] =
                            InferenceEngine::PrecisionUtils::f32tof16(src);
                    }
                }
            }
        }
    }
}

void printPerformanceCounts(const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap, const std::string report) {
    std::vector<std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>> perfVec(perfMap.begin(),
                                                                                             perfMap.end());
    std::sort(perfVec.begin(), perfVec.end(),
        [=](const std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo> &pair1,
          const std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo> &pair2) -> bool {
          return pair1.second.execution_index < pair2.second.execution_index;
        });

    size_t maxLayerName = 0u, maxExecType = 0u;
    for (auto &&entry : perfVec) {
        maxLayerName = std::max(maxLayerName, entry.first.length());
        maxExecType = std::max(maxExecType, std::strlen(entry.second.exec_type));
    }

    size_t indexWidth = 7, nameWidth = maxLayerName + 5, typeWidth = maxExecType + 5, timeWidth = 10;
    size_t totalWidth = indexWidth + nameWidth + typeWidth + timeWidth;

    std::cout << std::endl << "Detailed " << report << " Profile" << std::endl;
    for (size_t i = 0; i < totalWidth; i++)
        std::cout << "=";
    std::cout << std::endl;
    std::cout << std::setw(static_cast<int>(indexWidth)) << std::left << "Index"
              << std::setw(static_cast<int>(nameWidth)) << std::left << "Name"
              << std::setw(static_cast<int>(typeWidth)) << std::left << "Type"
              << std::setw(static_cast<int>(timeWidth)) << std::right << "Time (ms)"
              << std::endl;

    for (size_t i = 0; i < totalWidth; i++)
        std::cout << "-";
    std::cout << std::endl;

    long long totalTime = 0;
    for (const auto& p : perfVec) {
        const auto& stageName = p.first;
        const auto& info = p.second;
        if (info.status == InferenceEngine::InferenceEngineProfileInfo::EXECUTED) {
            std::cout << std::setw(static_cast<int>(indexWidth)) << std::left << info.execution_index
                      << std::setw(static_cast<int>(nameWidth))  << std::left << stageName
                      << std::setw(static_cast<int>(typeWidth))  << std::left << info.exec_type
                      << std::setw(static_cast<int>(timeWidth))  << std::right << info.realTime_uSec / 1000.0
                      << std::endl;

            totalTime += info.realTime_uSec;
        }
    }

    for (std::size_t i = 0; i < totalWidth; i++)
        std::cout << "-";
    std::cout << std::endl;
    std::cout << std::setw(static_cast<int>(totalWidth / 2)) << std::right << "Total inference time:"
              << std::setw(static_cast<int>(totalWidth / 2 + 1)) << std::right << totalTime / 1000.0
              << std::endl;
    for (std::size_t i = 0; i < totalWidth; i++)
        std::cout << "-";
    std::cout << std::endl;
}

std::vector<std::string> extractFilesByExtension(const std::string& directory, const std::string& extension) {
    return extractFilesByExtension(directory, extension, std::numeric_limits<std::size_t>::max());
}

std::vector<std::string> extractFilesByExtension(const std::string& directory, const std::string& extension,
                                                 std::size_t max_size) {
    if (max_size == 0) {
        return {};
    }

    std::vector<std::string> files;

    DIR* dir = opendir(directory.c_str());
    if (!dir) {
        throw std::invalid_argument("Can not open " + directory);
    }

    auto getExtension = [](const std::string& name) {
        auto extensionPosition = name.rfind('.', name.size());
        return extensionPosition == std::string::npos ? "" : name.substr(extensionPosition + 1, name.size() - 1);
    };

    dirent* ent = nullptr;
    while ((ent = readdir(dir)) && files.size() < max_size) {
        std::string file_name = ent->d_name;
        if (getExtension(file_name) != extension) {
            continue;
        }

        std::string full_file_name = directory + "/" + file_name;

        struct stat st = {};
        if (stat(full_file_name.c_str(), &st) != 0) {
            continue;
        }

        bool is_directory = (st.st_mode & S_IFDIR) != 0;
        if (is_directory) {
            continue;
        }

        files.emplace_back(full_file_name);
    }

    closedir(dir);

    return files;
}

void loadBinaryTensor(const std::string &binaryFileName, InferenceEngine::Blob::Ptr& blob) {
    InferenceEngine::TensorDesc tensDesc = blob->getTensorDesc();
    if (tensDesc.getPrecision() != InferenceEngine::Precision::FP16) {
        throw std::invalid_argument("Input must have FP16 precision");
    }

    std::ifstream binaryFile(binaryFileName, std::ios_base::binary | std::ios_base::ate);
    if (!binaryFile) {
        throw std::invalid_argument("Can not open \"" + binaryFileName + "\"");
    }

    auto fileSize = static_cast<std::size_t>(binaryFile.tellg());
    binaryFile.seekg(0, std::ios_base::beg);
    if (!binaryFile.good()) {
        throw std::invalid_argument("Can not read \"" + binaryFileName + "\"");
    }

    auto expected_size = blob->byteSize();
    if (fileSize != expected_size) {
        throw std::invalid_argument("File \"" + binaryFileName + "\" contains " + std::to_string(fileSize) + " bytes "
                                    "but network expects " + std::to_string(expected_size));
    }
    /* try to read 32 bits data */
    std::int16_t *blobDataPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<std::int16_t>>(blob)->data();
    for (std::size_t i = 0; i < blob->size(); i++) {
        float tmp = 0.f;
        binaryFile.read(reinterpret_cast<char *>(&tmp), sizeof(float));
        blobDataPtr[i] = InferenceEngine::PrecisionUtils::f32tof16(tmp);
    }
}
