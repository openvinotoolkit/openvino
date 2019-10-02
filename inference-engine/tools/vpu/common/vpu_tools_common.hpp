//
// Copyright (C) 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include <string>
#include <map>
#include <memory>
#include <vector>

#include "inference_engine.hpp"

InferenceEngine::CNNNetwork readNetwork(const std::string &xmlFileName);

/* Set all precisions to FP16 */
void setPrecisions(const InferenceEngine::CNNNetwork &network);

std::map<std::string, std::string> parseConfig(const std::string &configName, char comment = '#');

class BitMap {
private:
    typedef struct {
        unsigned short type;                /* Magic identifier            */
        unsigned int size;                  /* File size in bytes          */
        unsigned int reserved;
        unsigned int offset;                /* Offset to image data, bytes */
    } BmpHeader;

    typedef struct {
        unsigned int size;                  /* Header size in bytes      */
        int width, height;                  /* Width and height of image */
        unsigned short planes;              /* Number of colour planes   */
        unsigned short bits;                /* Bits per pixel            */
        unsigned int compression;           /* Compression type          */
        unsigned int imagesize;             /* Image size in bytes       */
        int xresolution, yresolution;       /* Pixels per meter          */
        unsigned int ncolours;              /* Number of colours         */
        unsigned int importantcolours;      /* Important colours         */
    } BmpInfoHeader;

public:
    explicit BitMap(const std::string &filename);

    ~BitMap() = default;

    size_t _height = 0;
    size_t _width = 0;
    std::shared_ptr<unsigned char> _data;

public:
    size_t size() const { return _width * _height * 3; }
    size_t width() const { return _width; }
    size_t height() const { return _height; }

    std::shared_ptr<unsigned char> getData() {
        return _data;
    }
};

void loadImage(const std::string &imageFilename, InferenceEngine::Blob::Ptr &blob);

void printPerformanceCounts(const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap, const std::string report = "per_layer");

std::vector<std::string> extractFilesByExtension(const std::string& directory, const std::string& extension);
std::vector<std::string> extractFilesByExtension(const std::string& directory, const std::string& extension,
                                                 std::size_t max_size);

void loadBinaryTensor(const std::string &binaryFileName, InferenceEngine::Blob::Ptr& blob);
