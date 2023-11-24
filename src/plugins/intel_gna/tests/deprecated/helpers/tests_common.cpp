// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <vector>

#include "tests_common.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"

#ifdef _WIN32
# ifndef NOMINMAX
#  define NOMINMAX
# endif
# ifndef _WINSOCKAPI_
#  define _WINSOCKAPI_
# endif
# ifndef _WINSOCK2API_
#  define _WINSOCK2API_
# endif
# include <winsock2.h>
# include <windows.h>
# include "psapi.h"
#endif

static size_t parseLine(char* line) {
    // This assumes that a digit will be found and the line ends in " Kb".
    size_t i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = (size_t)atoi(p);
    return i;
}

static size_t getVmSizeInKB(){
    FILE* file = fopen("/proc/self/status", "r");
    size_t result = 0;
    if (file != nullptr) {
        char line[128];

        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, "VmSize:", 7) == 0) {
                result = parseLine(line);
                break;
            }
        }
        fclose(file);
    }
    return result;
}

#ifdef _WIN32
static size_t getVmSizeInKBWin() {
        PROCESS_MEMORY_COUNTERS pmc;
        pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
        GetProcessMemoryInfo(GetCurrentProcess(),&pmc, pmc.cb);
        return pmc.WorkingSetSize;
}
#endif

void TestsCommon::SetUp() {
    auto memsize = getVmSizeInKB();
    if (memsize != 0) {
        std::cout << "\nMEM_USAGE=" << getVmSizeInKB() << "KB\n";
    }
    ov::threading::executor_manager()->clear();
}

void TestsCommon::TearDown() {
    ov::threading::executor_manager()->clear();
}

/**
 * @brief Copies a 8-bit RGB image to the blob.
 *
 * Throws an exception in case of dimensions or input size mismatch
 *
 * @tparam data_t Type of the target blob
 * @param RGB8 8-bit RGB image
 * @param RGB8_size Size of the image
 * @param blob Target blob to write image to
 */
template <typename data_t>
void copyFromRGB8(uint8_t* RGB8, size_t RGB8_size, InferenceEngine::TBlob<data_t>* blob) {
    InferenceEngine::SizeVector dims = blob->getTensorDesc().getDims();
    if (4 != dims.size())
        IE_THROW() << "Cannot write data to input blob! Blob has incorrect dimensions size " << dims.size();
    size_t num_channels = dims[1];  // because RGB
    size_t num_images = dims[0];
    size_t w = dims[3];
    size_t h = dims[2];
    size_t nPixels = w * h;

    if (RGB8_size != w * h * num_channels * num_images)
        IE_THROW() << "input pixels mismatch, expecting " << w * h * num_channels * num_images
                           << " bytes, got: " << RGB8_size;

    std::vector<data_t*> dataArray;
    for (unsigned int n = 0; n < num_images; n++) {
        for (unsigned int i = 0; i < num_channels; i++) {
            if (!n && !i && dataArray.empty()) {
                dataArray.push_back(blob->data());
            } else {
                dataArray.push_back(dataArray.at(n * num_channels + i - 1) + nPixels);
            }
        }
    }
    for (size_t n = 0; n < num_images; n++) {
        size_t n_num_channels = n * num_channels;
        size_t n_num_channels_nPixels = n_num_channels * nPixels;
        for (size_t i = 0; i < nPixels; i++) {
            size_t i_num_channels = i * num_channels + n_num_channels_nPixels;
            for (size_t j = 0; j < num_channels; j++) {
                dataArray.at(n_num_channels + j)[i] = RGB8[i_num_channels + j];
            }
        }
    }
}

/**
 * @brief Splits the RGB channels to either I16 Blob or float blob.
 *
 * The image buffer is assumed to be packed with no support for strides.
 *
 * @param imgBufRGB8 Packed 24bit RGB image (3 bytes per pixel: R-G-B)
 * @param lengthbytesSize Size in bytes of the RGB image. It is equal to amount of pixels times 3 (number of channels)
 * @param input Blob to contain the split image (to 3 channels)
 */
void ConvertImageToInput(unsigned char* imgBufRGB8, size_t lengthbytesSize, InferenceEngine::Blob& input) {
   InferenceEngine::TBlob<float>* float_input = dynamic_cast<InferenceEngine::TBlob<float>*>(&input);
    if (float_input != nullptr)
        copyFromRGB8(imgBufRGB8, lengthbytesSize, float_input);

    InferenceEngine::TBlob<short>* short_input = dynamic_cast<InferenceEngine::TBlob<short>*>(&input);
    if (short_input != nullptr)
        copyFromRGB8(imgBufRGB8, lengthbytesSize, short_input);

    InferenceEngine::TBlob<uint8_t>* byte_input = dynamic_cast<InferenceEngine::TBlob<uint8_t>*>(&input);
    if (byte_input != nullptr)
        copyFromRGB8(imgBufRGB8, lengthbytesSize, byte_input);
}

