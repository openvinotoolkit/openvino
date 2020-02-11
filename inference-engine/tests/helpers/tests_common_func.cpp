// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <utility>
#include <string>
#include <vector>

#include "tests_common_func.hpp"

using namespace InferenceEngine;

inline void bswap_32(char* ptr, size_t size) {
    char* end = ptr + size;
    char tmp;
    for (; ptr < end; ptr+=4) {
        tmp = ptr[0]; ptr[0] = ptr[3]; ptr[3] = tmp;
        tmp = ptr[1]; ptr[1] = ptr[2]; ptr[2] = tmp;
    }
}

InferenceEngine::Blob::Ptr readBMP(std::string path, unsigned batch) {

    std::ifstream input(path, std::ios::binary);
    if (!input) return nullptr;

    unsigned char bmpFileHeader[14];
    input.read((char*)bmpFileHeader, sizeof(bmpFileHeader));
    if(bmpFileHeader[0]!='B' || bmpFileHeader[1]!='M') return nullptr;
    if(bmpFileHeader[11]!=0  || bmpFileHeader[12]!=0 || bmpFileHeader[13]!=0 ) return nullptr;

    unsigned char bmpInfoHeader[40];
    input.read((char*)bmpInfoHeader, sizeof(bmpInfoHeader));
    if(bmpInfoHeader[14]!=24) return nullptr; // bits per pixel
    if(bmpInfoHeader[16]!=0) return nullptr; // compression is not supported

    bool  rowsReversed = (*(int32_t*)(bmpInfoHeader + 8)) < 0;
    uint32_t width  = *(int32_t*)(bmpInfoHeader + 4);
    uint32_t height = abs(*(int32_t*)(bmpInfoHeader + 8));

    size_t padSize = width & 3;
    char pad[3];

    InferenceEngine::Blob::Ptr blob(new InferenceEngine::TBlob<float>(
        {InferenceEngine::Precision::FP32, {batch, 3, height, width}, InferenceEngine::Layout::NCHW}));
    blob->allocate();
    float *blob_ptr = (float*)(void*)blob->buffer();

    unsigned int offset = *(unsigned int *)(bmpFileHeader + 10);
    for (int b = 0; b < batch; b++) {
        int b_off = 3*width*height*b;
        input.seekg(offset, std::ios::beg);
        //reading by rows in invert vertically
        for (uint32_t i = 0; i < height; i++) {
            int storeAt = rowsReversed ? i : height - 1 - i;

            for (uint32_t j = 0; j < width; j++) {
                unsigned char RGBA[3];
                input.read((char *) RGBA, sizeof(RGBA));

                blob_ptr[b_off + j + storeAt * width] = RGBA[0];
                blob_ptr[b_off + j + storeAt * width + height * width * 1] = RGBA[1];
                blob_ptr[b_off + j + storeAt * width + height * width * 2] = RGBA[2];
            }
            input.read(pad, padSize);
        }
    }

    return blob;
}

InferenceEngine::Blob::Ptr readUbyte(std::string path, unsigned batch) {

    std::ifstream input(path, std::ios::binary);
    struct {
        uint32_t magic_number;
        uint32_t n_images;
        uint32_t n_rows;
        uint32_t n_cols;
    } hdr;

    input.read((char *) &hdr, sizeof(hdr));
    bswap_32((char *) &hdr, sizeof(hdr));
    if (hdr.magic_number != 2051) return nullptr; // Invalid MNIST image file

    InferenceEngine::Blob::Ptr blob(new InferenceEngine::TBlob<float>({InferenceEngine::Precision::FP32,
                                                                       {batch, hdr.n_images, hdr.n_rows, hdr.n_cols},
                                                                      InferenceEngine::NCHW}));
    blob->allocate();
    float *blob_ptr = (float*)(void*)blob->buffer();
    for (int b = 0; b < batch; b++) {
        input.seekg(sizeof(hdr), std::ios::beg);
        int b_off = b*hdr.n_images*hdr.n_rows*hdr.n_cols;
        for (uint32_t i = 0; i < hdr.n_images; ++i) {
            for (uint32_t r = 0; r < hdr.n_rows; ++r) {
                for (uint32_t c = 0; c < hdr.n_cols; ++c) {
                    unsigned char temp = 0;
                    input.read((char *) &temp, sizeof(temp));
                    blob_ptr[b_off + i * hdr.n_rows * hdr.n_cols + r * hdr.n_cols + c] = temp;
                }
            }
        }
    }
    return blob;
}

InferenceEngine::Blob::Ptr TestsCommonFunc::readInput(std::string path, int batch) {
    if ( path.substr(path.rfind('.') + 1) == "bmp" ) return readBMP(path, batch);
    if ( path.substr(path.rfind('-') + 1) == "ubyte" ) return readUbyte(path, batch);
    return nullptr;
}

bool compareTopLikeObjDetection (
    InferenceEngine::Blob& blob,
    std::vector<std::pair<int, float>> &ref_top,
    int batch_to_compare = 0,
    const bool compareRawValues = true) {
    assert(blob.getTensorDesc().getDims().back() == 7);

    const int box_info_size = 7;

    int top_num = (int)ref_top.size();
    float *data_ptr = blob.buffer().as<float*>();
    const int data_size = blob.size();
    if (data_size/box_info_size < top_num) {
        EXPECT_TRUE(data_size/box_info_size >= top_num) << "Dst blob contains less data then expected";
        return false;
    }

#ifdef DISPLAY_RESULTS
    std::cout << "actual:" << std::endl;
    for (int i = 0; i < top_num; i++) {
        std::cout << "{" << data_ptr[i*box_info_size + 1] << ", " << data_ptr[i*box_info_size + 2] << "}" << std::endl;
    }

    std::cout << "reference:" << std::endl;
    for (int i = 0; i < top_num; i++) {
        std::cout << "{" << ref_top[i].first << ", " << ref_top[i].second << "}" << std::endl;
    }
#endif

    for (int i=0; i<top_num; i++) {
        int label = data_ptr[i*box_info_size + 1];
        float confidence = data_ptr[i*box_info_size + 2];

        if (label != ref_top[i].first) {
            EXPECT_EQ(label, ref_top[i].first) << "Label mismatch";
            return false;
        }

        if (compareRawValues) {
            if (fabs(confidence - ref_top[i].second) / ref_top[i].second > 0.005) {
                EXPECT_NEAR(confidence, ref_top[i].second, ref_top[i].second * 0.005);
                return false;
            }
        }
    }

    return true;
}

bool compareTopLikeClassification(
    InferenceEngine::Blob& blob,
    std::vector<std::pair<int, float>> &ref_top,
    int batch_to_compare = 0,
    float threshold = 0.005f,
    const size_t classesCanBeChangedIndex = 9999,
    const bool compareRawValues = true) {
    int top_num = (int)ref_top.size();

    size_t data_size = blob.size();
    float *data_ptr = (float*)(void*)blob.buffer();

    int batch_size = blob.getTensorDesc().getDims()[0];
    assert(batch_size > batch_to_compare);

    const std::vector<size_t> dims = blob.getTensorDesc().getDims();
    if ((dims.size() != 2ul) || (dims[1] != 1ul)) {
        data_size /= batch_size;
    }
    data_ptr += data_size*batch_to_compare;

    std::vector<int> top(data_size);

    for (size_t i = 0; i < data_size; i++) top[i] = (int)i;
    std::partial_sort (top.begin(), top.begin()+top_num, top.end(),
                       [&](int l, int r) -> bool { return data_ptr[l] > data_ptr[r]; } );

#ifdef DISPLAY_RESULTS
    std::cout << "actual:" << std::endl;
    for (int i = 0; i < top_num; i++) {
        std::cout << "{" << top[i] << ", " << data_ptr[top[i]] << "}" << std::endl;
    }

    std::cout << "reference:" << std::endl;
    for (int i = 0; i < top_num; i++) {
        std::cout << "{" << ref_top[i].first << ", " << ref_top[i].second << "}" << std::endl;
    }
#endif

    for (int i = 0 ; i < top_num; i++) {
        if (top[i] != ref_top[i].first) {
            if (i >= classesCanBeChangedIndex) {
                bool wasFound = false;
                for (int refIndex = 0; refIndex < top_num; refIndex++) {
                    if (top[i] == ref_top[refIndex].first) {
                        wasFound = true;
                        break;
                    }
                }

                if (!wasFound) {
                    EXPECT_EQ(top[i], ref_top[i].first) << "class is different for element " << i << ": " << top[i] << ", reference: " << ref_top[i].first;
                    return false;
                }
            } else {
                EXPECT_EQ(top[i], ref_top[i].first) << "class is different for element " << i << ": " << top[i] << ", reference: " << ref_top[i].first;
                return false;
            }
        }

        if (compareRawValues && (fabs(data_ptr[top[i]] - ref_top[i].second)/ref_top[i].second > threshold)) {
            EXPECT_NEAR(data_ptr[top[i]] , ref_top[i].second , ref_top[i].second * threshold);
            return false;
        }
    }
    return true;
}

bool TestsCommonFunc::compareTop(
    InferenceEngine::Blob& blob,
    std::vector<std::pair<int, float>> &ref_top,
    int batch_to_compare,
    float threshold,
    const size_t classesCanBeChangedIndex,
    const bool compareRawValues) {
    if (blob.getTensorDesc().getDims().back() == 7)
        return compareTopLikeObjDetection(blob, ref_top, batch_to_compare, compareRawValues);
    else
        return compareTopLikeClassification(blob, ref_top, batch_to_compare, threshold, classesCanBeChangedIndex, compareRawValues);
}
