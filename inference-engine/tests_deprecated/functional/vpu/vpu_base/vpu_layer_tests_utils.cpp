// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_layer_tests_utils.hpp"
#include <fstream>
#include "common_layers_params.hpp"

using namespace InferenceEngine;


void PrintTo(const IRVersion& version, std::ostream* os)
{
    IE_ASSERT(version == IRVersion::v7 || version == IRVersion::v10);
    *os << (version == IRVersion::v7 ? "[IR v7]" : "[IR v10]");
}

void PrintTo(const tensor_test_params& sz, std::ostream* os)
{
    *os << "{" << std::setw(2) << sz.n << ", " << std::setw(3) << sz.c << ", "
            << std::setw(3) << sz.h << ", " << std::setw(3) << sz.w << "}";
}

void print_buffer_HWC_fp16(ie_fp16 *src_data, int32_t IW, int32_t IH, int32_t IC, const char * tname, int32_t iw0, int32_t iw1, int32_t ih0, int32_t ih1, int32_t ic0, int32_t ic1 )
{
    iw1 = (iw1 == -1) ? IW-1 : iw1;
    ih1 = (ih1 == -1) ? IH-1 : ih1;
    ic1 = (ic1 == -1) ? IC-1 : ic1;

    printf("%s: H=%i, W=%i, C=%i\n", tname, IH, IW, IC);
    for (int ih = ih0; ih <= ih1; ih++)
    {
        printf("h %i: ", ih);
        for (int iw = iw0; iw <= iw1 ; iw++)
        {
            printf("(");
            for (int ic = ic0; ic <= ic1; ic++)
            {
                printf("%8.4f ", PrecisionUtils::f16tof32(src_data[ic + iw * IC + ih * IC * IW]));
            }
            printf("), ");
        }
        printf("\n");
    }
}

void print_tensor_HWC_fp16(const Blob::Ptr src, const char * tname, int32_t iw0, int32_t iw1, int32_t ih0, int32_t ih1, int32_t ic0, int32_t ic1)
{
    ie_fp16 *src_data = static_cast<ie_fp16*>(src->buffer());

    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    get_dims(src, IW, IH, IC);

    print_buffer_HWC_fp16(src_data, IW, IH, IC, tname, iw0, iw1, ih0, ih1, ic0, ic1);
}

void get_ndims(const InferenceEngine::Blob::Ptr blob,
               int32_t &dimx,
               int32_t &dimy,
               int32_t &dimz,
               int32_t &dimn) {
    ASSERT_NE(blob, nullptr);
    auto dims = blob->getTensorDesc().getDims();

    if (dims.size() == 1) {
        dimn = 1;
        dimz = dims[0];
        dimy = 1;
        dimx = 1;
    }
    else if (dims.size() == 2) {
        dimn = 1;
        dimz = 1;
        dimy = dims[0];
        dimx = dims[1];
    } else if (dims.size() == 3) {
        dimx = dims[2];
        dimy = dims[1];
        dimz = dims[0];
        dimn = 1;
    } else if (dims.size() == 4) {
        dimx = dims[3];
        dimy = dims[2];
        dimz = dims[1];
        dimn = dims[0];
    }
}

void get_dims(const InferenceEngine::Blob::Ptr blob,
                    int32_t &dimx,
                    int32_t &dimy,
                    int32_t &dimz) {
    ASSERT_NE(blob, nullptr);
    CommonTestUtils::get_common_dims(*blob.get(), dimx, dimy, dimz);
}

void get_dims(const InferenceEngine::SizeVector& input_dims,
                    int32_t &IW,
                    int32_t &IH,
                    int32_t &IC) {
    IW = 0;
    IH = 0;
    IC = 0;
    int32_t stub = 0;

    get_dims(input_dims, IW, IH, IC, stub);
}

void get_dims(const InferenceEngine::SizeVector& input_dims,
                    int32_t &IW,
                    int32_t &IH,
                    int32_t &IC,
                    int32_t &I_N) {
    IW = 0;
    IH = 0;
    IC = 0;
    I_N = 1;
    switch (input_dims.size()) {
        case 2:
            /* Fully connected tests */
            IW = 1;
            IC = 1;
            IC = input_dims[1];
            break;
        case 3:
            IW = input_dims[2];
            IH = input_dims[1];
            IC = input_dims[0];
            break;
        case 4:
            IW = input_dims[3];
            IH = input_dims[2];
            IC = input_dims[1];
            I_N = input_dims[0];
            break;
        default:
            FAIL() << "Unsupported input dimension.";
            break;
    }
}

void gen_dims(InferenceEngine::SizeVector& out_dims,
              int32_t dimension,
              int32_t IW,
              int32_t IH,
              int32_t IC) {
    if (dimension < 2 ||
        dimension > 4)
        FAIL() << "Unsupported input dimension:" << dimension;
    out_dims.reserve(dimension);
    switch (dimension) {
        case 4:
            out_dims.push_back(1);
        case 3:
            out_dims.push_back(IC);
            out_dims.push_back(IH);
            out_dims.push_back(IW);
            break;
        default:
            break;
    }
}

void gen_dims(InferenceEngine::SizeVector& out_dims,
              int32_t dimension,
              int32_t IW,
              int32_t IH,
              int32_t IC,
              int32_t I_N) {
    if (dimension < 2 ||
        dimension > 4)
        FAIL() << "Unsupported input dimension:" << dimension;
    out_dims.reserve(dimension);
    switch (dimension) {
        case 4:
            out_dims.push_back(I_N);
        case 3:
            out_dims.push_back(IC);
            out_dims.push_back(IH);
            out_dims.push_back(IW);
            break;
        default:
            break;
    }
}

void zeroWeightsRange(uint16_t* ptr, size_t weightsSize) {
    ASSERT_NE(ptr, nullptr);
    for (size_t count = 0 ; count < weightsSize; ++count) {
        ptr[count] = PrecisionUtils::f32tof16(0.);
    }
}

void defaultWeightsRange(uint16_t* ptr, size_t weightsSize) {
    ASSERT_NE(ptr, nullptr);
    float scale  = 2.0f / RAND_MAX;
    for (size_t count = 0 ; count < weightsSize; ++count) {
        float val = rand();
        val = val * scale - 1.0f;
        ptr[count] = PrecisionUtils::f32tof16(val);
    }
}

void smallWeightsRange(uint16_t* ptr, size_t weightsSize) {
    ASSERT_NE(ptr, nullptr);
    float scale  = 2.0f / RAND_MAX;
    for (size_t count = 0 ; count < weightsSize; ++count) {
        float val = rand();
        val = (val * scale - 1.0f) / 512;
        ptr[count] = PrecisionUtils::f32tof16(val);
    }
}

std::string gen_param(const param_size& in_param) {
    std::string res = std::to_string(in_param.x) + ",";
    res += std::to_string(in_param.y);
    return res;
}

void GenRandomData(InferenceEngine::Blob::Ptr blob)
{
    GenRandomDataCommon(blob);
}

bool fromBinaryFile(std::string input_binary, InferenceEngine::Blob::Ptr blob) {

    std::ifstream in(input_binary, std::ios_base::binary | std::ios_base::ate);

    size_t sizeFile = in.tellg();
    in.seekg(0, std::ios_base::beg);
    size_t count = blob->size();
    bool status = false;
    if(in.good()) {
        if (blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP16) {
            ie_fp16 *blobRawDataFP16 = blob->buffer().as<ie_fp16 *>();
            if(sizeFile == count * sizeof(float)) {
                for (size_t i = 0; i < count; i++) {
                    float tmp;
                    in.read(reinterpret_cast<char *>(&tmp), sizeof(float));
                    blobRawDataFP16[i] = PrecisionUtils::f32tof16(tmp);
                }
                status = true;
            } else if(sizeFile == count * sizeof(ie_fp16)) {
                for (size_t i = 0; i < count; i++) {
                    ie_fp16 tmp;
                    in.read(reinterpret_cast<char *>(&tmp), sizeof(ie_fp16));
                    blobRawDataFP16[i] = tmp;
                }
                status = true;
            }
        }else if (blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
            float *blobRawData = blob->buffer();
            if(sizeFile == count * sizeof(float)) {
                in.read(reinterpret_cast<char *>(blobRawData), count * sizeof(float));
                status = true;
            }
        }
    }
    return status;
}


WeightsBlob* GenWeights(size_t sz, float min_val, float max_val) {
    // TODO: pass seed as parameter

    float scale  = (max_val - min_val) / RAND_MAX;
    WeightsBlob *weights = new WeightsBlob({InferenceEngine::Precision::U8, {(sz) * sizeof(uint16_t)}, InferenceEngine::C});
    weights->allocate();
    uint16_t *inputBlobRawDataFp16 = weights->data().as<uint16_t *>();
    size_t indx = 0;

    for (; indx < sz; ++indx) {
        float val = rand();
        val = val * scale + min_val;
        inputBlobRawDataFp16[indx] = PrecisionUtils::f32tof16(val);
    }
    return weights;
}
