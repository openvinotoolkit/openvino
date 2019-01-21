// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <ie_blob.h>
#include <ie_layers_property.hpp>
#include <ie_precision.hpp>
#include <inference_engine/precision_utils.h>
#include <gtest/gtest.h>
#include "single_layer_common.hpp"
#include <math.h>

using namespace InferenceEngine;

void get_common_dims(const Blob &blob,
                     int32_t &dimx,
                     int32_t &dimy,
                     int32_t &dimz) {
    if (blob.dims().size() == 2) {
        dimz = 1;
        dimy = blob.dims()[1];
        dimx = blob.dims()[0];
    } else if (blob.dims().size() == 3 || (blob.dims().size() == 4 && blob.dims()[3] == 1)) {
        dimx = blob.dims()[0];
        dimy = blob.dims()[1];
        dimz = blob.dims()[2];
    }
}

void get_common_dims(const Blob &blob,
                     int32_t &dimx,
                     int32_t &dimy,
                     int32_t &dimz,
                     int32_t &dimn) {
    dimn = 1;
    if (blob.dims().size() == 2) {
        dimz = 1;
        dimy = blob.dims()[1];
        dimx = blob.dims()[0];
    } else if (blob.dims().size() == 3 || (blob.dims().size() == 4 && blob.dims()[3] == 1)) {
        dimx = blob.dims()[0];
        dimy = blob.dims()[1];
        dimz = blob.dims()[2];
    } else {
        if (blob.dims().size() == 4 && blob.dims()[3] != 1) {
            dimx = blob.dims()[0];
            dimy = blob.dims()[1];
            dimz = blob.dims()[2];
            dimn = blob.dims()[3];
        }
    }
}

void GenRandomDataCommon(Blob::Ptr blob) {
    if (blob->precision() == Precision::U8) {
        auto * blobRawDataU8 = blob->buffer().as<uint8_t*>();
        size_t count = blob->size();
        for (size_t i = 0; i < count; i++) {
            auto val = static_cast<uint8_t>(rand() % 256);
            blobRawDataU8[i] = val;
        }
    } else if (blob->precision() == Precision::FP16) {
        float scale = 2.0f / RAND_MAX;
        /* fill by random data in the range (-1, 1)*/
        auto * blobRawDataFp16 = blob->buffer().as<ie_fp16 *>();
        size_t count = blob->size();
        for (size_t indx = 0; indx < count; ++indx) {
            float val = rand();
            val = val * scale - 1.0f;
            blobRawDataFp16[indx] = PrecisionUtils::f32tof16(val);
        }
    } else if (blob->precision() == Precision::FP32) {
        float scale = 2.0f / RAND_MAX;
        /* fill by random data in the range (-1, 1)*/
        auto * blobRawDataFp16 = blob->buffer().as<float*>();
        size_t count = blob->size();
        for (size_t i = 0; i < count; i++) {
            float val = rand();
            val = val * scale - 1.0f;
            blobRawDataFp16[i] = val;
        }
    }
}

BufferWrapper::BufferWrapper(const Blob::Ptr& blob) : BufferWrapper(blob, blob->precision()) {}

BufferWrapper::BufferWrapper(const Blob::Ptr& blob, Precision _precision) : precision(_precision) {
    if (precision == Precision::FP16) {
        fp16_ptr = blob->buffer().as<ie_fp16*>();
    } else if (precision == Precision::FP32) {
        fp32_ptr = blob->buffer().as<float*>();
    } else {
        THROW_IE_EXCEPTION << "Unsupported precision for compare: " << precision;
    }
}

float BufferWrapper::operator[](size_t index) {
    if (precision == Precision::FP16) return PrecisionUtils::f16tof32(fp16_ptr[index]);
    return fp32_ptr[index];
}

void BufferWrapper::insert(size_t index, float value) {
    if (precision == Precision::FP16) {
        fp16_ptr[index] = PrecisionUtils::f32tof16(value);
    } else {
        fp32_ptr[index] = value;
    }
}

void CompareCommon(const Blob::Ptr& actual, const Blob::Ptr& expected, float tolerance) {
    ASSERT_NE(actual, nullptr);
    ASSERT_NE(expected, nullptr);

    Layout res_layout = actual->layout();
    Layout ref_layout = expected->layout();
    SizeVector res_dims = actual->getTensorDesc().getDims();

    BufferWrapper res_ptr(actual);
    BufferWrapper ref_ptr(expected);

    size_t res_size = actual->size();
    size_t ref_size = expected->size();
    ASSERT_EQ(res_size, ref_size);

    float max_error = 0;
    size_t actualMaxErrId = 0;
    size_t expectedMaxErrId = 0;

    if (res_layout == NCHW || res_layout == NHWC) {
        size_t N = res_dims[0];
        size_t C = res_dims[1];
        size_t H = res_dims[2];
        size_t W = res_dims[3];

        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        size_t actualIdx = res_layout == NCHW ?
                                           w + h * W + c * W * H + n * W * H * C : c + w * C + h * C * W +
                                                                                   n * W * H * C;
                        size_t expectedIdx = ref_layout == NCHW ?
                                             w + h * W + c * W * H + n * W * H * C : c + w * C + h * C * W +
                                                                                     n * C * W * H;
                        float cur_diff = fabs(res_ptr[actualIdx] - ref_ptr[expectedIdx]);
                        if (cur_diff > max_error) {
                            max_error = cur_diff;
                            actualMaxErrId = actualIdx;
                            expectedMaxErrId = expectedIdx;
                        }
                    }
                }
            }
        }
    } else {
        if (res_layout == NC) {

            size_t N = res_dims[0];
            size_t C = res_dims[1];
            for (size_t n = 0; n < N; n++) {
                for (size_t c = 0; c < C; c++) {
                    size_t actualIdx =   c +  n * C;
                    float cur_diff = fabs(res_ptr[actualIdx] - ref_ptr[actualIdx]);
                    if (cur_diff > max_error) {
                        max_error = cur_diff;
                        actualMaxErrId = actualIdx;
                        expectedMaxErrId = actualIdx;
                    }
                }
            }
        } else {
            for (size_t i = 0; i < ref_size; i++) {
                float cur_diff = fabs(res_ptr[i] - ref_ptr[i]);
                if (cur_diff > max_error) {
                    max_error = cur_diff;
                    actualMaxErrId = expectedMaxErrId = i;
                }
            }
        }
    }

    ASSERT_NEAR(ref_ptr[expectedMaxErrId], res_ptr[actualMaxErrId], tolerance)
                                << "expectedMaxErrId = " << expectedMaxErrId
                                << " actualMaxErrId = " << actualMaxErrId;
}

void fill_data_common(BufferWrapper& data, size_t size, size_t duty_ratio) {
    for (size_t i = 0; i < size; i++) {
        if ((i / duty_ratio) % 2 == 1) {
            data.insert(i, 0.0);
        } else {
            data.insert(i, sin((float) i));
        }
    }
}
