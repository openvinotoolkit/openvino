// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <float.h>

#include <vector>

#include "ie_api.h"

class INFERENCE_ENGINE_API_CLASS(DataStats) {
  public:
    template<typename T>
    static void GetDataMinMax(const T* data, size_t count, T& min, T& max);

    template<typename T>
    static void GetDataAverage(const T* data, size_t count, T& ave);

    template<typename T>
    static void GetDataAbsMax(const T* data, size_t count, T& max);

    template<typename T>
    static T GetAbsMax(T min, T max);
};

template<typename T>
void DataStats::GetDataMinMax(const T* data, size_t count, T& min, T& max) {
    for (size_t i = 0; i < count; i++) {
        T val = data[i];

        if (min > val) {
            min = val;
        }

        if (max < val) {
            max = val;
        }
    }
}

template<typename T>
void DataStats::GetDataAbsMax(const T* data, size_t count, T& max) {
    T min = FLT_MAX;

    GetDataMinMax(data, count, min, max);

    max = GetAbsMax(min, max);
}

template void DataStats::GetDataMinMax<float>(const float* data, size_t count, float& min, float& max);
template void DataStats::GetDataMinMax<uint8_t>(const uint8_t* data, size_t count, uint8_t& min, uint8_t& max);

template void DataStats::GetDataAbsMax<float>(const float* data, size_t count, float& max);

template<typename T>
void DataStats::GetDataAverage(const T* data, size_t count, T& ave) {
    ave = 0;

    for (size_t i = 0; i < count; i++) {
        ave += data[i];
    }

    ave /= count;
}

template void DataStats::GetDataAverage<float>(const float* data, size_t count, float& ave);

template<typename T>
T DataStats::GetAbsMax(T min, T max) {
    if (min < 0) {
        min *= -1;
    }

    if (max < 0) {
        max *= -1;
    }

    return (max > min) ? max : min;
}

template float DataStats::GetAbsMax<float>(float min, float max);
