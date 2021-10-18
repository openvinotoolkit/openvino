// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

// clang-format off
#include "inference_engine.hpp"

#include "infer_request_wrap.hpp"
#include "utils.hpp"
// clang-format on

class Buffer {
public:
    Buffer() : data(nullptr), size(0), total_size(-1) {}
    Buffer(size_t size, const InferenceEngine::Precision precision)
        : data(nullptr),
          precision(precision),
          total_size(size) {
        allocate(size, precision);
    }

    Buffer(const Buffer& buf) : data(nullptr), precision(buf.precision), size(buf.size), total_size(buf.total_size) {
        allocate(buf.size, buf.precision);
        memcpy(data, buf.data, buf.total_size);
    }

    Buffer(Buffer&& buf) noexcept
        : data(buf.data),
          precision(buf.precision),
          size(buf.size),
          total_size(buf.total_size) {
        buf.data = nullptr;
        buf.total_size = -1;
        buf.size = 0;
    }

    void allocate(const size_t _size, const InferenceEngine::Precision _precision) {
        if (data)
            deallocate();

        data = nullptr;
        precision = _precision;
        size = _size;
        total_size = _size * elem_size();

        if (total_size > 0) {
            data = malloc(total_size);
            if (!data)
                throw std::runtime_error("Failed to allocate " + std::to_string(total_size) + " bytes");
        }
    }

    void deallocate() {
        if (!data)
            return;
        free(data);
        data = nullptr;
        total_size = -1;
        size = 0;
    }

    static size_t elem_size(InferenceEngine::Precision precision) {
        if (precision == InferenceEngine::Precision::FP32)
            return sizeof(float);
        else if (precision == InferenceEngine::Precision::FP16)
            return sizeof(uint16_t);
        else if (precision == InferenceEngine::Precision::I32)
            return sizeof(int32_t);
        else if (precision == InferenceEngine::Precision::I64)
            return sizeof(int64_t);
        else if (precision == InferenceEngine::Precision::I8)
            return sizeof(int8_t);
        else if ((precision == InferenceEngine::Precision::U8) || (precision == InferenceEngine::Precision::BOOL))
            return sizeof(uint8_t);
        else if (precision == InferenceEngine::Precision::I16)
            return sizeof(int16_t);
        else if (precision == InferenceEngine::Precision::U16)
            return sizeof(uint16_t);

        throw std::runtime_error("Unsupported precision!");
    }

    size_t elem_size() const {
        return elem_size(precision);
    }

    ~Buffer() {
        deallocate();
    }

    template <typename T>
    T* get() {
        return (T*)data;
    }

    void* data;
    size_t size;
    InferenceEngine::Precision precision;
    size_t total_size;
};

void fillBlobs(const std::vector<std::string>& inputFiles,
               const size_t& batchSize,
               benchmark_app::InputsInfo& app_inputs_info,
               std::vector<InferReqWrap::Ptr> requests,
               bool supress = false);

std::vector<Buffer> prepareRandomInputs(std::vector<benchmark_app::InputsInfo>& app_inputs_info);

void fillBlob(InferenceEngine::Blob::Ptr& inputBlob, Buffer& data);
