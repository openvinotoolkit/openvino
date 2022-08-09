// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_property.h"

#include "common.h"

const std::map<ov_performance_mode_e, ov::hint::PerformanceMode> performance_mode_map = {
    {ov_performance_mode_e::UNDEFINED_MODE, ov::hint::PerformanceMode::UNDEFINED},
    {ov_performance_mode_e::THROUGHPUT, ov::hint::PerformanceMode::THROUGHPUT},
    {ov_performance_mode_e::LATENCY, ov::hint::PerformanceMode::LATENCY},
    {ov_performance_mode_e::CUMULATIVE_THROUGHPUT, ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT}};

ov_status_e ov_property_create(ov_property_t** property) {
    if (!property) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_property_t> _property(new ov_property_t);
        *property = _property.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_property_free(ov_property_t* property) {
    if (property)
        delete property;
}

ov_status_e ov_property_put(ov_property_t* property, ov_property_key_e key, ov_property_value_t* value) {
    if (!property || !value) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        switch (key) {
        case ov_property_key_e::PERFORMANCE_HINT_NUM_REQUESTS: {
            uint32_t v = *(static_cast<uint32_t*>(value->ptr));
            property->object.emplace(ov::hint::num_requests(v));
            break;
        }
        case ov_property_key_e::NUM_STREAMS: {
            uint32_t v = *(static_cast<uint32_t*>(value->ptr));
            property->object.emplace(ov::num_streams(v));
            break;
        }
        case ov_property_key_e::PERFORMANCE_HINT: {
            ov_performance_mode_e m = *(static_cast<ov_performance_mode_e*>(value->ptr));
            if (m > ov_performance_mode_e::CUMULATIVE_THROUGHPUT) {
                return ov_status_e::INVALID_C_PARAM;
            }
            auto v = performance_mode_map.at(m);
            property->object.emplace(ov::hint::performance_mode(v));
            break;
        }
        case ov_property_key_e::AFFINITY: {
            ov_affinity_e v = *(static_cast<ov_affinity_e*>(value->ptr));
            if (v < ov_affinity_e::NONE || v > ov_affinity_e::HYBRID_AWARE) {
                return ov_status_e::INVALID_C_PARAM;
            }
            ov::Affinity affinity = static_cast<ov::Affinity>(v);
            property->object.emplace(ov::affinity(affinity));
            break;
        }
        case ov_property_key_e::INFERENCE_NUM_THREADS: {
            int32_t v = *(static_cast<int32_t*>(value->ptr));
            property->object.emplace(ov::inference_num_threads(v));
            break;
        }
        case ov_property_key_e::INFERENCE_PRECISION_HINT: {
            ov_element_type_e v = *(static_cast<ov_element_type_e*>(value->ptr));
            if (v > ov_element_type_e::U64) {
                return ov_status_e::INVALID_C_PARAM;
            }

            ov::element::Type type(static_cast<ov::element::Type_t>(v));
            property->object.emplace(ov::hint::inference_precision(type));
            break;
        }
        case ov_property_key_e::CACHE_DIR: {
            char* dir = static_cast<char*>(value->ptr);
            property->object.emplace(ov::cache_dir(std::string(dir)));
            break;
        }
        default:
            return ov_status_e::OUT_OF_BOUNDS;
            break;
        }
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_property_value_clean(ov_property_value_t* value) {
    if (value) {
        if (value->ptr) {
            char* temp = static_cast<char*>(value->ptr);
            delete temp;
        }
        value->ptr = nullptr;
        value->cnt = 0;
    }
}