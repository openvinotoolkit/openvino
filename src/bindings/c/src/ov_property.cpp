// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_property.h"

#include "common.h"

ov_status_e ov_properties_create(ov_properties_t* property, size_t size) {
    if (!property || size < 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_property_t> _property(new ov_property_t[size]);
        property->list = _property.release();
        property->size = size;
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_properties_free(ov_properties_t* properties) {
    if (properties && properties->list) {
        delete[] properties->list;
        properties->size = 0;
    }
}
