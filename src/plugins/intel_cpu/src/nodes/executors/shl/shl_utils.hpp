// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "csinn/csinn_data_structure.h"
#include "csinn/csinn_runtime.h"
#include "memory_desc/cpu_memory_desc.h"

namespace ov::intel_cpu {

/**
* @brief Return Shl DataType that corresponds to the given precision
* @param precision precision to be converted
* @return Shl DataType
*/
inline csinn_dtype_enum precisionToShlDataType(ov::element::Type precision) {
    switch (precision) {
        case ov::element::i8:   return CSINN_DTYPE_INT8;
        case ov::element::u8:   return CSINN_DTYPE_UINT8;
        case ov::element::i16:  return CSINN_DTYPE_INT16;
        case ov::element::u16:  return CSINN_DTYPE_UINT16;
        case ov::element::i32:  return CSINN_DTYPE_INT32;
        case ov::element::u32:  return CSINN_DTYPE_UINT32;
        case ov::element::f16:  return CSINN_DTYPE_FLOAT16;
        case ov::element::f32:  return CSINN_DTYPE_FLOAT32;
        case ov::element::f64:  return CSINN_DTYPE_FLOAT64;
        case ov::element::i64:  return CSINN_DTYPE_INT64;
        case ov::element::bf16: return CSINN_DTYPE_BFLOAT16;
        default:
            OPENVINO_THROW("Unknown data type for Shl");
    }
}

/**
* @brief Return Shl DataLayout that corresponds to MemoryDecs layout
* @param desc MemoryDecs from which layout is retrieved
* @param is_weights True if it's a layout of weights
* @return Shl DataLayout or CSINN_LAYOUT_NULL if the layout is unknown
*/
inline csinn_layout_enum getShlDataLayoutByMemoryDesc(const MemoryDescPtr& desc, bool is_weights = false) {
    if (desc->hasLayoutType(LayoutType::ncsp)) {
        switch (desc->getShape().getRank()) {
            case 1: return is_weights ? CSINN_LAYOUT_O     : CSINN_LAYOUT_N;
            case 2: return is_weights ? CSINN_LAYOUT_OI    : CSINN_LAYOUT_NC;
            case 3: return is_weights ? CSINN_LAYOUT_OIW   : CSINN_LAYOUT_NCW;
            case 4: return is_weights ? CSINN_LAYOUT_OIHW  : CSINN_LAYOUT_NCHW;
            case 5: return is_weights ? CSINN_LAYOUT_OIDHW : CSINN_LAYOUT_NCDHW;
        }
    } else if (desc->hasLayoutType(LayoutType::nspc)) {
        switch (desc->getShape().getRank()) {
            case 3: return is_weights ? CSINN_LAYOUT_OWI   : CSINN_LAYOUT_NWC;
            case 4: return is_weights ? CSINN_LAYOUT_OHWI  : CSINN_LAYOUT_NHWC;
            case 5: return is_weights ? CSINN_LAYOUT_ODHWI : CSINN_LAYOUT_NDHWC;
        }
    }
    return CSINN_LAYOUT_NULL;
}

}  // namespace ov::intel_cpu
