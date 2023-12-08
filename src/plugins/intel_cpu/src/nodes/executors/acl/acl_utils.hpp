// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "memory_desc/cpu_memory_desc.h"
#include "arm_compute/core/Types.h"
// #include "openvino/core/type/element_type.hpp"
namespace ov {
namespace intel_cpu {

/**
* @brief ACL supports arm_compute::MAX_DIMS maximum. The method squashes the last
* dimensions in order to comply with this limitation
* @param dims vector of dimensions to squash
* @return vector of dimensions that complies to ACL
*/
inline VectorDims collapse_dims_to_max_rank(VectorDims dims) {
    const size_t MAX_NUM_SHAPE = arm_compute::MAX_DIMS;
    VectorDims result_dims(MAX_NUM_SHAPE - 1);
    if (dims.size() >= MAX_NUM_SHAPE) {
        for (size_t i = 0; i < MAX_NUM_SHAPE - 1; i++) {
            result_dims[i] = dims[i];
        }
        for (size_t i = MAX_NUM_SHAPE - 1; i < dims.size(); i++) {
            result_dims[MAX_NUM_SHAPE - 2] *= dims[i];
        }
    } else {
        result_dims = dims;
    }
    return result_dims;
}

/**
* @brief ACL handles NHWC specifically, it thinks it is NCHW, so we need to change layout manually:
* NCHW (0, 1, 2, 3) -> NHWC (0, 2, 3, 1)
* @param shape shape to convert
* @return none
*/
inline void changeLayoutToNhwc(VectorDims& shape) {
    std::swap(shape[1], shape[2]);
    std::swap(shape[2], shape[3]);
}

/**
* @brief Return ComputeLibrary TensorShape with reverted layout schema used in ACL 
* @param dims vector of dimensions to convert
* @return ComputeLibrary TensorShape object
*/
inline arm_compute::TensorShape shapeCast(const VectorDims& dims) {
    arm_compute::TensorShape tensorShape;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        tensorShape.set(dims.size() - i - 1, dims[i], false);
    }
    if (tensorShape.num_dimensions() == 0) {
        tensorShape.set(0, 1, false);
        tensorShape.set_num_dimensions(1);
    }
    return tensorShape;
}

inline std::size_t axisCast(const std::size_t axis, const std::size_t shapeSize) {
    return shapeSize - axis - 1;
}

inline Dim vectorProduct(const VectorDims& vec, size_t size) {
    Dim prod = 1;
    for (size_t i = 0; i < size; ++i)
        prod *= vec[i];
    return prod;
}

/**
* @brief Return ComputeLibrary DataType that corresponds to the given precision
* @param precision precision to be converted
* @return ComputeLibrary DataType or UNKNOWN if precision is not mapped to DataType
*/
inline arm_compute::DataType precisionToAclDataType(ov::element::Type precision) {
    switch (precision) {
        case ov::element::i8:    return arm_compute::DataType::S8;
        case ov::element::u8:    return arm_compute::DataType::U8;
        case ov::element::i16:   return arm_compute::DataType::S16;
        case ov::element::u16:   return arm_compute::DataType::U16;
        case ov::element::i32:   return arm_compute::DataType::S32;
        case ov::element::u32:   return arm_compute::DataType::U32;
        case ov::element::f16:  return arm_compute::DataType::F16;
        case ov::element::f32:  return arm_compute::DataType::F32;
        case ov::element::f64:  return arm_compute::DataType::F64;
        case ov::element::i64:   return arm_compute::DataType::S64;
        case ov::element::bf16:  return arm_compute::DataType::BFLOAT16;
        default:                                return arm_compute::DataType::UNKNOWN;
    }
}

/**
* @brief Return ComputeLibrary DataLayout that corresponds to MemoryDecs layout
* @param desc MemoryDecs from which layout is retrieved
* @param treatAs4D the flag that treats MemoryDecs as 4D shape
* @return ComputeLibrary DataLayout or UNKNOWN if MemoryDecs layout is not mapped to DataLayout
*/
inline arm_compute::DataLayout getAclDataLayoutByMemoryDesc(MemoryDescCPtr desc) {
    if (desc->hasLayoutType(LayoutType::ncsp)) {
        if (desc->getShape().getRank() <= 4) return arm_compute::DataLayout::NCHW;
        if (desc->getShape().getRank() == 5) return arm_compute::DataLayout::NCDHW;
    } else if (desc->hasLayoutType(LayoutType::nspc)) {
        if (desc->getShape().getRank() <= 4) return arm_compute::DataLayout::NHWC;
        if (desc->getShape().getRank() == 5) return arm_compute::DataLayout::NDHWC;
    }
    return arm_compute::DataLayout::UNKNOWN;
}

}   // namespace intel_cpu
}   // namespace ov
