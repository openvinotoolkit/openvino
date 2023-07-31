// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "ie_precision.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "arm_compute/core/Types.h"

namespace ov {
namespace intel_cpu {

/**
* @brief Helper function to calculate the size (number of elements) from the given axis 
* to the given axis (exclusive)
* @param shape input shape
* @param start start axis 
* @param end end axis 
* @return number of elements
*/
inline int64_t calcShapeSize(Shape shape, size_t start, size_t end) {
    int64_t size = 1;
    for (size_t i = start; i < end; i++) {
        size *= shape.getDims()[i];
    }
    return size;
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
inline arm_compute::DataType precisionToAclDataType(InferenceEngine::Precision precision) {
    switch (precision) {
        case InferenceEngine::Precision::I8:    return arm_compute::DataType::S8;
        case InferenceEngine::Precision::U8:    return arm_compute::DataType::U8;
        case InferenceEngine::Precision::I16:   return arm_compute::DataType::S16;
        case InferenceEngine::Precision::U16:   return arm_compute::DataType::U16;
        case InferenceEngine::Precision::I32:   return arm_compute::DataType::S32;
        case InferenceEngine::Precision::U32:   return arm_compute::DataType::U32;
        case InferenceEngine::Precision::FP16:  return arm_compute::DataType::F16;
        case InferenceEngine::Precision::FP32:  return arm_compute::DataType::F32;
        case InferenceEngine::Precision::FP64:  return arm_compute::DataType::F64;
        case InferenceEngine::Precision::I64:   return arm_compute::DataType::S64;
        case InferenceEngine::Precision::BF16:  return arm_compute::DataType::BFLOAT16;
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
