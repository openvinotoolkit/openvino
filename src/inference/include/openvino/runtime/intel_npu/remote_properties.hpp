// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for properties of shared device contexts and shared device memory tensors for NPU device
 *        To use in constructors of Remote objects
 *
 * @file openvino/runtime/intel_npu/remote_properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace intel_npu {

using npu_handle_param = void*;

/**
 * @brief Enum to define the type of the shared memory buffer
 */
enum class MemType {
    L0_INTERNAL_BUF = 0,  //!< Internal L0 buffer type allocated by plugin
    SHARED_BUF = 1,       //!< Shared buffer
};

/**
 * @brief Enum to define the type of the tensor
 */
enum class TensorType {
    INPUT = 0,   //!< Tensor is only used as input
    OUTPUT = 1,  //!< Tensor is only used as output
    BINDED = 2   //!< Tensor could be used as input and output
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const MemType& share_mem_type) {
    switch (share_mem_type) {
    case MemType::L0_INTERNAL_BUF:
        return os << "L0_INTERNAL_BUF";
    case MemType::SHARED_BUF:
        return os << "SHARED_BUF";
    default:
        OPENVINO_THROW("Unsupported memory type");
    }
}

inline std::istream& operator>>(std::istream& is, MemType& share_mem_type) {
    std::string str;
    is >> str;
    if (str == "L0_INTERNAL_BUF") {
        share_mem_type = MemType::L0_INTERNAL_BUF;
    } else if (str == "SHARED_BUF") {
        share_mem_type = MemType::SHARED_BUF;
    } else {
        OPENVINO_THROW("Unsupported memory type: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief This key identifies type of internal shared memory
 * in a shared memory tensor parameter map.
 */
static constexpr Property<MemType> mem_type{"MEM_TYPE"};

/**
 * @brief This key identifies memory handle
 * in a shared memory tensor parameter map
 */
static constexpr Property<npu_handle_param> mem_handle{"MEM_HANDLE"};

/**
 * @brief This key identifies LevelZero context handle
 * in a shared context or shared memory tensor parameter map
 */
static constexpr Property<npu_handle_param> l0_context{"L0_CONTEXT"};

/**
 * @brief This key identifies type of the tensor
 * in a shared memory tensor parameter map.
 */
static constexpr Property<TensorType> tensor_type{"TENSOR_TYPE"};

}  // namespace intel_npu
}  // namespace ov
