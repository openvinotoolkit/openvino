// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for NPU plugin
 *        To use in set_property, compile_model, import_model, get_property methods
 *
 * @file openvino/runtime/intel_npu/properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @defgroup ov_runtime_npu_prop_cpp_api Intel NPU specific properties
 * @ingroup ov_runtime_cpp_api
 * Set of Intel NPU specific properties.
 */

/**
 * @brief Namespace with Intel NPU specific properties
 */
namespace intel_npu {

/**
 * @brief [Only for NPU Plugin]
 * Type: string
 * Type of NPU compiler to be used for compilation of a network
 * @note Configuration API v 2.0
 */
enum class CompilerType { PLUGIN, DRIVER, PREFER_PLUGIN };

/**
 * @brief Prints a string representation of ov::intel_npu::CompilerType to a stream
 * @param out An output stream to send to
 * @param fmt A compiler type value to print to a stream
 * @return A reference to the `out` stream
 * @note Configuration API v 2.0
 */
inline std::ostream& operator<<(std::ostream& out, const CompilerType& fmt) {
    switch (fmt) {
    case CompilerType::PLUGIN: {
        out << "PLUGIN";
    } break;
    case CompilerType::DRIVER: {
        out << "DRIVER";
    } break;
    case CompilerType::PREFER_PLUGIN: {
        out << "PREFER_PLUGIN";
    } break;
    default:
        out << static_cast<uint32_t>(fmt);
        break;
    }
    return out;
}

/**
 * @brief [Only for NPU Plugin]
 * Type: Arbitrary string.
 * This option allows to specify device.
 * The plugin accepts any value given through this option. If the device is not available, either the driver or the
 * compiler will throw an exception depending on the flow running at the time.
 */
static constexpr ov::Property<std::string> platform{"NPU_PLATFORM"};

/**
 * @brief [Only for NPU plugin]
 * Type: uint64_t
 * Read-only property to get size of already allocated NPU DDR memory (both for discrete/integrated NPU devices)
 *
 * Note: Queries driver both for discrete/integrated NPU devices
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<uint64_t, ov::PropertyMutability::RO> device_alloc_mem_size{"NPU_DEVICE_ALLOC_MEM_SIZE"};

/**
 * @brief [Only for NPU plugin]
 * Type: uint64_t
 * Read-only property to get size of available NPU DDR memory (both for discrete/integrated NPU devices)
 *
 * Note: Queries driver both for discrete/integrated NPU devices
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<uint64_t, ov::PropertyMutability::RO> device_total_mem_size{"NPU_DEVICE_TOTAL_MEM_SIZE"};

/**
 * @brief [Only for NPU plugin]
 * Type: uint32_t
 * Read-only property to get NPU driver version (for both discrete/integrated NPU devices)
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<uint32_t, ov::PropertyMutability::RO> driver_version{"NPU_DRIVER_VERSION"};

/**
 * @brief [Only for NPU Plugin]
 * Type: string
 * Selects the type of NPU compiler to be used for compilation of a network.
 */
static constexpr ov::Property<CompilerType> compiler_type{"NPU_COMPILER_TYPE"};

/**
 * @brief [Only for NPU plugin]
 * Type: uint32_t
 * Read-only property to get NPU compiler version. Composite of Major (16bit MSB) and Minor (16bit LSB)
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<uint32_t, ov::PropertyMutability::RO> compiler_version{"NPU_COMPILER_VERSION"};

/**
 * @brief [Only for NPU compiler]
 * Type: std::string
 * Set various parameters supported by the NPU compiler.
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<std::string> compilation_mode_params{"NPU_COMPILATION_MODE_PARAMS"};

/**
 * @brief [Only for NPU compiler]
 * Type: boolean
 * Set or verify state of dynamic quantization in  the NPU compiler
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<bool> compiler_dynamic_quantization{"NPU_COMPILER_DYNAMIC_QUANTIZATION"};

/**
 * @brief [Only for NPU compiler]
 * Type: boolean
 * This option enables additional optimizations and balances performance and accuracy for QDQ format models, quantized
 * using ONNX Runtime
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<bool> qdq_optimization{"NPU_QDQ_OPTIMIZATION"};

/**
 * @brief [Only for NPU compiler]
 * Type: boolean
 * This option enables additional optimizations to improve performance for QDQ format models, quantized using ONNX
 * Runtime
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<bool> qdq_optimization_aggressive{"NPU_QDQ_OPTIMIZATION_AGGRESSIVE"};

/**
 * @brief [Only for NPU plugin]
 * Type: std::bool
 * Set turbo on or off. The turbo mode, where available, provides a hint to the system to maintain the
 * maximum NPU frequency and memory throughput within the platform TDP limits.
 * Turbo mode is not recommended for sustainable workloads due to higher power consumption and potential impact on other
 * compute resources.
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<bool> turbo{"NPU_TURBO"};

/**
 * @brief [Only for NPU Compiler]
 * Type: integer, default is -1
 * Sets the number of npu tiles to compile the model for.
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<int64_t> tiles{"NPU_TILES"};

/**
 * @brief
 * Type: integer, default is -1
 * Maximum number of tiles supported by the device we compile for. Can be set for offline compilation. If not set, it
 * will be populated by driver.
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<int64_t> max_tiles{"NPU_MAX_TILES"};

/**
 * @brief [Only for NPU plugin]
 * Type: std::bool
 * Bypass caching of the compiled model by UMD cache.
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<bool> bypass_umd_caching{"NPU_BYPASS_UMD_CACHING"};

/**
 * @brief [Only for NPU Plugin]
 * Type: boolean, default is false
 * This option allows to delay loading the weights until inference is created
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<bool> defer_weights_load{"NPU_DEFER_WEIGHTS_LOAD"};

/**
 * @brief [Only for NPU Plugin]
 * Type: boolean, default is false.
 * This option allows running inferences in async mode sequentially in the order in which they are started to optimize
 * host scheduling.
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<bool> run_inferences_sequentially{"NPU_RUN_INFERENCES_SEQUENTIALLY"};

/**
 * @brief [Only for NPU Plugin]
 * Type: boolean, default is false.
 * This option allows to disable pruning of memory during idle time.
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<bool> disable_idle_memory_prunning{"NPU_DISABLE_IDLE_MEMORY_PRUNING"};

/**
 * @brief [Only for NPU Plugin]
 * Type: std::string, default is empty
 * Enables custom stride support for specified input/output tensors by name. This allows working with non-contiguous
 * memory layouts without copying data. The plugin automatically maps these names to the appropriate input/output
 * indices for the compiler.
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<std::string> enable_strides_for("NPU_ENABLE_STRIDES_FOR");

}  // namespace intel_npu
}  // namespace ov
