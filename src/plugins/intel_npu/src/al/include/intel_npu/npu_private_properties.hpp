// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/intel_npu/properties.hpp"

namespace ov {
namespace intel_npu {

namespace Platform {

constexpr std::string_view AUTO_DETECT = "AUTO_DETECT";  // Auto detection
constexpr std::string_view NPU3720 = "3720";             // NPU3720
constexpr std::string_view NPU4000 = "4000";             // NPU4000

/**
 * @brief Converts the given platform value to the standard one.
 * @details The same platform value can be defined in multiple ways (e.g. "3720" vs "VPU3720" vs "NPU3720"). The current
 * function converts the prefixed variants to the non-prefixed ones in order to enable the comparison between platform
 * values.
 *
 * The values already found in the standard form are returned as they are.
 *
 * @param platform The value to be converted.
 * @return The same platform value given as parameter but converted to the standard form.
 */
inline std::string standardize(const std::string_view platform) {
    constexpr std::string_view VPUPrefix = "VPU";
    constexpr std::string_view NPUPrefix = "NPU";

    if (!platform.compare(0, VPUPrefix.length(), VPUPrefix) || !platform.compare(0, NPUPrefix.length(), NPUPrefix)) {
        return std::string(platform).substr(NPUPrefix.length());
    }

    return std::string(platform);
}

}  // namespace Platform

/**
 * @enum ColorFormat
 * @brief Extra information about input color format for preprocessing
 * @note Configuration API v 2.0
 */
enum ColorFormat : uint32_t {
    RAW = 0u,  ///< Plain blob (default), no extra color processing required
    RGB,       ///< RGB color format
    BGR,       ///< BGR color format, default in DLDT
    RGBX,      ///< RGBX color format with X ignored during inference
    BGRX,      ///< BGRX color format with X ignored during inference
};

/**
 * @brief Prints a string representation of ov::intel_npu::ColorFormat to a stream
 * @param out An output stream to send to
 * @param fmt A color format value to print to a stream
 * @return A reference to the `out` stream
 * @note Configuration API v 2.0
 */
inline std::ostream& operator<<(std::ostream& out, const ColorFormat& fmt) {
    switch (fmt) {
    case ColorFormat::RAW: {
        out << "RAW";
    } break;
    case ColorFormat::RGB: {
        out << "RGB";
    } break;
    case ColorFormat::BGR: {
        out << "BGR";
    } break;
    case ColorFormat::RGBX: {
        out << "RGBX";
    } break;
    case ColorFormat::BGRX: {
        out << "BGRX";
    } break;
    default:
        out << static_cast<uint32_t>(fmt);
        break;
    }
    return out;
}

/**
 * @brief [Only for NPU Plugin]
 * Type: string, default is MLIR.
 * Type of NPU compiler to be used for compilation of a network
 * @note Configuration API v 2.0
 */
enum class CompilerType { MLIR, DRIVER };

/**
 * @brief Prints a string representation of ov::intel_npu::CompilerType to a stream
 * @param out An output stream to send to
 * @param fmt A compiler type value to print to a stream
 * @return A reference to the `out` stream
 * @note Configuration API v 2.0
 */
inline std::ostream& operator<<(std::ostream& out, const CompilerType& fmt) {
    switch (fmt) {
    case CompilerType::MLIR: {
        out << "MLIR";
    } break;
    case CompilerType::DRIVER: {
        out << "DRIVER";
    } break;
    default:
        out << static_cast<uint32_t>(fmt);
        break;
    }
    return out;
}

/**
 * @brief [Only for NPU Plugin]
 * Type: String. Default is "AUTO".
 * This option is added for enabling batching on plugin.
 * Possible values: "AUTO", "COMPILER", "PLUGIN".
 */
enum class BatchMode {
    AUTO = 0,
    COMPILER = 1,
    PLUGIN = 2,
};

/**
 * @brief Prints a string representation of ov::intel_npu::BatchMode to a stream
 * @param out An output stream to send to
 * @param fmt A value for batching on plugin to print to a stream
 * @return A reference to the `out` stream
 * @note Configuration API v 2.0
 */
inline std::ostream& operator<<(std::ostream& out, const BatchMode& fmt) {
    switch (fmt) {
    case BatchMode::AUTO: {
        out << "AUTO";
    } break;
    case BatchMode::COMPILER: {
        out << "COMPILER";
    } break;
    case BatchMode::PLUGIN: {
        out << "PLUGIN";
    } break;
    default:
        out << static_cast<uint32_t>(fmt);
        break;
    }
    return out;
}

/**
 * @brief [Only for NPU Plugin]
 * Default is "ITERATIVE".
 * Switches between different implementations of the "weights separation" feature.
 */
enum class WSVersion {
    ONE_SHOT = 0,
    ITERATIVE = 1,
};

inline std::ostream& operator<<(std::ostream& out, const WSVersion& wsVersion) {
    switch (wsVersion) {
    case WSVersion::ONE_SHOT: {
        out << "ONE_SHOT";
    } break;
    case WSVersion::ITERATIVE: {
        out << "ITERATIVE";
    } break;
    default: {
        OPENVINO_THROW("Unsupported value for the weights separation version:", wsVersion);
    }
    }
    return out;
}

inline std::istream& operator>>(std::istream& is, WSVersion& wsVersion) {
    std::string str;
    is >> str;
    if (str == "ONE_SHOT") {
        wsVersion = WSVersion::ONE_SHOT;
    } else if (str == "ITERATIVE") {
        wsVersion = WSVersion::ITERATIVE;
    } else {
        OPENVINO_THROW("Unsupported value for the weights separation version:", str);
    }
    return is;
}

/**
 * @brief [Only for NPU Plugin]
 * Type: string, default is MODEL.
 * Type of profiling to execute. Can be Model (default) or INFER (based on npu timestamps)
 * @note Configuration API v 2.0
 */
enum class ProfilingType { MODEL, INFER };

/**
 * @brief Prints a string representation of ov::intel_npu::ProfilingType to a stream
 * @param out An output stream to send to
 * @param fmt A profiling type value to print to a stream
 * @return A reference to the `out` stream
 * @note Configuration API v 2.0
 */
inline std::ostream& operator<<(std::ostream& out, const ProfilingType& fmt) {
    switch (fmt) {
    case ProfilingType::MODEL: {
        out << "MODEL";
    } break;
    case ProfilingType::INFER: {
        out << "INFER";
    } break;
    default:
        out << static_cast<uint32_t>(fmt);
        break;
    }
    return out;
}

/**
 * @brief Defines the options corresponding to the legacy set of values.
 */
enum class LegacyPriority {
    LOW = 0,     //!<  Low priority
    MEDIUM = 1,  //!<  Medium priority
    HIGH = 2     //!<  High priority
};

inline std::ostream& operator<<(std::ostream& os, const LegacyPriority& priority) {
    switch (priority) {
    case LegacyPriority::LOW:
        return os << "MODEL_PRIORITY_LOW";
    case LegacyPriority::MEDIUM:
        return os << "MODEL_PRIORITY_MED";
    case LegacyPriority::HIGH:
        return os << "MODEL_PRIORITY_HIGH";
    default:
        OPENVINO_THROW("Unsupported model priority value");
    }
}

inline std::istream& operator>>(std::istream& is, LegacyPriority& priority) {
    std::string str;
    is >> str;
    if (str == "MODEL_PRIORITY_LOW") {
        priority = LegacyPriority::LOW;
    } else if (str == "MODEL_PRIORITY_MED") {
        priority = LegacyPriority::MEDIUM;
    } else if (str == "MODEL_PRIORITY_HIGH") {
        priority = LegacyPriority::HIGH;
    } else {
        OPENVINO_THROW("Unsupported model priority: ", str);
    }
    return is;
}

/**
 * @brief Due to driver compatibility constraints, the set of model priority values corresponding to the OpenVINO legacy
 * API is being maintained here.
 * @details The OpenVINO API has made changes with regard to the values used to describe model priorities (e.g.
 * "MODEL_PRIORITY_MED" -> "MEDIUM"). The NPU plugin can't yet discard this since the newer values may not be
 * recognized by older drivers.
 */
static constexpr ov::Property<LegacyPriority, ov::PropertyMutability::RO> legacy_model_priority{"MODEL_PRIORITY"};

/**
 * @brief [Only for NPU Plugin]
 * Type: Arbitrary string.
 * This option allows to specify device.
 * The plugin accepts any value given through this option. If the device is not available, either the driver or the
 * compiler will throw an exception depending on the flow running at the time.
 */
static constexpr ov::Property<std::string> platform{"NPU_PLATFORM"};

/**
 * @brief
 * Type: integer, default is -1
 * Device stepping ID. If unset, it will be automatically obtained from driver
 */
static constexpr ov::Property<int64_t> stepping{"NPU_STEPPING"};

/**
 * @brief [Only for NPU Plugin]
 * Type: string, default is DRIVER.
 * Selects the type of NPU compiler to be used for compilation of a network.
 * 'DRIVER' is the default value.
 */
static constexpr ov::Property<CompilerType> compiler_type{"NPU_COMPILER_TYPE"};

/**
 * @brief
 * Selects different compilation pipelines.
 */
static constexpr ov::Property<std::string> compilation_mode{"NPU_COMPILATION_MODE"};

/**
 * @brief [Only for NPU Plugin]
 * Type: integer, default is -1
 * Sets the number of DMA engines that will be used to execute the model.
 */
static constexpr ov::Property<int64_t> dma_engines{"NPU_DMA_ENGINES"};

/**
 * @brief
 * Type: Boolean. Default is "NO".
 * Determines which branch we use for dynamic shapes.
 * If set to 'YES', we immediately apply the bounds so that we have a static shape for further work.
 * If not, we store the related information in TensorAttr and the IE representation looks
 * like this: tensor<1x?x3xf32, {bounds = [1, 18, 3], ..}>.
 * Possible values: "YES", "NO".
 */
static constexpr ov::Property<std::string> dynamic_shape_to_static{"NPU_DYNAMIC_SHAPE_TO_STATIC"};

/**
 * @brief [Only for NPU Plugin]
 * Type: string, default is empty.
 * MODEL - model layer profiling is done
 * INFER - npu inference performance numbers are measured
 * Model layers profiling are used if this string is empty
 */
static constexpr ov::Property<ProfilingType> profiling_type{"NPU_PROFILING_TYPE"};

/**
 * @brief [Only for NPU Plugin]
 * Type: String. Default is "AUTO".
 * This option is added for enabling batching on plugin, otherwise batching will be handled by compiler.
 * Possible values: "AUTO", "PLUGIN", "COMPILER".
 */
static constexpr ov::Property<BatchMode> batch_mode{"NPU_BATCH_MODE"};

/**
 * @brief [Experimental, only for NPU Plugin]
 * Type: enum. Default is "ITERATIVE".
 *
 * The value stored in this entry indicates which implementation of the "weights separation" feature will be used.
 * Note: NPU_COMPILER_TYPE = DRIVER & NPU_SEPARATE_WEIGHTS_VERSION = ONE_SHOT are not compatible.
 */
static constexpr ov::Property<WSVersion> separate_weights_version{"NPU_SEPARATE_WEIGHTS_VERSION"};

/**
 * @brief [Only for NPU Plugin]
 * Type: bool. Default is "false".
 *
 * This option enables/disables the "weights separation" feature. If enabled, the result of compilation will be a binary
 * object stripped of a significant amount of weights. Before running the model, these weights need to be provided by
 * external means.
 */
static constexpr ov::Property<bool> weightless_blob{"NPU_WEIGHTLESS_BLOB"};

/**
 * @brief [Only for NPU Plugin]
 * Type: bool. Default is "true".
 *
 * This config option concerns the algorithm used for serializing the "ov::Model" at compilation time in order to be
 * passed through the driver.
 *
 * The base serializer is the OV implementation of the "XmlSerializer" without any extensions. All weights are copied in
 * a separate buffer. By turning this off, the NPU extension of the serializer is enabled. This allows optimizing the
 * process by reducing the amount of weights that will be copied in a separate buffer. However, this solution may be
 * less reliable.
 */
static constexpr ov::Property<bool> use_base_model_serializer{"NPU_USE_BASE_MODEL_SERIALIZER"};

/**
 * @brief [Only for NPU Plugin]
 * Type: size_t. Default is 0.
 *
 * Effective only if "use_base_model_serializer" is set to false. All "ov::Constant" buffers smaller than this value
 * (bytes size) will be copied in a separate buffer. The rest of the weights will be reconstructed at de-serialization
 * time using buffer pointers.
 */
static constexpr ov::Property<size_t> serialization_weights_size_threshold{"NPU_SERIALIZATION_WEIGHTS_SIZE_THRESHOLD"};

/**
 * @brief [Experimental, only for NPU Plugin]
 * Type: integer.
 *
 * Used for communicating a state to the compiler when compiling a model using the compiler-in-driver interfaces. This
 * takes effect only when weights separation is enabled and "NPU_SEPARATE_WEIGHTS_VERSION" is set to "ITERATIVE".
 */
static constexpr ov::Property<uint32_t> ws_compile_call_number{"WS_COMPILE_CALL_NUMBER"};

/**
 * @brief [Only for NPU Plugin]
 * Type: String. Default is "".
 * This option is added for providing a fine-grained batched model compilation control, otherwise batching compilation
 * params will be determined automatically. Should be specified only when a model compilation is failed due to incorrect
 * detection of batch dimension presence including false-positive and false-negative cases. NPU compiler supports two
 * batch compile options by now: "unroll" and "debatch" - either can be activated using by setting
 * "batch-compile-method" into the desired value. Leveragind the compile method "debatch" allows the additional param
 * "debatcher-settings" being configured, which introduces the declared fine-grained compilation control suboptions. The
 * suboption "debatcher-input-coefficients-partitions" determines how to split or debatch input tensors of an original
 * model.
 *
 * Let's look at the following example:
 * "batch-compile-method=debatch debatcher-settings={debatcher-input-coefficients-partitions=[0-1],[13-4],[1-1]}".
 *
 * These mean that we want to "debatch" inputs of a batched network providing that:
 * - a batch dimension N of a first intput is on the 0-position (of its layout abbreviation);
 * - the N dimension of a second input is on 13th-position of its layout;
 * - and the N dimension of a third input is on 1-position of its layout accordingly.
 * Thus the first digit of a pair of values enclosed by symbols'[' and ']' determines N dimension position in a layout
 * of a corresponding input. A second value of the pair represents a "native" value of N-dimension of a tensor in
 * assumption that having this value, the tensor becomes "non-batched" or a plain tensor. In the example above:
 * - the non-batched tensor of the first input is assumed to have 1 in N-dimension (on the 0 position);
 * - the second tensor assumed non-batched when it got 4 as a valua of N-dimension on the 13th-position
 * - the third tensor is a plain tensor when it has 1 in N-dimension on the 1-position of its layout
 *
 * The given "debatcher-input-coefficients-partitions" provides the NPU compiler with sufficient information in order to
 * compile a complicatied batched model, which might not be auto recognized by intrinsic heuristics
 *
 * Possible values: "", "batch-compile-method=unroll batch-unroll-settings={skip-unroll-batch=false}",
 * "batch-compile-method=debatch debatcher-settings={debatcher-input-coefficients-partitions=[0-1],[0-1],[0-1]}".
 */
static constexpr ov::Property<std::string> batch_compiler_mode_settings{"NPU_BATCH_COMPILER_MODE_SETTINGS"};

/**
 * @brief [Only for NPU Plugin]
 * Type: integer, default is 1
 * This option allows to omit creating an executor and therefore to omit running an inference when its value is 0
 */
static constexpr ov::Property<int64_t> create_executor{"NPU_CREATE_EXECUTOR"};

/**
 * @brief Read-only property to get the name of used backend
 */
static constexpr ov::Property<std::string, ov::PropertyMutability::RO> backend_name{"NPU_BACKEND_NAME"};

/**
 * @brief [Only for NPU compiler]
 * Type: std::string, default is empty.
 * Config for Backend pipeline

 * Available values: enable-memory-side-cache=true/false
 * Available values: enable-partial-workload-management=true/false
 */
static constexpr ov::Property<std::string> backend_compilation_params{"NPU_BACKEND_COMPILATION_PARAMS"};

/**
 * @brief [Only for NPU Plugin]
 * Type: boolean, default is false.
 * This option allows to skip the blob version check
 */
static constexpr ov::Property<bool> disable_version_check{"NPU_DISABLE_VERSION_CHECK"};

}  // namespace intel_npu
}  // namespace ov
