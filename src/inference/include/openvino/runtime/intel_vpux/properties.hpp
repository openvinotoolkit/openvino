// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace intel_vpux {


/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("config/target"), default: "";
 */
static constexpr ov::Property<std::string> targetDescriptorPath{"VPU_COMPILER_TARGET_DESCRIPTOR_PATH"};

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Default: "release_kmb";
 */
static constexpr ov::Property<std::string> targetDescriptor{"VPU_COMPILER_TARGET_DESCRIPTOR"};

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("config/compilation"), default: "";
 */
static constexpr ov::Property<std::string> compilationDescriptorPath{"VPU_COMPILER_COMPILATION_DESCRIPTOR_PATH"};

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Default: "release_kmb";
 */
static constexpr ov::Property<std::string> compilationDescriptor{"VPU_COMPILER_COMPILATION_DESCRIPTOR"};

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable eltwise scales alignment
 */
static constexpr ov::Property<bool> eltwiseScalesAlignment{"VPU_COMPILER_ELTWISE_SCALES_ALIGNMENT"};

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable concat scales alignment
 */
static constexpr ov::Property<bool> concatScalesAlignment{"VPU_COMPILER_CONCAT_SCALES_ALIGNMENT"};

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable weights zero points alignment
 */
static constexpr ov::Property<bool> weightsZeroPointsAlignment{"VPU_COMPILER_WEIGHTS_ZERO_POINTS_ALIGNMENT"};

/**
 * @brief [Only for vpu compiler]
 * Type: std::string, default is empty.
 * Provide path to custom layer binding xml file.
 * If layer is present in such an xml, it would be used during inference even if the layer is natively supported.
 */
static constexpr ov::Property<std::string> customLayers{"VPU_COMPILER_CUSTOM_LAYERS"};

/**
 * @brief [Only for vpu compiler]
 * Type: std::string, default is empty.
 * Semicolon separated list of comma separated group and pass values.
 * Removes {group, pass} value from mcm compilation descriptor.
 */
static constexpr ov::Property<std::string> compilationPassBanList{"VPU_COMPILER_COMPILATION_PASS_BAN_LIST"};

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable fusing scaleshift
 */
static constexpr ov::Property<bool> scaleFuseInput{"VPU_COMPILER_SCALE_FUSE_INPUT"};

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "NO".
 * Allow NCHW layout to be set as MCM Model input layout
 */
static constexpr ov::Property<bool> allowNchwMcmInput{"VPU_COMPILER_ALLOW_NCHW_MCM_INPUT"};

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Permute no-op layer can be used as dummy SW layer
 * Used as workaround in HETERO plugin
 */
static constexpr ov::Property<bool> removePermuteNoop{"VPU_COMPILER_REMOVE_PERMUTE_NOOP"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 0. SetNumUpaShaves is not called in that case.
 * Number of shaves to be used by NNCore plug-in during inference
 */
static constexpr ov::Property<int64_t> inferenceShaves{"VPUX_INFERENCE_SHAVES"};

/**
 * Type: Arbitrary string. Default is "-1".
 * This option allows to specify CSRAM size in bytes
 * When the size is -1, low-level SW is responsible for determining the required amount of CSRAM
 * When the size is 0, CSRAM isn't used
 */
static constexpr ov::Property<std::string> csramSize{"VPUX_CSRAM_SIZE"};

} // namespace intel_vpux
} // namespace ov