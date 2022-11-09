// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for GNA plugin
 *        To use in set_property() and get_property() methods of plugins
 *
 * @file config.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @defgroup ov_runtime_gna_prop_cpp_api Intel GNA specific properties
 * @ingroup ov_runtime_cpp_api
 * Set of Intel GNA specific properties.
 */

/**
 * @brief Namespace with Intel GNA specific properties
 */
namespace intel_gna {

/**
 * @brief Property to get an std::string of GNA Library version, usually in the form
 * <API_REVISION>.<RELEASE_LINE>.<RELEASE>.<BUILD>
 * @ingroup ov_runtime_gna_prop_cpp_api
 */
static constexpr Property<std::string, PropertyMutability::RO> library_full_version{"GNA_LIBRARY_FULL_VERSION"};

/**
 * @brief Scale factor provided by the user to use static quantization.
 * This option should be used with floating point value serialized to string with . (dot) as a decimal separator
 * @ingroup ov_runtime_gna_prop_cpp_api
 * @details In the case of multiple inputs, individual scale factors can be provided using the
 *  map where key is layer name and value is scale factor
 * Example:
 * \code{.cpp}
 * ov::Core core;
 * auto model = core.read_model(model_path);
 * std::map<std::string, float> scale_factors;
 * for (auto& input : model->inputs()) {
 *     scale_factors[input.get_any_name()] = 1.0f;
 * }
 * core.set_property("GNA", ov::intel_gna::scale_factors_per_input(scale_factors));
 * \endcode
 */
static constexpr Property<std::map<std::string, float>> scale_factors_per_input{"GNA_SCALE_FACTOR_PER_INPUT"};

/**
 * @brief if turned on, dump GNA firmware model into specified file
 * @ingroup ov_runtime_gna_prop_cpp_api
 */
static constexpr Property<std::string> firmware_model_image_path{"GNA_FIRMWARE_MODEL_IMAGE"};

/**
 * @brief Enum to define software acceleration mode
 * @ingroup ov_runtime_gna_prop_cpp_api
 */
enum class ExecutionMode {
    AUTO = 0,  //!< Uses Intel GNA if available, otherwise uses software execution mode on CPU.
    HW = 1,    //!< Uses Intel GNA if available, otherwise raises an error.
    HW_WITH_SW_FBACK =
        2,         //!< Uses Intel GNA if available, otherwise raises an error.
                   //!< If the hardware queue is not empty, automatically falls back to CPU in the bit-exact mode.
    SW_EXACT = 3,  //!< Executes the GNA-compiled graph on CPU performing calculations
                   //!< in the same precision as the Intel GNA in the bit-exact mode.
    SW_FP32 = 4,   //!< Executes the GNA-compiled graph on CPU but substitutes parameters and calculations
                   //!< from low precision to floating point
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const ExecutionMode& execution_mode) {
    switch (execution_mode) {
    case ExecutionMode::AUTO:
        return os << "GNA_AUTO";
    case ExecutionMode::HW:
        return os << "GNA_HW";
    case ExecutionMode::HW_WITH_SW_FBACK:
        return os << "GNA_HW_WITH_SW_FBACK";
    case ExecutionMode::SW_EXACT:
        return os << "GNA_SW_EXACT";
    case ExecutionMode::SW_FP32:
        return os << "GNA_SW_FP32";
    default:
        throw ov::Exception{"Unsupported execution mode!"};
    }
}

inline std::istream& operator>>(std::istream& is, ExecutionMode& execution_mode) {
    std::string str;
    is >> str;
    if (str == "GNA_AUTO") {
        execution_mode = ExecutionMode::AUTO;
    } else if (str == "GNA_HW") {
        execution_mode = ExecutionMode::HW;
    } else if (str == "GNA_HW_WITH_SW_FBACK") {
        execution_mode = ExecutionMode::HW_WITH_SW_FBACK;
    } else if (str == "GNA_SW_EXACT") {
        execution_mode = ExecutionMode::SW_EXACT;
    } else if (str == "GNA_SW_FP32") {
        execution_mode = ExecutionMode::SW_FP32;
    } else {
        throw ov::Exception{"Unsupported execution mode: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief Enum to define HW compile and execution targets
 * @ingroup ov_runtime_gna_prop_cpp_api
 */
enum class HWGeneration {
    UNDEFINED = 0,  //!< GNA HW generation is undefined
    GNA_2_0 = 1,    //!< GNA HW generation 2.0
    GNA_3_0 = 2,    //!< GNA HW generation 3.0
    GNA_3_5 = 3,    //!< GNA HW generation 3.5
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const HWGeneration& hw_generation) {
    switch (hw_generation) {
    case HWGeneration::UNDEFINED:
        return os << "UNDEFINED";
    case HWGeneration::GNA_2_0:
        return os << "GNA_2_0";
    case HWGeneration::GNA_3_0:
        return os << "GNA_3_0";
    case HWGeneration::GNA_3_5:
        return os << "GNA_3_5";
    default:
        throw ov::Exception{"Unsupported HW generation!"};
    }
}

inline std::istream& operator>>(std::istream& is, HWGeneration& hw_generation) {
    std::string str;
    is >> str;
    if (str == "UNDEFINED") {
        hw_generation = HWGeneration::UNDEFINED;
    } else if (str == "GNA_2_0") {
        hw_generation = HWGeneration::GNA_2_0;
    } else if (str == "GNA_3_0") {
        hw_generation = HWGeneration::GNA_3_0;
    } else if (str == "GNA_3_5") {
        hw_generation = HWGeneration::GNA_3_5;
    } else {
        throw ov::Exception{"Unsupported HW generation: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief GNA proc_type setting that should be one of AUTO, HW, GNA_HW_WITH_SW_FBACK,
 *        GNA_SW_EXACT or SW_FP32
 * @ingroup ov_runtime_gna_prop_cpp_api
 */
static constexpr Property<ExecutionMode> execution_mode{"GNA_DEVICE_MODE"};

/**
 * @brief The option to override the GNA HW execution target. May be one of GNA_2_0, GNA_3_0.
 * By default (in case of no value set) the behavior depends on GNA HW availability:
 * If GNA HW is present, use the option corresponding to this HW.
 * If HW is not present, use the option corresponding to the latest fully supported GNA HW generation.
 * A fully supported GNA HW generation means it must be supported by both the OV GNA Plugin and the core GNA Library.
 * Currently, the latest supported GNA HW generation corresponds to GNA_3_0.
 * @ingroup ov_runtime_gna_prop_cpp_api
 */
static constexpr Property<HWGeneration> execution_target{"GNA_HW_EXECUTION_TARGET"};

/**
 * @brief The option to override the GNA HW compile target. May be one of GNA_2_0, GNA_3_0.
 * By default the same as execution_target.
 * @ingroup ov_runtime_gna_prop_cpp_api
 */
static constexpr Property<HWGeneration> compile_target{"GNA_HW_COMPILE_TARGET"};

/**
 * @brief if enabled produced minimum memory footprint for compiled model in GNA memory, default value is true
 * @ingroup ov_runtime_gna_prop_cpp_api
 */
static constexpr Property<bool> memory_reuse{"GNA_COMPACT_MODE"};

/**
 * @brief Enum to define PWL design algorithm
 * @ingroup ov_runtime_gna_prop_cpp_api
 */
enum class PWLDesignAlgorithm {
    UNDEFINED = 0,             //!< PWL approximation algorithm is undefined
    RECURSIVE_DESCENT = 1,     //!< Recursive Descent Algorithm
    UNIFORM_DISTRIBUTION = 2,  //!< Uniform distribution algorithm
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const PWLDesignAlgorithm& pwl_design_algo) {
    switch (pwl_design_algo) {
    case PWLDesignAlgorithm::UNDEFINED:
        return os << "UNDEFINED";
    case PWLDesignAlgorithm::RECURSIVE_DESCENT:
        return os << "RECURSIVE_DESCENT";
    case PWLDesignAlgorithm::UNIFORM_DISTRIBUTION:
        return os << "UNIFORM_DISTRIBUTION";
    default:
        throw ov::Exception{"Unsupported PWL design algorithm!"};
    }
}

inline std::istream& operator>>(std::istream& is, PWLDesignAlgorithm& pwl_design_algo) {
    std::string str;
    is >> str;
    if (str == "UNDEFINED") {
        pwl_design_algo = PWLDesignAlgorithm::UNDEFINED;
    } else if (str == "RECURSIVE_DESCENT") {
        pwl_design_algo = PWLDesignAlgorithm::RECURSIVE_DESCENT;
    } else if (str == "UNIFORM_DISTRIBUTION") {
        pwl_design_algo = PWLDesignAlgorithm::UNIFORM_DISTRIBUTION;
    } else {
        throw ov::Exception{"Unsupported PWL design algorithm: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief The option to set PWL design algorithm.
 * By default the optimized algorithm called "Recursive Descent Algorithm for Finding
 * the Optimal Minimax Piecewise Linear Approximation of Convex Functions" is used.
 * If value is UNIFORM_DISTRIBUTION then simple uniform distribution is used to create
 * PWL approximation of activation functions.
 * Uniform distribution usually gives poor approximation with the same number of segments
 * @ingroup ov_runtime_gna_prop_cpp_api
 */
static constexpr Property<PWLDesignAlgorithm> pwl_design_algorithm{"GNA_PWL_DESIGN_ALGORITHM"};

/**
 * @brief The option to allow to specify the maximum error percent that the optimized algorithm finding
 * will be used to find PWL functions.
 * By default (in case of NO value set), 1.0 value is used.
 * @ingroup ov_runtime_gna_prop_cpp_api
 */
static constexpr Property<float> pwl_max_error_percent{"GNA_PWL_MAX_ERROR_PERCENT"};

}  // namespace intel_gna
}  // namespace ov
