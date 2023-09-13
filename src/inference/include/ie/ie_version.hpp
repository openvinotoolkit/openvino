// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides versioning information for the Inference Engine library
 *
 * @file ie_version.hpp
 */
#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

/**
 * @def IE_VERSION_MAJOR
 * @brief Defines Inference Engine major version
 *
 * @def IE_VERSION_MINOR
 * @brief Defines Inference Engine minor version
 *
 * @def IE_VERSION_PATCH
 * @brief Defines Inference Engine patch version
 */

#define IE_VERSION_MAJOR 2023
#define IE_VERSION_MINOR 2
#define IE_VERSION_PATCH 0

#include "ie_api.h"

/**
 * @brief Inference Engine C++ API
 */
namespace InferenceEngine {
IE_SUPPRESS_DEPRECATED_START

/**
 * @struct Version
 * @brief  Represents version information that describes plugins and the inference engine runtime library
 */
#pragma pack(push, 1)
struct INFERENCE_ENGINE_1_0_DEPRECATED Version {
    IE_SUPPRESS_DEPRECATED_START
    /**
     * @deprecated Use IE_VERSION_[MAJOR|MINOR|PATCH] definitions, buildNumber property
     * @brief An API version reflects the set of supported features
     */
    struct INFERENCE_ENGINE_1_0_DEPRECATED ApiVersion {
        INFERENCE_ENGINE_DEPRECATED("Use IE_VERSION_[MAJOR|MINOR|PATCH] definitions, buildNumber property")
        int major;  //!< A major version
        INFERENCE_ENGINE_DEPRECATED("Use IE_VERSION_[MAJOR|MINOR|PATCH] definitions, buildNumber property")
        int minor;  //!< A minor version

        /**
         * @brief A default construtor
         */
        ApiVersion() {
            major = 0;
            minor = 0;
        }

        /**
         * @brief A default construtor
         * @param v A version to copy
         */
        ApiVersion(const ApiVersion& v) {
            major = v.major;
            minor = v.minor;
        }

        /**
         * @brief A default construtor
         * @param _major A major version to copy
         * @param _minor A minor version to copy
         */
        ApiVersion(int _major, int _minor) {
            major = _major;
            minor = _minor;
        }

        /**
         * @brief A copy operator
         * @param other An object to copy
         * @return A copy
         */
        ApiVersion& operator=(const ApiVersion& other) {
            major = other.major;
            minor = other.minor;
            return *this;
        }
    };

    /**
     * @brief An API version reflects the set of supported features
     */
    ApiVersion apiVersion;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief A null terminated string with build number
     */
    const char* buildNumber;

    /**
     * @brief A null terminated description string
     */
    const char* description;
};
#pragma pack(pop)

/**
 * @brief Gets the current Inference Engine version
 *
 * @return The current Inference Engine version
 */
INFERENCE_ENGINE_API(const Version*) GetInferenceEngineVersion() noexcept;

IE_SUPPRESS_DEPRECATED_END
}  // namespace InferenceEngine
