// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of abstraction over platform specific shared objects
 *
 * @file ie_so_loader.h
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

#include <memory>

#include "ie_api.h"

namespace InferenceEngine {
namespace details {

/**
 * @deprecated This is internal stuff. Use Inference Engine Plugin API
 * @brief This class provides an OS shared module abstraction
 */
class INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CLASS(SharedObjectLoader) {
    std::shared_ptr<void> _so;

public:
    /**
     * @brief Constructs from existing object
     */
    SharedObjectLoader(const std::shared_ptr<void>& so);

    /**
     * @brief Default constructor
     */
    SharedObjectLoader() = default;

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Loads a library with the wide char name specified.
     * @param pluginName Full or relative path to the plugin library
     */
    explicit SharedObjectLoader(const wchar_t* pluginName);
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

    /**
     * @brief Loads a library with the name specified.
     * @param pluginName Full or relative path to the plugin library
     */
    explicit SharedObjectLoader(const char* pluginName);

    /**
     * @brief A destructor
     */
    ~SharedObjectLoader();

    /**
     * @brief Searches for a function symbol in the loaded module
     * @param symbolName Name of function to find
     * @return A pointer to the function if found
     * @throws Exception if the function is not found
     */
    void* get_symbol(const char* symbolName) const;

    /**
     * @brief Retruns reference to type erased implementation
     * @throws Exception if the function is not found
     */
    std::shared_ptr<void> get() const;
};

}  // namespace details
}  // namespace InferenceEngine
