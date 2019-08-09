// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file contains a wrapper class for handling plugin instantiation and releasing resources
 * @file ie_plugin_ptr.hpp
 */
#pragma once

#include "details/ie_so_pointer.hpp"
#include "ie_plugin.hpp"
#include "ie_ihetero_plugin.hpp"
#include "ie_extension.h"
#include <string>

namespace InferenceEngine {
namespace details {

/**
 * @brief This class defines the name of the fabric for creating an IInferencePlugin object in DLL
 */
template<>
class SOCreatorTrait<IInferencePlugin> {
public:
    /**
     * @brief A name of the fabric for creating IInferencePlugin object in DLL
     */
    static constexpr auto name = "CreatePluginEngine";
};

/**
 * @brief This class defines the name of the fabric for creating an IHeteroInferencePlugin object in DLL
 */
IE_SUPPRESS_DEPRECATED_START
template<>
class SOCreatorTrait<IHeteroInferencePlugin> {
public:
    /**
     * @brief A name of the fabric for creating IInferencePlugin object in DLL
     */
    static constexpr auto name = "CreatePluginEngine";
};
IE_SUPPRESS_DEPRECATED_END

}  // namespace details

/**
 * @brief A C++ helper to work with objects created by the plugin.
 * Implements different interfaces.
 */
using InferenceEnginePluginPtr = InferenceEngine::details::SOPointer<IInferencePlugin>;

/** @copybrief InferenceEnginePluginPtr */
IE_SUPPRESS_DEPRECATED_START
using HeteroPluginPtr = InferenceEngine::details::SOPointer<IHeteroInferencePlugin>;
IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
