// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <details/ie_so_pointer.hpp>
#include "ie_reader.hpp"

namespace InferenceEngine {
namespace details {

/**
 * @brief This class defines the name of the fabric for creating an IReader object in DLL
 */
template <>
class SOCreatorTrait<IReader> {
public:
    /**
     * @brief A name of the fabric for creating IReader object in DLL
     */
    static constexpr auto name = "CreateReader";
};

}  // namespace details

/**
 * @brief A C++ helper to work with objects created by the plugin.
 *
 * Implements different interfaces.
 */
using IReaderPtr = InferenceEngine::details::SOPointer<IReader>;

}  // namespace InferenceEngine
