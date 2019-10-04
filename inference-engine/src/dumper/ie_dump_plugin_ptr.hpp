// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Convinience wrapper class for handling plugin instanciation and releasing resources.
 * @file ie_dump_plugin_ptr.hpp
 */
#pragma once

#include "details/ie_so_pointer.hpp"
#include "ie_dump_plugin.hpp"
#include <string>

namespace InferenceEngine {
namespace details {

template<>
class SOCreatorTrait<IDumpPlugin> {
public:
    static constexpr auto name = "CreateDumpPlugin";
};

}  // namespace details


}  // namespace InferenceEngine


/**
* @typedef DumpPluginPtr
* @brief c++ helper to work with plugin's created objects, implements different interface
*/
typedef InferenceEngine::details::SOPointer<IDumpPlugin> DumpPluginPtr;


