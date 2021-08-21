// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>

namespace AutoPlugin {
class AutoInferencePlugin : public InferenceEngine::IInferencePlugin {
public:
    AutoInferencePlugin() = default;
    ~AutoInferencePlugin() = default;
};

}  // namespace AutoPlugin
