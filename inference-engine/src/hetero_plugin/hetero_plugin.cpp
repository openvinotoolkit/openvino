// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <hetero/hetero_plugin.hpp>

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode) CreatePluginEngine(
        InferenceEngine::IInferencePlugin *&plugin,
        InferenceEngine::ResponseDesc *resp) noexcept {
    return HeteroPlugin::CreateHeteroPluginEngine(plugin, resp);
}
