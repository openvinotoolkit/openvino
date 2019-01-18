// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ie_plugin.hpp>
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include "gna_plugin_internal.hpp"

using namespace InferenceEngine;
using namespace std;
using namespace GNAPluginNS;

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin *&plugin, ResponseDesc *resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin({1, 5, "GNAPlugin", "GNAPlugin"}, make_shared<GNAPluginInternal>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}
