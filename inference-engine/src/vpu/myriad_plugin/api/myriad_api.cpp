// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "myriad_plugin.h"
#include "myriad_mvnc_wraper.h"

using namespace InferenceEngine;
using namespace vpu::MyriadPlugin;

IE_SUPPRESS_DEPRECATED_START

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePluginInternal *&plugin, ResponseDesc *resp) noexcept {
    try {
        auto mvnc = std::make_shared<Mvnc>();
        // plugin = make_ie_compatible_plugin({{2, 1}, CI_BUILD_NUMBER, "myriadPlugin"},
        //     std::make_shared<Engine>(mvnc));
        plugin = new Engine(mvnc);
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}
