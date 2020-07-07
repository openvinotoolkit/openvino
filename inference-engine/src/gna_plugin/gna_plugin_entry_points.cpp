// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include "gna_plugin_internal.hpp"

using namespace InferenceEngine;
using namespace std;
using namespace GNAPluginNS;

static const Version gnaPluginDescription = {
        {2, 1},
        CI_BUILD_NUMBER
#if GNA_LIB_VER == 2
        "_with_GNA_LIB_VER==2"
#endif
        ,
        "GNAPlugin"
};

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin *&plugin, ResponseDesc *resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin(gnaPluginDescription, make_shared<GNAPluginInternal>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}
