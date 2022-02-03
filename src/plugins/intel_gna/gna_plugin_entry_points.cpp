// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "gna_plugin_internal.hpp"

using namespace InferenceEngine;
using namespace std;
using namespace GNAPluginNS;

static const Version gnaPluginDescription = {
        {2, 1},
        CI_BUILD_NUMBER
        "_with_GNA_LIB_VER==2"
        ,
        "openvino_intel_gna_plugin"
};

IE_DEFINE_PLUGIN_CREATE_FUNCTION(GNAPluginInternal, gnaPluginDescription)
