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

IE_DEFINE_PLUGIN_CREATE_FUNCTION(GNAPluginInternal, gnaPluginDescription)
