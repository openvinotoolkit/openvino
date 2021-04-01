// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_param_containers.hpp"

//------------------------------------------------------------------------------
// Initialization of members of class VpuTestParamsContainer
//------------------------------------------------------------------------------

PluginNamesVector VpuTestParamsContainer::testing_plugin_ = {
        { ::vpu::tests::pluginName(), ::vpu::tests::deviceName() }
};

