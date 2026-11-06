// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

static const ov::Version version = {CI_BUILD_NUMBER, "openvino_hetero_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::hetero::Plugin, version)

// This plugin does not participate in device-name dispatch; export the probe as a stub.
OV_DEFINE_PLUGIN_ENUMERATE_STUB()