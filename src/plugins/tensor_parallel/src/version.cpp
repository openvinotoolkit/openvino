// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

static const ov::Version version = {CI_BUILD_NUMBER, "openvino_tensor_parallel_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::tp::Plugin, version)
