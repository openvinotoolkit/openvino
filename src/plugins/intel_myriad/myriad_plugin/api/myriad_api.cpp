// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "myriad_plugin.h"
#include "myriad_mvnc_wrapper.h"

using namespace InferenceEngine;
using namespace vpu::MyriadPlugin;

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "openvino_intel_myriad_plugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, version, std::make_shared<Mvnc>())
