// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"

namespace ov {
namespace proxy {

void create_plugin(std::shared_ptr<InferenceEngine::IInferencePlugin>& plugin);

}  // namespace proxy
}  // namespace ov
