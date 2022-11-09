// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "serial/headers/2dot8/gna_model_header.hpp"

namespace GNAPluginNS {
namespace HeaderLatest {
using ModelHeader = GNAPluginNS::Header2dot8::ModelHeader;
using RuntimeEndPoint = GNAPluginNS::Header2dot8::RuntimeEndPoint;

template <typename A, typename B>
bool IsFirstVersionLower(const A& first, const B& second) {
    return first.major < second.major || (first.major == second.major && first.minor < second.minor);
}
} // namespace HeaderLatest
} // namespace GNAPluginNS
