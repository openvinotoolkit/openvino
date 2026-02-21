// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <sstream>

#include "node.h"

namespace ov::intel_cpu {

std::stringstream& printInfo(std::stringstream& stream, const NodePtr& node, bool colorUp);
std::stringstream& printDuration(std::stringstream& stream, const NodePtr& node);
void printPluginInfoOnce();
}  // namespace ov::intel_cpu
