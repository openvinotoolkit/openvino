// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace vpu {

std::string createIOShapeName(std::string srcName);

bool isIOShapeName(std::string name);

} //namespace vpu
