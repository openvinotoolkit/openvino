// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace vpu {

int parseInt(const std::string& src);
bool Positive(int value);
bool Negative(int value);

}  // namespace vpu
