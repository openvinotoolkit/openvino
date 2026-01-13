// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

// Method returns path to the requested model name.
// If model can't be found by some reason, empty path
// is returned.
const std::string find_model(const std::string& name);
