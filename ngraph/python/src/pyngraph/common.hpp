
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace Common
{
    const char* string_to_char_arr(const std::string& str);
};
