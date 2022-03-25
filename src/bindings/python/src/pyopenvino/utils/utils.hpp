// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <openvino/core/any.hpp>

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

ov::Any py_object_to_any(const pybind11::object& py_obj);
