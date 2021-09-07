// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <exception>
#include <sstream>
#include <vector>

#include "ngraph/except.hpp"
#include "openvino/core/check.hpp"

namespace ngraph {
using ov::write_all_to_stream;

using ov::CheckFailure;
using ov::CheckLocInfo;
}  // namespace ngraph

#define NGRAPH_CHECK_HELPER2(exc_class, ctx, check, ...) OV_CHECK_HELPER2(exc_class, ctx, check, __VA_ARGS__)

#define NGRAPH_CHECK_HELPER1(exc_class, ctx, check) OV_CHECK_HELPER1(exc_class, ctx, check)

#define NGRAPH_CHECK(...) OV_CHECK(__VA_ARGS__)

#define NGRAPH_UNREACHABLE(...)                  NGRAPH_CHECK(false, "Unreachable: ", __VA_ARGS__)
#define NGRAPH_CHECK_HELPER(exc_class, ctx, ...) OV_CHECK_HELPER(exc_class, ctx, __VA_ARGS__)
