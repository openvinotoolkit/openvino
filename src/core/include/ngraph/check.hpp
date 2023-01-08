// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <exception>
#include <sstream>
#include <vector>

#include "ngraph/except.hpp"
#include "openvino/core/except.hpp"

namespace ngraph {
using ov::write_all_to_stream;

using CheckFailure = ov::AssertFailure;
using ov::CheckLocInfo;
}  // namespace ngraph

#define NGRAPH_CHECK_HELPER2(exc_class, ctx, check, ...) OPENVINO_ASSERT_HELPER2(exc_class, ctx, check, __VA_ARGS__)

#define NGRAPH_CHECK_HELPER1(exc_class, ctx, check) OPENVINO_ASSERT_HELPER1(exc_class, ctx, check)

#define NGRAPH_CHECK(...) OPENVINO_ASSERT(__VA_ARGS__)

#define NGRAPH_UNREACHABLE(...)                  NGRAPH_CHECK(false, "Unreachable: ", __VA_ARGS__)
#define NGRAPH_CHECK_HELPER(exc_class, ctx, ...) OPENVINO_ASSERT_HELPER(exc_class, ctx, __VA_ARGS__)
