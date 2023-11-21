// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <exception>
#include <sstream>
#include <vector>

#include "ngraph/except.hpp"
#include "openvino/core/except.hpp"

namespace ngraph {
using ov::write_all_to_stream;

using CheckFailure = ov::AssertFailure;
}  // namespace ngraph

#define NGRAPH_CHECK_HELPER2(exc_class, ctx, check, ...) OPENVINO_ASSERT_HELPER2(exc_class, ctx, check, __VA_ARGS__)

#define NGRAPH_CHECK_HELPER1(exc_class, ctx, check) OPENVINO_ASSERT_HELPER1(exc_class, ctx, check)

#define NGRAPH_CHECK(...) OPENVINO_ASSERT(__VA_ARGS__)

#define NGRAPH_UNREACHABLE(...)                  NGRAPH_CHECK(false, "Unreachable: ", __VA_ARGS__)
#define NGRAPH_CHECK_HELPER(exc_class, ctx, ...) OPENVINO_ASSERT_HELPER(exc_class, ctx, __VA_ARGS__)
