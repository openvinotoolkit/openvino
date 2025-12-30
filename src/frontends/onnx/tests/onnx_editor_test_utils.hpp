// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gtest/gtest.h"
#include "onnx_utils.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace tests {

inline const char* editor_skip_message() {
    return "ONNX Editor functionality is not available when the GraphIterator is enabled. Set ONNX_ITERATOR=0 to run this test.";
}

}  // namespace tests
}  // namespace onnx
}  // namespace frontend
}  // namespace ov

#define SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED()                                        \
    do {                                                                                    \
        if (::ov::frontend::onnx::tests::is_graph_iterator_enabled()) {                     \
            GTEST_SKIP() << ::ov::frontend::onnx::tests::editor_skip_message();             \
        }                                                                                   \
    } while (0)
