// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <string>
#include <sstream>
#include <iterator>
#include <vector>
#include <set>
#include <chrono>
#include <ostream>
#include <memory>

#include <cpp/ie_cnn_network.h>
#include <ngraph/function.hpp>
#include <openvino/core/any.hpp>

namespace ov {
namespace test {
    static void PrintTo(const Any& any, std::ostream* os) {
        any.print(*os);
    }
}  // namespace test
}  // namespace ov
