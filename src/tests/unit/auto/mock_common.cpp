// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_common.hpp"

//  getMetric will return a fake ov::Any, gmock will call ostreamer << ov::Any
//  it will cause core dump, so add this special implemented
namespace testing {
namespace internal {
    template<>
    void PrintTo<ov::Any>(const ov::Any& a, std::ostream* os) {
        *os << "using custom PrintTo ov::Any";
    }
}
}
