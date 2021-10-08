// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/dimension.hpp>

namespace ov {
    template<class T>
    bool dims_are_equal(const T &d1, const T &d2) {
        return d1 == d2;
    }

    template<>
    bool dims_are_equal(const Dimension &d1, const Dimension &d2) {
        const auto intersection = d1 & d2;
        return !intersection.get_interval().empty();
    }
}