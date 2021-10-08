// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/dimension.hpp>

namespace ov {
    template<class T>
    bool dim_check(const T &dim) {
        return dim != 0;
    }

    template<>
    bool dim_check(const Dimension &dim) {
        return !dim.get_interval().empty();
    }
}