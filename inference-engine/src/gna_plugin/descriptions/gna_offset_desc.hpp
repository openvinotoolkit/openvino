// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace GNAPluginNS {
struct OffsetDesc {
    size_t value = 0;
    /**
     * @brief connectTo is true is alternative to positive or equal to zero offset
     * in case when we would like to use zero offset and connect from  pointer set this to negative
     */
    bool connectTo = true;

    OffsetDesc() = default;
    OffsetDesc(size_t new_value) : value(new_value), connectTo(true) { }
    OffsetDesc(size_t new_value, bool new_connectTo) : value(new_value), connectTo(new_connectTo) {
    }
};
} // namespace GNAPluginNS
