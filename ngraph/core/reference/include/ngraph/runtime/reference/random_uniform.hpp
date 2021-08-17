// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ctime>
#include <ngraph/type/element_type.hpp>

#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
void random_uniform(const uint64_t* out_shape,
                    const char* min_val,
                    const char* max_val,
                    char* out,
                    const Shape& out_shape_shape,
                    ngraph::element::Type elem_type,
                    uint64_t seed,
                    uint64_t seed2);

const uint32_t crush_resistance_const_lower_value = 0x9E3779B9;
const uint32_t crush_resistance_const_upper_value = 0xBB67AE85;
const uint64_t statistic_maximizing_multiplier_n = 0xD2511F53;
const uint64_t statistic_maximizing_multiplier_counter = 0xCD9E8D57;

}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
