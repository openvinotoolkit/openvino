// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ctime>
#include <ngraph/type/element_type.hpp>
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void random_uniform(const uint64_t* out_shape,
                                const char* min_val,
                                const char* max_val,
                                char* out,
                                const Shape& out_shape_shape,
                                ngraph::element::Type elem_type,
                                uint64_t seed,
                                uint64_t seed2);

        } // namespace reference
    }     // namespace runtime
} // namespace ngraph