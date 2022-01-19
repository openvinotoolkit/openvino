// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/format.hpp"

TEST(format, to_string) {
    typedef std::underlying_type<cldnn::format::type>::type format_underlying_type;
    for (format_underlying_type i = 0; i < static_cast<format_underlying_type>(cldnn::format::format_num); i++) {
        cldnn::format fmt = static_cast<cldnn::format::type>(i);
        ASSERT_NO_THROW(fmt.to_string()) << "Can't convert to string format " << i;
    }
}

TEST(format, traits) {
    typedef std::underlying_type<cldnn::format::type>::type format_underlying_type;
    for (format_underlying_type i = 0; i < static_cast<format_underlying_type>(cldnn::format::format_num); i++) {
        cldnn::format fmt = static_cast<cldnn::format::type>(i);
        ASSERT_NO_THROW(cldnn::format::traits(fmt)) << "Can't get traits for format " << i;
    }
}
