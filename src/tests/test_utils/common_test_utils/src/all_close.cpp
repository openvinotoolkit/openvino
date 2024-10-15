// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/all_close.hpp"

#include "openvino/core/type/element_type_traits.hpp"

namespace ov {
namespace test {
namespace utils {

::testing::AssertionResult all_close(const ov::Tensor& a, const ov::Tensor& b, float rtol, float atol) {
    if (a.get_element_type() != b.get_element_type()) {
        return ::testing::AssertionFailure() << "Cannot compare tensors with different element types";
    }

#define all_close_ov_type(type)                                                        \
    case ov::element::type:                                                            \
        return all_close<ov::element_type_traits<ov::element::type>::value_type>(      \
            a,                                                                         \
            b,                                                                         \
            static_cast<ov::element_type_traits<ov::element::type>::value_type>(rtol), \
            static_cast<ov::element_type_traits<ov::element::type>::value_type>(atol));

    switch (a.get_element_type()) {
        all_close_ov_type(u8);
        all_close_ov_type(u16);
        all_close_ov_type(u32);
        all_close_ov_type(u64);
        all_close_ov_type(i8);
        all_close_ov_type(i16);
        all_close_ov_type(i32);
        all_close_ov_type(i64);
        // all_close_ov_type(bf16);
        // all_close_ov_type(f16);
        all_close_ov_type(f32);
        all_close_ov_type(f64);
        all_close_ov_type(boolean);
    case ov::element::i4:
    case ov::element::u4:
        return all_close(static_cast<const uint8_t*>(a.data()),
                         static_cast<const uint8_t*>(b.data()),
                         a.get_byte_size(),
                         static_cast<uint8_t>(rtol),
                         static_cast<uint8_t>(atol));
        ;
    default:
        return ::testing::AssertionFailure()
               << "Cannot compare tensors with unsupported element type: " << a.get_element_type();
    }
}
}  // namespace utils
}  // namespace test
}  // namespace ov
