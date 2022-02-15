// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "all_close.hpp"

::testing::AssertionResult ov::test::all_close(const ov::Tensor& a, const ov::Tensor& b, float rtol, float atol) {
    if (a.get_element_type() != b.get_element_type()) {
        return ::testing::AssertionFailure() << "Cannot compare tensors with different element types";
    }
    switch (a.get_element_type()) {
    case ov::element::u8:
        return all_close<ov::element_type_traits<ov::element::u8>::value_type>(
            a,
            b,
            static_cast<ov::element_type_traits<ov::element::u8>::value_type>(rtol),
            static_cast<ov::element_type_traits<ov::element::u8>::value_type>(atol));
    case ov::element::u16:
        return all_close<ov::element_type_traits<ov::element::u16>::value_type>(
            a,
            b,
            static_cast<ov::element_type_traits<ov::element::u16>::value_type>(rtol),
            static_cast<ov::element_type_traits<ov::element::u16>::value_type>(atol));
    case ov::element::u32:
        return all_close<ov::element_type_traits<ov::element::u32>::value_type>(
            a,
            b,
            static_cast<ov::element_type_traits<ov::element::u32>::value_type>(rtol),
            static_cast<ov::element_type_traits<ov::element::u32>::value_type>(atol));
    case ov::element::u64:
        return all_close<ov::element_type_traits<ov::element::u64>::value_type>(
            a,
            b,
            static_cast<ov::element_type_traits<ov::element::u64>::value_type>(rtol),
            static_cast<ov::element_type_traits<ov::element::u64>::value_type>(atol));
    case ov::element::i8:
        return all_close<ov::element_type_traits<ov::element::i8>::value_type>(
            a,
            b,
            static_cast<ov::element_type_traits<ov::element::i8>::value_type>(rtol),
            static_cast<ov::element_type_traits<ov::element::i8>::value_type>(atol));
    case ov::element::i16:
        return all_close<ov::element_type_traits<ov::element::i16>::value_type>(
            a,
            b,
            static_cast<ov::element_type_traits<ov::element::i16>::value_type>(rtol),
            static_cast<ov::element_type_traits<ov::element::i16>::value_type>(atol));
    case ov::element::boolean:
        return all_close<ov::element_type_traits<ov::element::boolean>::value_type>(
            a,
            b,
            static_cast<ov::element_type_traits<ov::element::boolean>::value_type>(rtol),
            static_cast<ov::element_type_traits<ov::element::boolean>::value_type>(atol));
    case ov::element::i32:
        return all_close<ov::element_type_traits<ov::element::i32>::value_type>(
            a,
            b,
            static_cast<ov::element_type_traits<ov::element::i32>::value_type>(rtol),
            static_cast<ov::element_type_traits<ov::element::i32>::value_type>(atol));
    case ov::element::i64:
        return all_close<ov::element_type_traits<ov::element::i64>::value_type>(
            a,
            b,
            static_cast<ov::element_type_traits<ov::element::i64>::value_type>(rtol),
            static_cast<ov::element_type_traits<ov::element::i64>::value_type>(atol));
    // case ov::element::bf16:
    //     return all_close<ov::element_type_traits<ov::element::bf16>::value_type>(
    //         a,
    //         b,
    //         static_cast<ov::element_type_traits<ov::element::bf16>::value_type>(rtol),
    //         static_cast<ov::element_type_traits<ov::element::bf16>::value_type>(atol));
    // case ov::element::f16:
    //     return all_close<ov::element_type_traits<ov::element::f16>::value_type>(
    //         a,
    //         b,
    //         static_cast<ov::element_type_traits<ov::element::f16>::value_type>(rtol),
    //         static_cast<ov::element_type_traits<ov::element::f16>::value_type>(atol));
    case ov::element::f32:
        return all_close<ov::element_type_traits<ov::element::f32>::value_type>(
            a,
            b,
            static_cast<ov::element_type_traits<ov::element::f32>::value_type>(rtol),
            static_cast<ov::element_type_traits<ov::element::f32>::value_type>(atol));
    case ov::element::f64:
        return all_close<ov::element_type_traits<ov::element::f64>::value_type>(
            a,
            b,
            static_cast<ov::element_type_traits<ov::element::f64>::value_type>(rtol),
            static_cast<ov::element_type_traits<ov::element::f64>::value_type>(atol));
    default:
        return ::testing::AssertionFailure()
               << "Cannot compare tensors with unsupported element type: " << a.get_element_type();
    }
}
