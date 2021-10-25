// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "functional_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "ngraph/coordinate_transform.hpp"
#include <queue>

namespace ov {
namespace test {
 ov::runtime::Tensor create_and_fill_tensor(
        const ov::element::Type element_type,
        const ov::Shape& shape,
        const uint32_t range,
        const int32_t start_from,
        const int32_t resolution,
        const int seed) {
    auto tensor = ov::runtime::Tensor{element_type, shape};
#define CASE(X) case X: ::CommonTestUtils::fill_data_random(        \
    tensor.data<element_type_traits<X>::value_type>(),  \
    shape_size(shape), \
    range, start_from, resolution, seed); break;
    switch (element_type) {
        CASE(ov::element::Type_t::boolean)
        CASE(ov::element::Type_t::bf16)
        CASE(ov::element::Type_t::f16)
        CASE(ov::element::Type_t::f32)
        CASE(ov::element::Type_t::f64)
        CASE(ov::element::Type_t::i8)
        CASE(ov::element::Type_t::i16)
        CASE(ov::element::Type_t::i32)
        CASE(ov::element::Type_t::i64)
        CASE(ov::element::Type_t::u8)
        CASE(ov::element::Type_t::u16)
        CASE(ov::element::Type_t::u32)
        CASE(ov::element::Type_t::u64)
        case ov::element::Type_t::u1:
        case ov::element::Type_t::i4:
        case ov::element::Type_t::u4:
            ::CommonTestUtils::fill_data_random(
                static_cast<uint8_t*>(tensor.data()),
                tensor.get_byte_size(),
                range, start_from, resolution, seed); break;
        default: OPENVINO_UNREACHABLE("Unsupported element type: ", element_type);
    }
#undef CASE
    return tensor;
}
}  // namespace test
}  // namespace ov