// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//================================================================================================
// ElementType
//================================================================================================

#pragma once

#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ngraph {
namespace element {
using ov::element::Type;
using ov::element::Type_t;
using TypeVector = std::vector<Type>;

using ov::element::bf16;
using ov::element::boolean;
using ov::element::dynamic;
using ov::element::f16;
using ov::element::f32;
using ov::element::f64;
using ov::element::i16;
using ov::element::i32;
using ov::element::i4;
using ov::element::i64;
using ov::element::i8;
using ov::element::u1;
using ov::element::u16;
using ov::element::u32;
using ov::element::u4;
using ov::element::u64;
using ov::element::u8;
using ov::element::undefined;

template <typename T>
Type from() {
    return ov::element::from<T>();
}
}  // namespace element

/// \brief Return the number of bytes in the compile-time representation of the element type.
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
size_t compiler_byte_size(element::Type_t et);
}  // namespace ngraph
