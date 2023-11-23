// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//================================================================================================
// ElementType
//================================================================================================

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include "ngraph/deprecated.hpp"
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
using ov::element::nf4;
using ov::element::u1;
using ov::element::u16;
using ov::element::u32;
using ov::element::u4;
using ov::element::u64;
using ov::element::u8;
using ov::element::string;
using ov::element::undefined;

template <typename T>
NGRAPH_API_DEPRECATED Type from() {
    return ov::element::from<T>();
}
}  // namespace element

/// \brief Return the number of bytes in the compile-time representation of the element type.
NGRAPH_API_DEPRECATED
size_t compiler_byte_size(element::Type_t et);
}  // namespace ngraph
