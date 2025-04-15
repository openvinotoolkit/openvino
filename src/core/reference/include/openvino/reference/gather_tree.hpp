// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace reference {
void gather_tree(const char* step_ids,
                 const char* parent_ids,
                 const char* max_seq_len,
                 const char* end_token,
                 char* out,
                 const Shape& step_ids_shape,
                 const Shape& parent_ids_shape,
                 const Shape& max_seq_len_shape,
                 const Shape& end_token_shape,
                 const element::Type& type);
}
}  // namespace ov
