// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/core/type.hpp"

namespace ov {
namespace reference {

void paged_attention(char* out,
                        const char* query,		 
                        const char* key, 		
                        const char* value, 
                        const char* key_cache,	
                        const char* value_cache,
                        const element::Type dtype,
                        const Shape& qkv_shape,		
                        const Shape& kv_cache_shape,
                        const int32_t* past_lens,
                        const int32_t* subsequence_begins,	
                        const int32_t* block_indices,		
                        const int32_t* block_indices_begins,                        
                        const int32_t scale,					
                        const int32_t sliding_window, 		
                        const int32_t* alibi_slopes,			
                        const int32_t max_context_len);
}  // namespace reference
}  // namespace ov
