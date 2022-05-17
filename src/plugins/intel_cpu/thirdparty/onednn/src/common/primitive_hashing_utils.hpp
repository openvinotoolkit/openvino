/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef COMMON_PRIMITIVE_HASHING_UTILS_HPP
#define COMMON_PRIMITIVE_HASHING_UTILS_HPP

#include "c_types_map.hpp"
#include "oneapi/dnnl/dnnl.h"
#include "primitive_attr.hpp"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace primitive_hashing {

size_t get_md_hash(const memory_desc_t &md);
size_t get_attr_hash(const primitive_attr_t &attr);
size_t get_post_op_hash(size_t seed, const post_ops_t &post_ops);

template <typename T>
size_t get_array_hash(size_t seed, const T *v, int size) {
    for (int i = 0; i < size; i++) {
        seed = hash_combine(seed, v[i]);
    }
    return seed;
}

template <>
inline size_t get_array_hash<memory_desc_t>(
        size_t seed, const memory_desc_t *v, int size) {
    for (int i = 0; i < size; i++) {
        seed = hash_combine(seed, get_md_hash(v[i]));
    }
    return seed;
}

template <>
inline size_t get_array_hash<float>(size_t seed, const float *v, int size) {
    for (int i = 0; i < size; i++) {
        seed = hash_combine(seed, float2int(v[i]));
    }
    return seed;
}

template<typename T, typename A>
size_t get_vector_hash(size_t seed, const std::vector<T, A> &vec) {
    return get_array_hash(seed, vec.data(), vec.size());
}

} // namespace primitive_hashing
} // namespace impl
} // namespace dnnl

#endif
