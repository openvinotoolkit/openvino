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

#ifndef COMMON_PRIMITIVE_HASHING_HPP
#define COMMON_PRIMITIVE_HASHING_HPP

#include <typeindex>
#include <type_traits>

#include "primitive_hashing_utils.hpp"

#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
#include "engine_id.hpp"
#endif

namespace dnnl {
namespace impl {

struct primitive_desc_t;
namespace primitive_hashing {

struct key_t {
    key_t(const engine_t *engine, const op_desc_t *op_desc,
            const primitive_attr_t *attr, int pd_iterator_offset,
            const std::vector<memory_desc_t> &hint_mds);

    key_t(const primitive_desc_t *pd, const engine_t *engine);

    bool operator==(const key_t &other) const;
    const std::thread::id &thread_id() const { return thread_id_; }

    primitive_kind_t primitive_kind_;
    // Make these data fields mutable to be able to update them without removing
    // and adding a key (extract is available in C++17 only).
    mutable const op_desc_t *op_desc_;
    mutable const primitive_attr_t *attr_;
    int pd_iterator_offset_;
    int impl_nthr_;
    std::vector<memory_desc_t> hint_mds_;
#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
    engine_id_t engine_id_;
#else
    engine_kind_t engine_kind_;
    runtime_kind_t runtime_kind_;
    device_id_t device_id_;
#endif

private:
    template <typename desc_t>
    static const desc_t &cast_to_desc(const void *p) {
        return *(reinterpret_cast<const desc_t *>(p));
    }

    static primitive_kind_t get_pkind(primitive_kind_t pkind);

    // Thread ID is not used as part of the key, it's only used to get
    // information about what thread inserted the key and the corresponding
    // primitive to handle some multithreaded scenarios.
    std::thread::id thread_id_;
};

size_t get_desc_hash(const concat_desc_t &desc);
size_t get_desc_hash(const batch_normalization_desc_t &desc);
size_t get_desc_hash(const binary_desc_t &desc);
size_t get_desc_hash(const convolution_desc_t &desc);
size_t get_desc_hash(const eltwise_desc_t &desc);
size_t get_desc_hash(const gemm_desc_t &desc);
size_t get_desc_hash(const inner_product_desc_t &desc);
size_t get_desc_hash(const layer_normalization_desc_t &desc);
size_t get_desc_hash(const lrn_desc_t &desc);
size_t get_desc_hash(const matmul_desc_t &desc);
size_t get_desc_hash(const pooling_desc_t &desc);
size_t get_desc_hash(const pooling_v2_desc_t &desc);
size_t get_desc_hash(const prelu_desc_t &desc);
size_t get_desc_hash(const reduction_desc_t &desc);
size_t get_desc_hash(const reorder_desc_t &desc);
size_t get_desc_hash(const resampling_desc_t &desc);
size_t get_desc_hash(const rnn_desc_t &desc);
size_t get_desc_hash(const shuffle_desc_t &desc);
size_t get_desc_hash(const softmax_desc_t &desc);
size_t get_desc_hash(const sum_desc_t &desc);
size_t get_desc_hash(const zero_pad_desc_t &desc);

} // namespace primitive_hashing
} // namespace impl
} // namespace dnnl

// inject a specialization of std::hash for key_t in std namespace
namespace std {
template <>
struct hash<dnnl::impl::primitive_hashing::key_t> {
    using argument_type = dnnl::impl::primitive_hashing::key_t;
    using result_type = std::size_t;
    result_type operator()(const argument_type &key) const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;
        size_t seed = 0;
        // Compute hash for primitive_kind_, attr_, impl_id_ and impl_nthr_
        seed = hash_combine(seed,
                hash_combine(0, static_cast<size_t>(key.primitive_kind_)));
        seed = hash_combine(seed, get_attr_hash(*key.attr_));
        seed = hash_combine(seed, hash_combine(0, key.pd_iterator_offset_));
        seed = hash_combine(seed, hash_combine(0, key.impl_nthr_));

#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
        seed = hash_combine(seed, key.engine_id_.hash());
#else
        seed = hash_combine(
                seed, hash_combine(0, static_cast<size_t>(key.engine_kind_)));
        seed = hash_combine(
                seed, hash_combine(0, static_cast<size_t>(key.runtime_kind_)));
        seed = hash_combine(seed, hash_combine(0, std::get<0>(key.device_id_)));
        seed = hash_combine(seed, hash_combine(0, std::get<1>(key.device_id_)));
        seed = hash_combine(seed, hash_combine(0, std::get<2>(key.device_id_)));
#endif
        // Combine hash for op_desc with the computed hash
#define CASE(pkind) \
    case primitive_kind::pkind: \
        seed = hash_combine( \
                seed, get_desc_hash(*(pkind##_desc_t *)key.op_desc_)); \
        break;

        // clang-format off
        switch ((int)key.primitive_kind_) {
            CASE(batch_normalization)
            CASE(binary)
            CASE(concat)
            CASE(convolution)
            CASE(deconvolution)
            CASE(eltwise)
            CASE(gemm)
            CASE(inner_product)
            CASE(layer_normalization)
            CASE(lrn)
            CASE(matmul)
            CASE(pooling)
            CASE(pooling_v2)
            CASE(prelu)
            CASE(reduction)
            CASE(reorder)
            CASE(resampling)
            CASE(rnn)
            CASE(shuffle)
            CASE(softmax)
            CASE(sum)
            CASE(zero_pad)
            default: assert(!"unknown primitive_kind");
        }
            // clang-format on
#undef CASE
        seed = get_array_hash(
                seed, key.hint_mds_.data(), (int)key.hint_mds_.size());

        return seed;
    }
};

} // namespace std

#endif
