/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
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

/// @file
/// C++ API

#pragma once

#ifdef _WIN32
#include "../../../common/core/cm/base_ops.hpp"
#include "../../../common/core/cm/base_types.hpp"
#include "../../../common/core/cm/common.hpp"
#include "../../../common/utils/cm/limitation.hpp"

#else
#include "common/core/cm/base_ops.hpp"
#include "common/core/cm/base_types.hpp"
#include "common/core/cm/common.hpp"
#include "common/utils/cm/limitation.hpp"
#endif

namespace gpu::xetla {

namespace detail {

/// @brief lookup table for cache hint.
///
///
constexpr CacheHint get_cache_hint(gpu::xetla::cache_hint ch) {
    switch (ch) {
        case gpu::xetla::cache_hint::none: return CacheHint::Default;
        case gpu::xetla::cache_hint::uncached: return CacheHint::Uncached;
        case gpu::xetla::cache_hint::cached: return CacheHint::Cached;
        case gpu::xetla::cache_hint::write_back: return CacheHint::WriteBack;
        case gpu::xetla::cache_hint::write_through:
            return CacheHint::WriteThrough;
        case gpu::xetla::cache_hint::streaming: return CacheHint::Streaming;
        case gpu::xetla::cache_hint::read_invalidate:
            return CacheHint::ReadInvalidate;
    }
}

/// @brief lookup table for data size.
///
///
constexpr DataSize get_data_size(gpu::xetla::data_size ds) {
    switch (ds) {
        case gpu::xetla::data_size::default_size: return DataSize::Default;
        case gpu::xetla::data_size::u8: return DataSize::U8;
        case gpu::xetla::data_size::u16: return DataSize::U16;
        case gpu::xetla::data_size::u32: return DataSize::U32;
        case gpu::xetla::data_size::u64: return DataSize::U64;
        case gpu::xetla::data_size::u8u32: return DataSize::U8U32;
        case gpu::xetla::data_size::u16u32: return DataSize::U16U32;
        case gpu::xetla::data_size::u16u32h: return DataSize::U16U32H;
    }
}

/// @brief lookup table for memory kind.
///
///
constexpr LSC_SFID get_memory_kind(gpu::xetla::memory_kind mk) {
    switch (mk) {
        case gpu::xetla::memory_kind::untyped_global: return LSC_SFID::LSC_UGM;
        case gpu::xetla::memory_kind::untyped_global_low_pri:
            return LSC_SFID::LSC_UGML;
        case gpu::xetla::memory_kind::typed_global: return LSC_SFID::LSC_TGM;
        case gpu::xetla::memory_kind::shared_local: return LSC_SFID::LSC_SLM;
    }
}

/// @brief lookup table for fence op.
///
///
constexpr LSC_FENCE_OP get_fence_op(gpu::xetla::fence_op fo) {
    switch (fo) {
        case gpu::xetla::fence_op::none: return LSC_FENCE_OP::LSC_FENCE_OP_NONE;
        case gpu::xetla::fence_op::evict:
            return LSC_FENCE_OP::LSC_FENCE_OP_EVICT;
        case gpu::xetla::fence_op::invalidate:
            return LSC_FENCE_OP::LSC_FENCE_OP_INVALIDATE;
        case gpu::xetla::fence_op::discard:
            return LSC_FENCE_OP::LSC_FENCE_OP_DISCARD;
        case gpu::xetla::fence_op::clean:
            return LSC_FENCE_OP::LSC_FENCE_OP_CLEAN;
        case gpu::xetla::fence_op::flushl2:
            return LSC_FENCE_OP::LSC_FENCE_OP_FLUSHL3;
    }
}

/// @brief lookup table for fence scope.
///
///
constexpr LSC_SCOPE get_fence_scope(gpu::xetla::fence_scope fs) {
    switch (fs) {
        case gpu::xetla::fence_scope::group: return LSC_SCOPE::LSC_SCOPE_GROUP;
        case gpu::xetla::fence_scope::local: return LSC_SCOPE::LSC_SCOPE_LOCAL;
        case gpu::xetla::fence_scope::tile: return LSC_SCOPE::LSC_SCOPE_TILE;
        case gpu::xetla::fence_scope::gpu: return LSC_SCOPE::LSC_SCOPE_GPU;
        case gpu::xetla::fence_scope::gpus: return LSC_SCOPE::LSC_SCOPE_GPUS;
        case gpu::xetla::fence_scope::system:
            return LSC_SCOPE::LSC_SCOPE_SYSTEM;
        case gpu::xetla::fence_scope::sysacq:
            return LSC_SCOPE::LSC_SCOPE_SYSACQ;
    }
}

/// @brief lookup table for atomic op.
///
///
constexpr AtomicOp get_atomic_op(gpu::xetla::atomic_op ao) {
    switch (ao) {
        case gpu::xetla::atomic_op::iinc: return AtomicOp::IINC;
        case gpu::xetla::atomic_op::idec: return AtomicOp::IDEC;
        case gpu::xetla::atomic_op::iadd: return AtomicOp::IADD;
        case gpu::xetla::atomic_op::isub: return AtomicOp::ISUB;
        case gpu::xetla::atomic_op::smin: return AtomicOp::SMIN;
        case gpu::xetla::atomic_op::smax: return AtomicOp::SMAX;
        case gpu::xetla::atomic_op::cmpxchg: return AtomicOp::ICAS;
        case gpu::xetla::atomic_op::fadd: return AtomicOp::FADD;
        case gpu::xetla::atomic_op::fsub: return AtomicOp::FSUB;
        case gpu::xetla::atomic_op::fmin: return AtomicOp::FMIN;
        case gpu::xetla::atomic_op::fmax: return AtomicOp::FMAX;
        case gpu::xetla::atomic_op::fcmpxchg: return AtomicOp::FCAS;
        case gpu::xetla::atomic_op::umin: return AtomicOp::UMIN;
        case gpu::xetla::atomic_op::umax: return AtomicOp::UMAX;
        case gpu::xetla::atomic_op::bit_and: return AtomicOp::AND;
        case gpu::xetla::atomic_op::bit_or: return AtomicOp::OR;
        case gpu::xetla::atomic_op::bit_xor: return AtomicOp::XOR;
        case gpu::xetla::atomic_op::load: return AtomicOp::LOAD;
        case gpu::xetla::atomic_op::store: return AtomicOp::STORE;
        default: return AtomicOp::STORE;
    }
}
} // namespace detail

/// @addtogroup xetla_core_memory
/// @{

/// @brief Stateless scattered prefetch.
/// Prefetches elements located at specified address.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.ugm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to prefetch per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets in bytes.
/// @param pred    [in] is predicates.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::cached,
        cache_hint L2H = cache_hint::cached, int N>
__XETLA_API void xetla_prefetch_global(
        Ty *p, xetla_vector<uint32_t, N> offsets, xetla_mask<N> pred = 1) {
    using T = native_type_t<Ty>;
    constexpr DataSize _DS = details::lsc_expand_ds(
            details::lsc_data_size<T, gpu::xetla::detail::get_data_size(DS)>());
    cm_ptr_prefetch<details::lsc_vector_size<NElts>(), _DS,
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L2H), N>(
            (const unsigned *const)p, offsets, pred);
}

/// @brief Stateless block prefetch (transposed gather with 1 channel).
/// Prefetches elements located at specified address.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.ugm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to prefetch per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @param p      [in] is the base pointer.
/// @param offset [in] is the zero-based offset in bytes.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::cached,
        cache_hint L2H = cache_hint::cached>
__XETLA_API void xetla_prefetch_global(Ty *p, uint64_t offset = 0) {
    using T = native_type_t<Ty>;
    constexpr DataSize _DS = details::lsc_expand_ds(
            details::lsc_data_size<T, gpu::xetla::detail::get_data_size(DS)>());
    cm_ptr_prefetch<NElts, _DS, gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L2H)>(
            (const unsigned *const)p, offset);
}

/// @brief Stateless scattered load.
/// Collects elements located at specified address and returns them
/// to a single \ref xetla_vector object.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.ugm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to load per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets in bytes.
/// @param pred    [in] is predicates.
/// @return  is a xetla_vector of type T and size N * NElts.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L2H = cache_hint::none,
        int N, typename Toffset = uint32_t>
__XETLA_API xetla_vector<Ty, N * NElts> xetla_load_global(
        Ty *p, xetla_vector<Toffset, N> offsets, xetla_mask<N> pred = 1) {
    using T = native_type_t<Ty>;
    return cm_ptr_load<T, details::lsc_vector_size<NElts>(),
            gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L2H), N>((T *)p, offsets, pred);
}

/// @brief Stateless block load (transposed gather with 1 channel).
/// Collects elements located at specified address and returns them
/// to a single \ref xetla_vector object.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.ugm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to load per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @param p      [in] is the base pointer.
/// @param offset [in] is the zero-based offset in bytes.
/// @return is a xetla_vector of type T and size NElts.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L2H = cache_hint::none>
__XETLA_API xetla_vector<Ty, NElts> xetla_load_global(
        Ty *p, uint64_t offset = 0) {
    using T = native_type_t<Ty>;
    return cm_ptr_load<T, NElts, gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L2H)>((T *)p, offset);
}

/// @brief Stateless scattered store.
/// Writes elements to specific address.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_store.ugm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to store per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets in bytes.
/// @param vals    [in] is values to store.
/// @param pred    [in] is predicates.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L2H = cache_hint::none,
        int N>
__XETLA_API void xetla_store_global(Ty *p, xetla_vector<uint32_t, N> offsets,
        xetla_vector<Ty, N * NElts> vals, xetla_mask<N> pred = 1) {
    using T = native_type_t<Ty>;
    cm_ptr_store<T, details::lsc_vector_size<NElts>(),
            gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L2H), N>(
            (T *)p, offsets, vals, pred);
}

/// @brief Stateless block store (transposed scatter with 1 channel).
/// Writes elements to specific address.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_store.ugm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to store per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @param p      [in] is the base pointer.
/// @param offset [in] is the zero-based offset in bytes.
/// @param vals   [in] is values to store.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L2H = cache_hint::none>
__XETLA_API void xetla_store_global(
        Ty *p, uint64_t offset, xetla_vector<Ty, NElts> vals) {
    using T = native_type_t<Ty>;
    cm_ptr_store<T, NElts, gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L2H)>((T *)p, offset, vals);
}

/// @brief Stateless scattered atomic (0 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets.
/// @param pred    [in] is predicates.
///
template <atomic_op Op, typename T, int N,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L2H = cache_hint::none>
__XETLA_API xetla_vector<T, N> xetla_atomic_global(
        T *p, xetla_vector<uint32_t, N> offsets, xetla_mask<N> pred) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_ptr_atomic<gpu::xetla::detail::get_atomic_op(Op), T,
            VectorSize::N1, gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L2H)>(p, offsets, pred);
}

/// @brief Stateless scattered atomic (1 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets.
/// @param src0    [in] is the first atomic operand.
/// @param pred    [in] is predicates.
///
template <atomic_op Op, typename T, int N,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L2H = cache_hint::none>
__XETLA_API xetla_vector<T, N> xetla_atomic_global(T *p,
        xetla_vector<uint32_t, N> offsets, xetla_vector<T, N> src0,
        xetla_mask<N> pred) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_ptr_atomic<gpu::xetla::detail::get_atomic_op(Op), T,
            VectorSize::N1, gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L2H)>(p, offsets, src0, pred);
}

/// @brief Stateless scattered atomic (2 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets.
/// @param src0    [in] is the first atomic operand.
/// @param src1    [in] is the second atomic operand.
/// @param pred    [in] is predicates.
///
template <atomic_op Op, typename T, int N,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L2H = cache_hint::none>
__XETLA_API xetla_vector<T, N> xetla_atomic_global(T *p,
        xetla_vector<uint32_t, N> offsets, xetla_vector<T, N> src0,
        xetla_vector<T, N> src1, xetla_mask<N> pred) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_ptr_atomic<gpu::xetla::detail::get_atomic_op(Op), T,
            VectorSize::N1, gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L2H)>(
            p, offsets, src0, src1, pred);
}
/// @brief Declare per-work-group slm size.
/// @tparam SLMSize  Shared Local Memory (SLM) size (in Bytes).
template <uint32_t SLMSize>
__XETLA_API void xetla_local_init() {
    cm_slm_init(SLMSize);
}

/// @brief SLM scattered load.
/// Collects elements located at slm and returns them as a single \ref xetla_vector object.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.slm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to load per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param offsets [in] is the zero-based offsets for SLM buffer in bytes.
/// @param pred    [in] is predicates.
/// @return is a xetla_vector of type T and size N * NElts.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size, int N>
__XETLA_API xetla_vector<Ty, N * NElts> xetla_load_local(
        xetla_vector<uint32_t, N> offsets, xetla_mask<N> pred = 1) {
    using T = native_type_t<Ty>;
    return cm_load_slm<T, details::lsc_vector_size<NElts>(),
            gpu::xetla::detail::get_data_size(DS), N>(offsets, pred);
}

/// @brief SLM block load. (transposed gather with 1 channel).
/// Collects elements located at slm and returns them as a single \ref xetla_vector object.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.slm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to load per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @param offset [in] is the zero-based offset for SLM buffer in bytes.
/// @return is a xetla_vector of type T and size NElts.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size>
__XETLA_API xetla_vector<Ty, NElts> xetla_load_local(uint32_t offset) {
    using T = native_type_t<Ty>;
    return cm_load_slm<T, NElts, gpu::xetla::detail::get_data_size(DS)>(offset);
}

/// @brief SLM scattered store.
/// Scatters elements located to slm.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_store.slm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to store per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param offsets [in] is the zero-based offsets for SLM buffer in bytes.
/// @param vals    [in] is values to store.
/// @param pred    [in] is predicates.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size, int N>
__XETLA_API void xetla_store_local(xetla_vector<uint32_t, N> offsets,
        xetla_vector<Ty, N * NElts> vals, xetla_mask<N> pred = 1) {
    using T = native_type_t<Ty>;
    cm_store_slm<T, details::lsc_vector_size<NElts>(),
            gpu::xetla::detail::get_data_size(DS), N>(offsets, vals, pred);
}

/// @brief SLM block store (transposed SLM scatter with 1 channel).
/// Scatters elements located to slm.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_store.slm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to store per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @param offset [in] is the zero-based offset for SLM buffer in bytes.
/// @param vals   [in] is values to store.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size>
__XETLA_API void xetla_store_local(
        uint32_t offset, xetla_vector<Ty, NElts> vals) {
    using T = native_type_t<Ty>;
    cm_store_slm<T, NElts, gpu::xetla::detail::get_data_size(DS)>(offset, vals);
}

/// @brief SLM scattered atomic (0 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @param offsets [in] is the zero-based offsets.
/// @param pred    [in] is predicates.
///
template <atomic_op Op, typename T, int N,
        data_size DS = data_size::default_size>
__XETLA_API xetla_vector<T, N> xetla_atomic_local(
        xetla_vector<uint32_t, N> offsets, xetla_mask<N> pred) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_atomic_slm<gpu::xetla::detail::get_atomic_op(Op), T,
            VectorSize::N1, gpu::xetla::detail::get_data_size(DS),
            CacheHint::Default, CacheHint::Default, N>(offsets, pred);
}

/// @brief SLM scattered atomic (1 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @param offsets [in] is the zero-based offsets.
/// @param src0    [in] is the first atomic operand.
/// @param pred    [in] is predicates.
///
template <atomic_op Op, typename T, int N,
        data_size DS = data_size::default_size>
__XETLA_API xetla_vector<T, N> xetla_atomic_local(
        xetla_vector<uint32_t, N> offsets, xetla_vector<T, N> src0,
        xetla_mask<N> pred) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_atomic_slm<gpu::xetla::detail::get_atomic_op(Op), T,
            VectorSize::N1, gpu::xetla::detail::get_data_size(DS),
            CacheHint::Default, CacheHint::Default, N>(offsets, src0, pred);
}

/// @brief SLM scattered atomic (2 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @param offsets [in] is the zero-based offsets.
/// @param src0    [in] is the first atomic operand.
/// @param src1    [in] is the second atomic operand.
/// @param pred    [in] is predicates.
///
template <atomic_op Op, typename T, int N,
        data_size DS = data_size::default_size>
__XETLA_API xetla_vector<T, N> xetla_atomic_local(
        xetla_vector<uint32_t, N> offsets, xetla_vector<T, N> src0,
        xetla_vector<T, N> src1, xetla_mask<N> pred) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_atomic_slm<gpu::xetla::detail::get_atomic_op(Op), T,
            VectorSize::N1, gpu::xetla::detail::get_data_size(DS),
            CacheHint::Default, CacheHint::Default, N>(
            offsets, src0, src1, pred);
}

/// @brief Memory fence.
/// Supported platforms: DG2, PVC
///
/// @tparam Kind is the Sfid shaded function.
/// @tparam FenceOp is the fence operation.
/// @tparam Scope is the operation scope.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param pred is predicates.
template <memory_kind Kind = memory_kind::untyped_global,
        fence_op FenceOp = fence_op::none,
        fence_scope Scope = fence_scope::group, int N = 16>
__XETLA_API void xetla_fence(xetla_mask<N> pred = 1) {
    cm_fence<gpu::xetla::detail::get_memory_kind(Kind),
            gpu::xetla::detail::get_fence_op(FenceOp),
            gpu::xetla::detail::get_fence_scope(Scope), N>();
}

/// @} xetla_core_memory

} // namespace gpu::xetla
