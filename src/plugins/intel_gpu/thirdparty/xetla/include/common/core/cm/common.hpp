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

#ifndef XETLA_NO_CM_INCLUDE
#include <cm/cm.h>
#include <cm/cmtl.h>
#endif

template <class T>
using remove_const_t = typename std::remove_const<T>::type;

/// @addtogroup xetla_core
/// @{

/// @brief KERNEL_MAIN macro.
/// Alias to CM `"_GENX_MAIN_"`.
///
#define KERNEL_MAIN _GENX_MAIN_

/// @brief KERNEL_FUNC macro.
/// Alias to empty.
///
#define KERNEL_FUNC

/// @} xetla_core

#define __XETLA_API inline

#define XETLA_WARNING(msg) CM_STATIC_WARNING(0, msg)

#define XETLA_MARKER(message) [[deprecated(message)]]

#define DEVICE_PRINTF(s, ...) \
    do { \
    } while (0)

#define DEVICE_ASSERT(c, s, ...) \
    do { \
    } while (0);

template <auto val>
XETLA_MARKER("Help function to print value")
inline constexpr void XETLA_PRINT() {}
template <typename type>
XETLA_MARKER("Help function to print type")
inline constexpr void XETLA_PRINT() {}

namespace gpu::xetla {

enum class gpu_arch : uint8_t { Xe = 0 };
enum class grf_mode : uint8_t { normal = 0, double_grf = 1 };

enum class mem_layout : uint8_t {
    row_major = 0,
    col_major = 1,
    nhwc = 2,
    hwio = 3,
};
enum class mem_space : uint8_t { global = 0, local = 1 };
enum class msg_type : uint8_t {
    block_2d = 0,
    block_1d = 1,
    scatter = 2,
    atomic_add = 3,
    unaligned_2d = 4
    // prefetch_2d = 4,
    // prefetch_1d = 5
};
/// L1 or L2 cache hint kinds.
enum class cache_hint : uint8_t {
    none = 0,
    uncached = 1,
    cached = 2,
    write_back = 3,
    write_through = 4,
    streaming = 5,
    read_invalidate = 6
};

/// Data size or format to read or store
enum class data_size : uint8_t {
    default_size = 0,
    u8 = 1,
    u16 = 2,
    u32 = 3,
    u64 = 4,
    u8u32 = 5, /// load 8b, zero extend to 32b; store the opposite
    u16u32 = 6, /// load 16b, zero extend to 32b; store the opposite
    u16u32h = 7, /// load 16b into high 16 of each 32b; store the high 16
};

/// The specific LSC shared function to fence with xetla_fence
enum class memory_kind : uint8_t {
    untyped_global = 0, /// untyped global memory
    untyped_global_low_pri = 1, /// low-priority untyped global memory
    typed_global = 2, /// typed global memory
    shared_local = 3, /// shared local memory
};

/// The xetla_fence operation to apply to caches
enum class fence_op : uint8_t {
    none = 0, /// no operation
    evict = 1, /// dirty lines evicted and invalidated from L1
    invalidate = 2, /// invalidate all clean lines
    discard = 3, /// direct and clean lines are discarded w/o eviction
    clean = 4, /// dirty lines are written to memory, but retained in cache
    /// in clean state
    flushl2 = 5, /// flush only L2
};
/// The scope that xetla_fence operation should apply to
enum class fence_scope : uint8_t {
    group = 0, /// flush out to the threadgroup's scope
    local = 1, /// flush out to the local scope
    tile = 2, /// tile, flush out to several DSSs
    gpu = 3, /// entire GPU, flush out to the GPUs LLC
    gpus = 4, /// all GPUs in the system, flush out to memory shared by all GPUs
    system = 5, /// the entire system memory space
    sysacq = 6, /// the entire system memory space with system-acquire semantics
};

/// Represents an atomic operation. Operations always return the old value(s) of
/// the target memory location(s) as it was before the operation was applied.
enum class atomic_op : uint8_t {
    /// Atomic increment of memory data and return the old value.
    iinc = 0x0,
    /// Atomic decrement of memory data and return the old value.
    idec = 0x1,
    /// Atomic signed int add of src1 from memory data and return the old value.
    iadd = 0x2,
    /// Atomic signed int subtract of src1 from memory data and return the old value.
    isub = 0x3,
    /// Atomic store the signed int min of src1 and memory data and return the old value.
    smin = 0x4,
    /// Atomic store the signed int max of src1 and memory data and return the old value.
    smax = 0x5,
    /// Atomic bit-compare src1_X and memory data and replace if equal with src1_Y. Returns the old value.
    cmpxchg = 0x6,
    /// Atomic float add of src1 from memory data and return the old value.
    fadd = 0x7,
    /// Atomic float subtract of src1 from memory data and return the old value.
    fsub = 0x8,
    /// Atomic store the float min of src1 and memory data and return the old value.
    fmin = 0x9,
    /// Atomic store the float max of src1 and memory data and return the old value.
    fmax = 0xa,
    /// Atomic float compare src1_X and memory data and replace if equal with src1_Y. Returns the old value.
    fcmpxchg = 0xb,
    /// Atomic store the unsigned int min of src1 and memory data and return the old value.
    umin = 0xc,
    /// Atomic store the unsigned int max of src1 and memory data and return the old value.
    umax = 0xd,
    /// Atomic store the bitwise AND of src1 and memory data and return the old value.
    bit_and = 0xe,
    /// Atomic store the bitwise OR of src1 and memory data and return the old value.
    bit_or = 0xf,
    /// Atomic store the bitwise XOR of src1 and memory data and return the old value.
    bit_xor = 0x10,
    /// Atomic read of the memory data value, without modifying the data.
    load = 0x11,
    /// Atomic store untyped data to memory.
    store = 0x12
};

/// xetla dpas argument typ
enum class argument_type : uint8_t {
    U2 = 2, // unsigned 2 bits
    S2 = 3, // signed 2 bits
    U4 = 4, // unsigned 4 bits
    S4 = 5, // signed 4 bits
    U8 = 6, // unsigned 8 bits
    S8 = 7, // signed 8 bits
    BF16 = 8, // bfloat 16
    FP16 = 9, // half float
    TF32 = 12, // tensorfloat 32
    DF = 13, // double (64bits)
    NUM_ARG_TYPES = 14
};

// Saturation tag
class xetla_saturation_on_tag {
public:
    static constexpr int value = _GENX_SAT;
};

class xetla_saturation_off_tag {
public:
    static constexpr int value = _GENX_NOSAT;
};

template <typename T>
using is_xetla_scalar = typename details::is_cm_scalar<T>;

/// xetla reduce op
enum class reduce_op : uint8_t {
    sum = 0, // performance reduce_sum
    prod = 1, // performance reduce_prod
    min = 2, // performance reduce_min
    max = 3, // performance reduce_max
};

/// SW_BARRIER, insert software scheduling barrier, for better code control
///
#define SW_BARRIER() cm_fence(CM_SW_BARRIER)

__XETLA_API void xetla_wait(uint16_t val) {
    __cm_builtin_dummy_mov(val);
}

} // namespace gpu::xetla
