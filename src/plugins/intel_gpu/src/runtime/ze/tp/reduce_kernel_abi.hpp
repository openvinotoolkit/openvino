// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace cldnn {
namespace ze {
namespace tp {

/// What the coordinator assumes about the reduction kernel the model brings
/// (ocl_v2/tp_allreduce.cl): `dst[i] = src0[i] + src1[i]` for i < n, in that
/// argument order, with each work item covering `elems_per_item` elements.
namespace reduce_kernel {

enum Arg : uint32_t { dst = 0, src0 = 1, src1 = 2, count = 3 };

/// Element count argument type (`ulong n`).
using count_t = uint64_t;

inline constexpr std::size_t elems_per_item = 8;
inline constexpr uint32_t group_size = 256;

}  // namespace reduce_kernel

}  // namespace tp
}  // namespace ze
}  // namespace cldnn
