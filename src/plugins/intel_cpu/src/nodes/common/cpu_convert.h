// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

/**
 * @brief Sequential precision conversion. Copies `size` elements from `srcPtr` to
 * `dstPtr`; if `srcPrc` and `dstPrc` differ, the conversion is performed element-wise.
 * Use this variant when the call site already runs inside an outer parallel region
 * (e.g. inside parallel_for / parallel_for2d) to avoid nested parallelism.
 *
 * Mirrors the cpu_memcpy / cpu_parallel_memcpy split.
 *
 * @param srcPtr  pointer to the buffer to convert from
 * @param dstPtr  pointer to the buffer to convert to
 * @param srcPrc  precision of the source buffer
 * @param dstPrc  precision of the destination buffer
 * @param size    number of elements in the buffers to be converted
 */
void cpu_convert(const void* srcPtr, void* dstPtr, ov::element::Type srcPrc, ov::element::Type dstPrc, size_t size);

/**
 * @copydoc cpu_convert(const void*, void*, ov::element::Type, ov::element::Type, size_t)
 *
 * @param interimPrc  intermediate precision used as a bridge when no direct
 *                    `srcPrc -> dstPrc` kernel exists
 */
void cpu_convert(const void* srcPtr,
                 void* dstPtr,
                 ov::element::Type srcPrc,
                 ov::element::Type interimPrc,
                 ov::element::Type dstPrc,
                 size_t size);

/**
 * @brief Parallel precision conversion. Same semantics as cpu_convert but internally
 * dispatches the loop body via parallel_for. Use this variant for top-level conversions
 * not nested inside an outer parallel region.
 *
 * @param srcPtr  pointer to the buffer to convert from
 * @param dstPtr  pointer to the buffer to convert to
 * @param srcPrc  precision of the source buffer
 * @param dstPrc  precision of the destination buffer
 * @param size    number of elements in the buffers to be converted
 */
void cpu_parallel_convert(const void* srcPtr,
                          void* dstPtr,
                          ov::element::Type srcPrc,
                          ov::element::Type dstPrc,
                          size_t size);

/**
 * @copydoc cpu_parallel_convert(const void*, void*, ov::element::Type, ov::element::Type, size_t)
 *
 * @param interimPrc  intermediate precision used as a bridge when no direct
 *                    `srcPrc -> dstPrc` kernel exists
 */
void cpu_parallel_convert(const void* srcPtr,
                          void* dstPtr,
                          ov::element::Type srcPrc,
                          ov::element::Type interimPrc,
                          ov::element::Type dstPrc,
                          size_t size);

bool is_supported_convert(ov::element::Type srcPrc, ov::element::Type dstPrc);

}  // namespace ov::intel_cpu
