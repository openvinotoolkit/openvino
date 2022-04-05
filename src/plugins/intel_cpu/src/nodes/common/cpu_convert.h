// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_precision.hpp>

namespace ov {
namespace intel_cpu {

/**
 * @brief Copy size elements from buffer specified srcPtr pointer to buffer specified dstPtr.
 * If the precisions srcPrc and dstPrc are different, a conversion from srcPrc to dstPrc is performed.
 * @param srcPtr
 * pointer to the buffer to convert from
 * @param dstPtr
 * pointer to the buffer to convert to
 * @param srcPrc
 * precision the buffer from which convert
 * @param dstPrc
 * precision the buffer to which convert
 * @param size
 * number of elements in buffers to be converted
 * @return none.
 */
void cpu_convert(const void *srcPtr,
                 void *dstPtr,
                 InferenceEngine::Precision srcPrc,
                 InferenceEngine::Precision dstPrc,
                 const size_t size);

/**
 * @brief Copy size elements from buffer specified srcPtr pointer to buffer specified dstPtr.
 * If the precisions srcPrc and dstPrc are different, a conversion from srcPrc to dstPrc is performed.
 * @param srcPtr
 * pointer to the buffer to convert from
 * @param dstPtr
 * pointer to the buffer to convert to
 * @param srcPrc
 * precision the buffer from which convert
 * @param interimPrc
 * intermediate precision used for type truncation
 * @param dstPrc
 * precision the buffer to which convert
 * @param size
 * number of elements in buffers to be converted
 * @return none.
 */
void cpu_convert(const void *srcPtr,
                 void *dstPtr,
                 InferenceEngine::Precision srcPrc,
                 InferenceEngine::Precision interimPrc,
                 InferenceEngine::Precision dstPrc,
                 const size_t size);

}   // namespace intel_cpu
}   // namespace ov
