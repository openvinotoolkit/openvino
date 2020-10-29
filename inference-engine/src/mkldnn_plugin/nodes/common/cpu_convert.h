// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_precision.hpp>

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

void cpu_convert(void *srcPtr, void *dstPtr, InferenceEngine::Precision srcPrc, InferenceEngine::Precision dstPrc, const size_t size);
