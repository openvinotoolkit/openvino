// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <numeric>
#include <vector>

#include "ie_common.h"
#include "ie_layouts.h"
#include "general_utils.h"
#include "precision_support.h"

namespace ov {
namespace intel_cpu {

// helper struct to tell wheter type T is any of given types U...
// termination case when U... is empty -> return std::false_type
template <class T, class... U>
struct is_any_of : public std::false_type {};

// helper struct to tell whether type is any of given types (U, Rest...)
// recurrence case when at least one type U is present -> returns std::true_type if std::same<T, U>::value is true,
// otherwise call is_any_of<T, Rest...> recurrently
template <class T, class U, class... Rest>
struct is_any_of<T, U, Rest...>
        : public std::conditional<std::is_same<T, U>::value, std::true_type, is_any_of<T, Rest...>>::type {};

/**
* @brief Returns normalized by size dims where missing dimensions are filled with units from the beginning
* Example: dims = {2, 3, 5}; ndims = 5; result = {1, 1, 2, 3, 5}
* @param dims
* shape to normalize
* @param ndims
* rank of resulting shape
* @return normalized vector
*/
inline std::vector<size_t> getNormalizedDimsBySize(const InferenceEngine::SizeVector &dims, size_t ndims) {
    if (dims.size() >= ndims)
        return dims;

    std::vector<size_t> normalizedDims = dims;
    for (size_t i = 0; i < (ndims - dims.size()); i++) {
        normalizedDims.insert(normalizedDims.begin(), 1);
    }
    return normalizedDims;
}

/**
* @brief Checked that secondInputDims unidirectional broadcastable per tensor or per channel to firstInputDims
* @param firstInputDims
* shape on which should be broadcastable
* @param secondInputDims
* shape which should be broadcastable
* @param weakComparison
* flag which specify how we compare C dims if value is undefined (weak or strong)
* @return true if broadcastable, false otherwise.
*/
inline bool isPerTensorOrPerChannelBroadcastable(const InferenceEngine::SizeVector &firstInputDims,
                                                 const InferenceEngine::SizeVector& secondInputDims,
                                                 int channelAxis,
                                                 bool weakComparison = false) {
    bool (*dimsEqual)(size_t, size_t) = weakComparison ? static_cast<bool (*)(size_t, size_t)>(dimsEqualWeak) :
                                                         static_cast<bool (*)(size_t, size_t)>(dimsEqualStrong);
    if (secondInputDims.size() > firstInputDims.size())
        return false;
    if (std::accumulate(secondInputDims.begin(), secondInputDims.end(), size_t(1), std::multiplies<size_t>()) == 1)
        return true;

    std::vector<size_t> normalizedSecondInputDims = getNormalizedDimsBySize(secondInputDims, firstInputDims.size());
    if (channelAxis >= 0) {
        for (size_t i = 0; i < normalizedSecondInputDims.size(); i++) {
            if ((i == static_cast<size_t>(channelAxis) &&
                 !dimsEqual(normalizedSecondInputDims[i], firstInputDims[i])) ||
                (i != static_cast<size_t>(channelAxis) && normalizedSecondInputDims[i] != 1))
                return false;
        }
    } else {
        for (size_t i = 0; i < normalizedSecondInputDims.size(); i++) {
            if (normalizedSecondInputDims[i] != 1)
                return false;
        }
    }
    return true;
}

inline bool isEmptyTensorDesc(const InferenceEngine::TensorDesc &td) {
    const auto dims = td.getDims();
    return std::any_of(dims.begin(), dims.end(), [](size_t dim) { return dim == 0; } );
}

/**
* @brief Return precision to which given precision must be converted to be supported in plug-in
* @param precision
* precision for convert
* @return plug-in supported precision or UNSPECIFIED if precision unsupported
*/
inline ov::element::Type normalizeToSupportedPrecision(ov::element::Type precision) {
    switch (precision) {
        case ov::element::bf16:
        case ov::element::f16: {
            if (!hasHardwareSupport(precision))
                precision = ov::element::f32;
        }
        case ov::element::u8:
        case ov::element::i8:
        case ov::element::i32:
        case ov::element::f32: {
            break;
        }
        case ov::element::f64: {
            precision = ov::element::f32;
            break;
        }
        case ov::element::boolean: {
            precision = ov::element::u8;
            break;
        }
        case ov::element::u16:
        case ov::element::i16:
        case ov::element::u32:
        case ov::element::i64:
        case ov::element::u64: {
            precision = ov::element::i32;
            break;
        }
        default: {
            precision = ov::element::undefined;
        }
    }

    return precision;
}

/**
* @brief Return aligned buffer by targetSize.
* If buffer has size 1, values are broadcasted with targetSize size.
* If aligned buffer size > targetSize, other values filled by zero.
* @param targetSize
* target size buffer
* @param buffer
* buffer to be aligned
* @param align
* alignment for targetSize
* @return aligned buffer
*/
inline std::vector<float> makeAlignedBuffer(size_t targetSize, const std::vector<float> &buffer, int align = -1) {
    if (buffer.empty()) {
        IE_THROW() << "Can't align buffer, becuase buffer is empty";
    }

    auto alignedBuffer = buffer;
    if (align == -1) {
        align = targetSize;
    }
    const size_t bufferSizeAligned = rnd_up(targetSize, align);

    alignedBuffer.resize(bufferSizeAligned, 0);
    if (buffer.size() == 1) {
        std::fill(alignedBuffer.begin() + 1, alignedBuffer.begin() + targetSize, buffer[0]);
    }
    return alignedBuffer;
}
}   // namespace intel_cpu
}   // namespace ov
