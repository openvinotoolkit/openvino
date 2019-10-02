// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the BlobIterator class
 * @file ie_blob_iterator.hpp
 */

#include "ie_locked_memory.hpp"
#include <utility>

namespace InferenceEngine {
namespace details {
/**
 * @brief This class provides range loops support for TBlob objects
 */
template<class T>
class BlobIterator {
    LockedMemory<T> _mem;
    size_t _offset;

public:
    /**
     * @brief A move constructor to create a BlobIterator instance from a LockedMemory instance.
     * Explicitly rejects implicit conversions.
     * @param lk Rvalue of the memory instance to move from
     * @param offset Size of offset in memory
     */
    explicit BlobIterator(LockedMemory<T> &&lk, size_t offset = 0)
            : _mem(std::move(lk)), _offset(offset) {
    }

    /**
     * @brief Increments an offset of the current BlobIterator instance
     * @return The current BlobIterator instance
     */
    BlobIterator &operator++() {
        _offset++;
        return *this;
    }

    /**
     * @brief An overloaded postfix incrementation operator
     * Implementation does not follow std interface since only move semantics is used
     */
    void operator++(int) {
        _offset++;
    }

    /**
     * @brief Checks if the given iterator is not equal to the current one
     * @param that Iterator to compare with
     * @return true if the given iterator is not equal to the current one, false - otherwise
     */
    bool operator!=(const BlobIterator &that) const {
        return !operator==(that);
    }

    /**
     * @brief Gets a value by the pointer to the current iterator
     * @return The value stored in memory for the current offset value
     */
    const T &operator*() const {
        return *(_mem.template as<const T *>() + _offset);
    }

    /**
     * @brief Gets a value by the pointer to the current iterator
     * @return The value stored in memory for the current offset value
     */
    T &operator*() {
        return *(_mem.template as<T *>() + _offset);
    }
    /**
     * @brief Compares the given iterator with the current one
     * @param that Iterator to compare with
     * @return true if the given iterator is equal to the current one, false - otherwise
     */
    bool operator==(const BlobIterator &that) const {
        return &operator*() == &that.operator*();
    }
};
}  // namespace details
}  // namespace InferenceEngine
