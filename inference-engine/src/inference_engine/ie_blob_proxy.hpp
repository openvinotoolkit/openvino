// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @brief A header file for the TBlobProxy class definition
 * @file ie_blob_proxy.hpp
 */

#include "ie_blob.h"
#include <utility>
#include <memory>

namespace InferenceEngine {

/**
 * @class TBlobProxy
 * @brief This class enables creation of several blobs based on a single allocation but using different offsets for read/write
 */
template<class T>
class TBlobProxy : public TBlob<T> {
    using base = TBlob<T>;

public:
    /**
     * @brief A shared pointer to the TBlobProxy object
     */
    using Ptr = std::shared_ptr<TBlobProxy<T>>;

    /**
     * @brief A move constructor
     * @param p Precision type
     * @param l Layout
     * @param blob Source TBlob object to move from. It is deleted after a constructor call, since the ownership is transferred to the proxy
     * @param offset Offset in memory
     * @param dims Dimensions of the given blob
     */
    TBlobProxy(Precision p, Layout l, TBlob <T> &&blob, size_t offset, const SizeVector &dims)
            : base(p, l, dims), realObject(make_shared_blob<T>(std::move(blob))),
              offset(offset * blob.element_size()) {
        checkWindow();
    }

    /**
     * @brief A move constructor
     * @param p Precision type
     * @param l Layout
     * @param blob Source Blob object to move from. It is deleted after a constructor call, since the ownership is transferred to the proxy
     * @param offset Offset in memory
     * @param dims Dimensions of the given blob
     */
    TBlobProxy(Precision p, Layout l, const Blob::Ptr &blob, size_t offset, const SizeVector &dims)
            : base(p, l, dims), realObject(blob), offset(offset * blob->element_size()) {
        checkWindow();
    }

    /**
     * A copy constructor
     * @param p Precision type
     * @param l Layout
     * @param blobProxy Source TBlobProxy object to copy from
     * @param offset Offset in memory
     * @param dims Dimensions of the given blob
     */
    TBlobProxy(Precision p, Layout l, const TBlobProxy<T> &blobProxy, size_t offset, const SizeVector &dims)
            : TBlob<T>(p, l, dims), realObject(blobProxy.realObject), offset(offset * sizeof(T)) {
        checkWindow();
    }

    /**
     * @brief Creates a new empty rvalue LockedMemory instance of type void
     * @return LockedMemory instance of type void
     */
    LockedMemory<void> buffer() override {
        return {getAllocator().get(), getHandle(), offset};
    }

    /**
     * @brief Creates a new empty rvalue LockedMemory instance of type const void
     * @return LockedMemory instance of type const void
     */
    LockedMemory<const void> cbuffer() const override {
        return {getAllocator().get(), getHandle(), offset};
    }

    /**
     * @brief Creates a LockedMemory instance of the given type
     * @return LockedMemory instance of the given type
     */
    LockedMemory <T> data() override {
        return {getAllocator().get(), getHandle(), offset};
    }

    /**
    * @brief Creates a readOnly LockedMemory instance of the given type
    * @return Read-only LockedMemory instance of the given type
    */
    LockedMemory<const T> readOnly() const override {
        return {getAllocator().get(), getHandle(), offset};
    }

protected:
    /**
     * @brief Gets an allocator
     * @return An allocator instance
     */
    const std::shared_ptr<IAllocator> &getAllocator() const noexcept override {
        return realObject->getAllocator();
    }

    /**
     * @brief Gets a handle pointer
     * @return A handle pointer
     */
    void *getHandle() const noexcept override {
        return realObject->getHandle();
    }

    /**
     * @brief Checks whether proxy can be created with the requested offset and size parameters
     */
    void checkWindow() {
        if (realObject->size() * realObject->element_size() < base::size() * base::element_size() + offset) {
            THROW_IE_EXCEPTION << "cannot create proxy, offsetInBytes=" << offset << ", sizeInBytes="
                               << base::size() * base::element_size() << ", out of original object size="
                               << realObject->size() * realObject->element_size();
        }
    }

    /**
     * @brief Allocates TBlobProxy data
     * Always throws exception. Not intended to be used
     */
    void allocate() override {
        THROW_IE_EXCEPTION
                << "Proxy blob are not to use allocate, it should rely on actual blob allocation it proxies";
    }

    /**
     * @brief Deallocates TBlobProxy data
     * Always throws exception. Not intended to be used
     */
    bool deallocate() override {
        THROW_IE_EXCEPTION
                << "Proxy blob are not to use deallocate(), it should rely on actual blob allocation it proxies";
    }

private:
    typename Blob::Ptr realObject;
    size_t offset;
};
}  // namespace InferenceEngine
