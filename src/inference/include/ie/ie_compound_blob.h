// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for CompoundBlob
 *
 * @file ie_compound_blob.h
 */
#pragma once

#include <initializer_list>
#include <memory>
#include <vector>

#include "ie_blob.h"

namespace InferenceEngine {
/**
 * @brief This class represents a blob that contains other blobs
 *
 * Compound blob is a wrapper blob over references to underlying blobs. These blobs should share
 * some properties and can be grouped into a single entity.
 */
class INFERENCE_ENGINE_API_CLASS(CompoundBlob) : public Blob {
public:
    /**
     * @brief A smart pointer to the CompoundBlob object
     */
    using Ptr = std::shared_ptr<CompoundBlob>;

    /**
     * @brief A smart pointer to the const CompoundBlob object
     */
    using CPtr = std::shared_ptr<const CompoundBlob>;

    /**
     * @brief Constructs a compound blob from a vector of blobs
     *
     * @param blobs A vector of blobs that is copied to this object
     */
    explicit CompoundBlob(const std::vector<Blob::Ptr>& blobs);

    /**
     * @brief Constructs a compound blob from a vector of blobs
     *
     * @param blobs A vector of blobs that is moved to this object
     */
    explicit CompoundBlob(std::vector<Blob::Ptr>&& blobs);

    /**
     * @brief Always returns `0`
     * @return Returns `0`
     */
    size_t byteSize() const override;

    /**
     * @brief Always returns `0`
     * @return Returns `0`
     */
    size_t element_size() const override;

    /**
     * @brief No operation is performed. Compound blob does not allocate/deallocate any data
     */
    void allocate() noexcept override;

    /**
     * @brief No operation is performed. Compound blob does not allocate/deallocate any data
     * @return Returns `false`
     */
    bool deallocate() noexcept override;

    /**
     * @brief Always returns an empty LockedMemory object
     * @return Empty locked memory
     */
    LockedMemory<void> buffer() noexcept override;

    /**
     * @brief Always returns an empty LockedMemory object
     * @return Empty locked memory
     */
    LockedMemory<const void> cbuffer() const noexcept override;

    /**
     * @brief Returns the number of underlying blobs in the compound blob
     * @return A number of underlying blobs
     */
    size_t size() const noexcept override;

    /**
     * @brief Returns an underlying blob at index i
     *
     * @param i the index of the underlying Blob object
     * @return A smart pointer to the underlying Blob object or nullptr in case of an error
     */
    virtual Blob::Ptr getBlob(size_t i) const noexcept;

    Blob::Ptr createROI(const ROI& roi) const override;

protected:
    /**
     * @brief Constructs a compound blob with specified descriptor
     *
     * @param tensorDesc A tensor descriptor for the compound blob
     */
    explicit CompoundBlob(const TensorDesc& tensorDesc);

    /**
     * @brief Compound blob container for underlying blobs
     */
    std::vector<Blob::Ptr> _blobs;

    const std::shared_ptr<IAllocator>& getAllocator() const noexcept override;
};

/**
 * @brief Represents a blob that contains two planes (Y and UV) in NV12 color format
 */
class INFERENCE_ENGINE_API_CLASS(NV12Blob) : public CompoundBlob {
public:
    /**
     * @brief A smart pointer to the NV12Blob object
     */
    using Ptr = std::shared_ptr<NV12Blob>;

    /**
     * @brief A smart pointer to the const NV12Blob object
     */
    using CPtr = std::shared_ptr<const NV12Blob>;

    /**
     * @brief Constructs NV12 blob from two planes Y and UV
     *
     * @param y Blob object that represents Y plane in NV12 color format
     * @param uv Blob object that represents UV plane in NV12 color format
     */
    NV12Blob(const Blob::Ptr& y, const Blob::Ptr& uv);

    /**
     * @brief Constructs NV12 blob from two planes Y and UV
     *
     * @param y Blob object that represents Y plane in NV12 color format
     * @param uv Blob object that represents UV plane in NV12 color format
     */
    NV12Blob(Blob::Ptr&& y, Blob::Ptr&& uv);

    /**
     * @brief Returns a shared pointer to Y plane
     * @return Y plane
     */
    virtual Blob::Ptr& y() noexcept;

    /**
     * @brief Returns a shared pointer to Y plane
     * @return Y plane
     */
    virtual const Blob::Ptr& y() const noexcept;

    /**
     * @brief Returns a shared pointer to UV plane
     * @return UV plane
     */
    virtual Blob::Ptr& uv() noexcept;

    /**
     * @brief Returns a shared pointer to UV plane
     * @return UV plane
     */
    virtual const Blob::Ptr& uv() const noexcept;

    Blob::Ptr createROI(const ROI& roi) const override;
};

/**
 * @brief Represents a blob that contains three planes (Y,U and V) in I420 color format
 */
class INFERENCE_ENGINE_API_CLASS(I420Blob) : public CompoundBlob {
public:
    /**
     * @brief A smart pointer to the I420Blob object
     */
    using Ptr = std::shared_ptr<I420Blob>;

    /**
     * @brief A smart pointer to the const I420Blob object
     */
    using CPtr = std::shared_ptr<const I420Blob>;

    /**
     * @brief Constructs I420 blob from three planes Y, U and V
     * @param y Blob object that represents Y plane in I420 color format
     * @param u Blob object that represents U plane in I420 color format
     * @param v Blob object that represents V plane in I420 color format
     */
    I420Blob(const Blob::Ptr& y, const Blob::Ptr& u, const Blob::Ptr& v);

    /**
     * @brief Constructs I420 blob from three planes Y, U and V
     * @param y Blob object that represents Y plane in I420 color format
     * @param u Blob object that represents U plane in I420 color format
     * @param v Blob object that represents V plane in I420 color format
     */
    I420Blob(Blob::Ptr&& y, Blob::Ptr&& u, Blob::Ptr&& v);

    /**
     * @brief Returns a reference to shared pointer to Y plane
     *
     * Please note that reference to Blob::Ptr is returned. I.e. the reference will be valid until
     * the I420Blob object is destroyed.
     *
     * @return reference to shared pointer object of Y plane
     */
    Blob::Ptr& y() noexcept;

    /**
     * @brief Returns a constant reference to shared pointer to Y plane
     *
     * Please note that reference to Blob::Ptr is returned. I.e. the reference will be valid until
     * the I420Blob object is destroyed.
     *
     * @return constant reference to shared pointer object of Y plane*
     */
    const Blob::Ptr& y() const noexcept;

    /**
     * @brief Returns a reference to shared pointer to U plane
     *
     * Please note that reference to Blob::Ptr is returned. I.e. the reference will be valid until
     * the I420Blob object is destroyed.
     *
     * @return reference to shared pointer object of U plane
     */
    Blob::Ptr& u() noexcept;

    /**
     * @brief Returns a constant reference to shared pointer to U plane
     *
     * Please note that reference to Blob::Ptr is returned. I.e. the reference will be valid until
     * the I420Blob object is destroyed.
     *
     * @return constant reference to shared pointer object of U plane
     */
    const Blob::Ptr& u() const noexcept;

    /**
     * @brief Returns a reference to shared pointer to V plane
     *
     * Please note that reference to Blob::Ptr is returned. I.e. the reference will be valid until
     * the I420Blob object is destroyed.
     *
     * @return reference to shared pointer object of V plane
     */
    Blob::Ptr& v() noexcept;

    /**
     * @brief Returns a constant reference to shared pointer to V plane
     *
     * Please note that reference to Blob::Ptr is returned. I.e. the reference will be valid until
     * the I420Blob object is destroyed.
     *
     * @return constant reference to shared pointer object of V plane
     */
    const Blob::Ptr& v() const noexcept;

    Blob::Ptr createROI(const ROI& roi) const override;
};

/**
 * @brief This class represents a blob that contains other blobs - one per batch
 * @details Plugin which supports BatchedBlob input should report BATCHED_BLOB
 * in the OPTIMIZATION_CAPABILITIES metric.
 */
class INFERENCE_ENGINE_API_CLASS(BatchedBlob) : public CompoundBlob {
public:
    /**
     * @brief A smart pointer to the BatchedBlob object
     */
    using Ptr = std::shared_ptr<BatchedBlob>;

    /**
     * @brief A smart pointer to the const BatchedBlob object
     */
    using CPtr = std::shared_ptr<const BatchedBlob>;

    /**
     * @brief Constructs a batched blob from a vector of blobs
     * @details All passed blobs should meet following requirements:
     * - all blobs have equal tensor descriptors,
     * - blobs layouts should be one of: NCHW, NHWC, NCDHW, NDHWC, NC, CN, C, CHW, HWC
     * - batch dimensions should be equal to 1 or not defined (C, CHW, HWC).
     * Resulting blob's tensor descriptor is constructed using tensor descriptors
     * of passed blobs by setting batch dimension to blobs.size()
     *
     * @param blobs A vector of blobs that is copied to this object
     */
    explicit BatchedBlob(const std::vector<Blob::Ptr>& blobs);

    /**
     * @brief Constructs a batched blob from a vector of blobs
     * @details All passed blobs should meet following requirements:
     * - all blobs have equal tensor descriptors,
     * - blobs layouts should be one of: NCHW, NHWC, NCDHW, NDHWC, NC, CN, C, CHW, HWC
     * - batch dimensions should be equal to 1 or not defined (C, CHW, HWC).
     * Resulting blob's tensor descriptor is constructed using tensor descriptors
     * of passed blobs by setting batch dimension to blobs.size()
     *
     * @param blobs A vector of blobs that is moved to this object
     */
    explicit BatchedBlob(std::vector<Blob::Ptr>&& blobs);
};
}  // namespace InferenceEngine
