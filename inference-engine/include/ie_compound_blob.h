// Copyright (C) 2018-2020 Intel Corporation
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
class INFERENCE_ENGINE_API_CLASS(CompoundBlob): public Blob {
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
     * @brief A virtual destructor
     */
    virtual ~CompoundBlob() = default;

    /**
     * @brief A copy constructor
     */
    CompoundBlob(const CompoundBlob& blob);

    /**
     * @brief A move constructor
     */
    CompoundBlob(CompoundBlob&& blob);

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
     * @brief Always returns 0
     */
    size_t byteSize() const noexcept override;

    /**
     * @brief Always returns 0
     */
    size_t element_size() const noexcept override;

    /**
     * @brief No operation is performed. Compound blob does not allocate/deallocate any data
     */
    void allocate() noexcept override;

    /**
     * @brief No operation is performed. Compound blob does not allocate/deallocate any data
     * @return false
     */
    bool deallocate() noexcept override;

    /**
     * @brief Always returns an empty LockedMemory object
     */
    LockedMemory<void> buffer() noexcept override;

    /**
     * @brief Always returns an empty LockedMemory object
     */
    LockedMemory<const void> cbuffer() const noexcept override;

    /**
     * @brief Returns the number of underlying blobs in the compound blob
     */
    size_t size() const noexcept override;

    /**
     * @brief Returns an underlying blob at index i
     *
     * @param i the index of the underlying Blob object
     * @return A smart pointer to the underlying Blob object or nullptr in case of an error
     */
    virtual Blob::Ptr getBlob(size_t i) const noexcept;

protected:
    /**
     * @brief A default constructor
     */
    CompoundBlob();

    /**
     * @brief Compound blob container for underlying blobs
     */
    std::vector<Blob::Ptr> _blobs;

    /**
     * @brief Returns nullptr as CompoundBlob is not allocator-based
     */
    const std::shared_ptr<IAllocator>& getAllocator() const noexcept override;

    /**
     * @brief Returns nullptr as CompoundBlob is not allocator-based
     */
    void* getHandle() const noexcept override;

    /**
     * @brief Clones this container. Cloned container references the same blobs as this one.
     *
     * @return Pointer to the cloned object
     */
    Blob* clone() const override;

    /**
     * @brief Sets ROI for the blob
     *
     * @param roiData region of interest and original blob
     */
    void setROI(const ROIData& roiData) override;
};

/**
 * @brief Represents a blob that contains two planes (Y and UV) in NV12 color format
 */
class INFERENCE_ENGINE_API_CLASS(NV12Blob): public CompoundBlob {
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
     * @brief A deleted default constructor
     */
    NV12Blob() = delete;

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
     * @brief A virtual destructor
     */
    virtual ~NV12Blob() = default;

    /**
     * @brief A copy constructor
     */
    NV12Blob(const NV12Blob& blob) = default;

    /**
     * @brief A move constructor
     */
    NV12Blob(NV12Blob&& blob) = default;

    /**
     * @brief Returns a shared pointer to Y plane
     */
    virtual Blob::Ptr& y() noexcept;

    /**
     * @brief Returns a shared pointer to Y plane
     */
    virtual const Blob::Ptr& y() const noexcept;

    /**
     * @brief Returns a shared pointer to UV plane
     */
    virtual Blob::Ptr& uv() noexcept;

    /**
     * @brief Returns a shared pointer to UV plane
     */
    virtual const Blob::Ptr& uv() const noexcept;

    /**
     * @brief Creates another Blob, which is copy of current and share the same memory area, but describes only
     * sub-region of that memory.
     *
     * @param roi region of interest
     * @return Pointer to the copied and modified blob
     */
    Blob::Ptr CreateROIBlob(const ROI& roi) const override;

protected:
    /**
     * @brief Clones Blob object of the particular class. Clone references the same memory as this object.
     *
     * @return Pointer to the cloned object
     */
    Blob* clone() const override;
};

/**
 * @brief Represents a blob that contains three planes (Y,U and V) in I420 color format
 */
class INFERENCE_ENGINE_API_CLASS(I420Blob): public CompoundBlob {
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
     * @brief A deleted default constructor
     */
    I420Blob() = delete;

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
     * @brief A virtual destructor. It is made out of line for RTTI to
     * work correctly on some platforms.
     */
    virtual ~I420Blob();

    /**
     * @brief A copy constructor
     */
    I420Blob(const I420Blob& blob) = default;

    /**
     * @brief A move constructor
     */
    I420Blob(I420Blob&& blob) = default;

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

    /**
     * @brief Creates another Blob, which is copy of current and share the same memory area, but describes only
     * sub-region of that memory.
     *
     * @param roi region of interest
     * @return Pointer to the copied and modified blob
     */
    Blob::Ptr CreateROIBlob(const ROI& roi) const override;

protected:
    /**
     * @brief Clones Blob object of the particular class. Clone references the same memory as this object.
     *
     * @return Pointer to the cloned object
     */
    Blob* clone() const override;
};
}  // namespace InferenceEngine
