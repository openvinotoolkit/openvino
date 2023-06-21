// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for CompoundBlob
 *
 * @file ie_compound_blob.h
 */
#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <initializer_list>
#include <memory>
#include <vector>

#include "ie_blob.h"

IE_SUPPRESS_DEPRECATED_START
namespace InferenceEngine {
/**
 * @brief This class represents a blob that contains other blobs
 *
 * Compound blob is a wrapper blob over references to underlying blobs. These blobs should share
 * some properties and can be grouped into a single entity.
 */
class INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CLASS(CompoundBlob) : public Blob {
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
 * @brief This class represents a blob that contains other blobs - one per batch
 * @details Plugin which supports BatchedBlob input should report BATCHED_BLOB
 * in the OPTIMIZATION_CAPABILITIES metric.
 */
class INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CLASS(BatchedBlob) : public CompoundBlob {
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
IE_SUPPRESS_DEPRECATED_END
