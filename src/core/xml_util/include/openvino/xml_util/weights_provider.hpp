// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <map>
#include <memory>

namespace ov {
class AlignedBuffer;
}  // namespace ov

namespace ov::util {

/**
 * @brief Interface for loading weight data regions from an underlying storage.
 *
 * Implementations may serve weights either from an in-memory buffer or from a
 * file-backed source.
 */
class WeightsProvider {
public:
    virtual ~WeightsProvider() = default;

    /**
     * @brief Make a contiguous region of weights.
     *
     * @param offset Byte offset from the beginning of the weights source.
     * @param size Number of bytes to load.
     * @return Buffer containing the requested weights region.
     */
    virtual std::shared_ptr<ov::AlignedBuffer> make_region(size_t offset, size_t size) = 0;

    /**
     * @brief Returns the total size of the weights source in bytes.
     *
     * @return Size of the underlying weights source.
     */
    virtual size_t size() const = 0;
};

/**
 * @brief Weights provider implementation backed by an already allocated buffer.
 */
class BufferWeightsProvider : public WeightsProvider {
public:
    /**
     * @brief Constructs a weights provider over an existing buffer.
     *
     * @param weights Buffer containing the full weights blob.
     */
    explicit BufferWeightsProvider(std::shared_ptr<ov::AlignedBuffer> weights);

    /**
     * @brief Returns a view of the requested region from the backing buffer.
     *
     * @param offset Byte offset from the beginning of the weights buffer.
     * @param size Number of bytes to expose.
     * @return Buffer referencing the requested region.
     */
    std::shared_ptr<ov::AlignedBuffer> make_region(size_t offset, size_t size) override;

    /**
     * @brief Returns the total size of the backing weights buffer in bytes.
     *
     * @return Size of the underlying buffer.
     */
    size_t size() const override;

private:
    std::shared_ptr<ov::AlignedBuffer> m_weights;
};

/**
 * @brief Weights provider implementation backed by a weights file on disk.
 */
class FileWeightsProvider : public WeightsProvider {
public:
    /**
     * @brief Constructs a weights provider for the specified file.
     *
     * @param weights_path Path to the weights file.
     */
    explicit FileWeightsProvider(std::filesystem::path weights_path);

    /**
     * @brief Loads the requested region from the weights file.
     *
     * Implementations may cache previously loaded regions.
     *
     * @param offset Byte offset from the beginning of the weights file.
     * @param size Number of bytes to load.
     * @return Buffer containing the requested file region.
     */
    std::shared_ptr<ov::AlignedBuffer> make_region(size_t offset, size_t size) override;

    /**
     * @brief Returns the total size of the weights file in bytes.
     *
     * @return Size of the file-backed weights source.
     */
    size_t size() const override;

private:
    using WeightsRegionKey = std::pair<size_t, size_t>;

    std::filesystem::path m_weights_path;
    size_t m_weights_size = 0;
    size_t m_weights_source_id = 0;
    std::shared_ptr<ov::AlignedBuffer> m_weights_source_handle;
    std::map<WeightsRegionKey, std::shared_ptr<ov::AlignedBuffer>> m_loaded_weights_regions;
};
}  // namespace ov::util
