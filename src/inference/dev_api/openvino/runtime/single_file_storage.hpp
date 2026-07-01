// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/runtime/icache_manager.hpp"
#include "openvino/runtime/tlv_format.hpp"
#include "openvino/util/ov_version.hpp"

namespace ov::runtime {
class SingleFileStorage final : public ICacheManager, public IContextStore {
public:
    /** @brief Current version of the single file storage format. */
    static constexpr util::Version m_version{0, 1, 0};

    enum class Tag : TLVTraits::TagType {
        String = 0x02,
        Blob = 0x03,
        BlobMap = 0x04,
        ConstantMeta = 0x10,
        WeightSource = 0x11,
    };

    explicit SingleFileStorage(const std::filesystem::path& path);

    /**
     * @brief Write a cache entry to the storage.
     * @param blob_id The identifier of the blob.
     * @param writer The function to write the blob data.
     */
    void write_cache_entry(const std::string& blob_id, StreamWriter writer) override;

    /**
     * @brief Read a cache entry from the storage.
     * @param blob_id The identifier of the blob.
     * @param enable_mmap Whether to use memory mapping for reading the blob data.
     * @param reader The function to read the blob data.
     */
    void read_cache_entry(const std::string& blob_id, bool mmap_enabled, StreamReader reader) override;

    /**
     * @brief Remove a cache entry from the storage.
     * @note This function does nothing - the storage is append-only.
     * @param blob_id The identifier of the blob to be removed.
     */
    void remove_cache_entry(const std::string& blob_id) override;

    /**
     * @brief Write the weight sharing context to the storage.
     * @param context The weight sharing context to be stored.
     */
    void write_context(const weight_sharing::Context& context) override;

    /**
     * @brief Get the weight sharing context from the storage.
     * @return The weight sharing context stored in the storage.
     */
    std::shared_ptr<wsh::Context> get_context() const override;

    void initialize(std::shared_ptr<ov::wsh::Context> weight_sharing_context = {}) override;

    using BlobIdType = uint64_t;
    using DataIdType = uint64_t;
    using PadSizeType = uint64_t;

    static const size_t blob_alignment;

private:
    std::filesystem::path m_file_path;

    struct BlobInfo {
        uint64_t offset;
        uint64_t size;
        std::string model_name;
    };
    std::unordered_map<BlobIdType, BlobInfo> m_blob_index;
    std::shared_ptr<wsh::Context> m_shared_context;
    bool build_content_index(std::ifstream& stream);

    static BlobIdType convert_blob_id(const std::string& blob_id);
    void write_blob_entry(std::fstream& stream, BlobIdType blob_id, StreamWriter& writer);
    bool has_blob_id(BlobIdType blob_id) const;
};
}  // namespace ov::runtime
