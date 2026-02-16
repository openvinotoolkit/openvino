// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Cache Manager class C++ API
 *
 * @file cache_manager.hpp
 */
#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <variant>

#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {

static inline std::ostream& operator<<(std::ostream& stream, const wsh::Context& cache) {
    stream << "--- sh ctx ---" << std::endl;
    for (const auto& [key, nodes_map] : cache.m_weight_registry) {
        for (const auto& [node_id, props] : nodes_map) {
            stream << "key: " << key << "   node id: " << node_id << " offset: " << props.m_offset
                   << " byte size: " << props.m_size << std::endl;
        }
    }
    stream << "-------------" << std::endl;
    return stream;
}

constexpr int CTX_TAG = 0;
constexpr int CTX_E_TAG = 1;
constexpr int BLOB_TAG = 2;

struct SharedCtxCacheDecoder {
    ov::wsh::Context* ctx;

    friend std::istream& operator>>(std::istream& stream, SharedCtxCacheDecoder& cache) {
        int tag{};
        do {
            size_t ctx_size{};
            stream.read(reinterpret_cast<char*>(&tag), sizeof(tag));
            if (!stream.good()) {
                break;
            }
            stream.read(reinterpret_cast<char*>(&ctx_size), sizeof(ctx_size));
            if (!stream.good() || ctx_size == 0) {
                break;
            }
            if (tag == CTX_TAG) {
                const auto end_pos = stream.tellg() + static_cast<std::streamoff>(ctx_size);
                do {
                    auto& cmap = cache.ctx->m_weight_registry;
                    ov::wsh::DataID id, const_id;
                    size_t offset, byte_size;
                    stream.read(reinterpret_cast<char*>(&id), sizeof(id));
                    stream.read(reinterpret_cast<char*>(&const_id), sizeof(const_id));
                    stream.read(reinterpret_cast<char*>(&offset), sizeof(offset));
                    stream.read(reinterpret_cast<char*>(&byte_size), sizeof(byte_size));
                    if (auto id_it = cmap.find(id); id_it != cmap.end()) {
                        id_it->second[const_id] = ov::wsh::WeightMetaData{offset, byte_size, ov::element::u8};
                    } else {
                        cmap[id] = {{const_id, ov::wsh::WeightMetaData{offset, byte_size, ov::element::u8}}};
                    }
                } while (stream.good() && stream.tellg() < end_pos);
            } else {
                stream.seekg(ctx_size ? ctx_size : 1, std::ios::cur);
            }
        } while (stream.good());

        return stream;
    }

    friend std::ostream& operator<<(std::ostream& stream, const SharedCtxCacheDecoder& cache) {
        if (cache.ctx == nullptr || cache.ctx->m_weight_registry.empty()) {
            return stream;
        }
        auto&& cmap = cache.ctx->m_weight_registry;
        stream.write(reinterpret_cast<const char*>(&CTX_TAG), sizeof(CTX_TAG));
        size_t ctx_size = 0;
        const auto size_offset = stream.tellp();
        stream.write(reinterpret_cast<const char*>(&ctx_size), sizeof(ctx_size));
        for (const auto& [id, consts] : cmap) {
            for (const auto& [const_id, props] : consts) {
                stream.write(reinterpret_cast<const char*>(&id), sizeof(id));
                stream.write(reinterpret_cast<const char*>(&const_id), sizeof(const_id));
                stream.write(reinterpret_cast<const char*>(&props.m_offset), sizeof(props.m_offset));
                stream.write(reinterpret_cast<const char*>(&props.m_size), sizeof(props.m_size));
                ctx_size += sizeof(id) + sizeof(const_id) + sizeof(props.m_offset) + sizeof(props.m_size);
            }
        }
        const auto end_pos = stream.tellp();
        stream.seekp(size_offset);
        stream.write(reinterpret_cast<const char*>(&ctx_size), sizeof(ctx_size));
        stream.seekp(end_pos);
        return stream;
    }
};

/**
 * @brief This class limits the locale env to a special value in sub-scope
 *
 */
class ScopedLocale {
public:
    ScopedLocale(int category, std::string newLocale) : m_category(category) {
        m_oldLocale = setlocale(category, nullptr);
        setlocale(m_category, newLocale.c_str());
    }
    ~ScopedLocale() {
        setlocale(m_category, m_oldLocale.c_str());
    }

private:
    int m_category;
    std::string m_oldLocale;
};

/**
 * @brief This class represents private interface for Cache Manager
 *
 */
class ICacheManager {
public:
    /**
     * @brief Default destructor
     */
    virtual ~ICacheManager() = default;

    /**
     * @brief Function passing created output stream
     *
     */
    using StreamWriter = std::function<void(std::ostream&)>;
    /**
     * @brief Callback when OpenVINO intends to write model to cache
     *
     * Client needs to call create std::ostream object and call writer(ostream)
     * Otherwise, model will not be cached
     *
     * @param id Id of cache (hash of the model)
     * @param writer Lambda function to be called when stream is created
     */
    virtual void write_cache_entry(const std::string& id, StreamWriter writer) = 0;

    /**
     * @brief Variant type for compiled blob representation
     */
    using CompiledBlobVariant = std::variant<const ov::Tensor, std::reference_wrapper<std::istream>>;

    /**
     * @brief Function passing created input stream
     */
    using StreamReader = std::function<void(CompiledBlobVariant&)>;

    /**
     * @brief Callback when OpenVINO intends to read model from cache
     *
     * Client needs to call create std::istream object and call reader(istream)
     * Otherwise, model will not be read from cache and will be loaded as usual
     *
     * @param id Id of cache (hash of the model)
     * @param enable_mmap use mmap or ifstream to read model file
     * @param reader Lambda function to be called when input stream is created
     */
    virtual void read_cache_entry(const std::string& id, bool enable_mmap, StreamReader reader) = 0;

    /**
     * @brief Callback when OpenVINO intends to remove cache entry
     *
     * Client needs to perform appropriate cleanup (e.g. delete a cache file)
     *
     * @param id Id of cache (hash of the model)
     */
    virtual void remove_cache_entry(const std::string& id) = 0;
};

class ISharedCtxCacheManager {
public:
    virtual void write_context_entry(const ov::wsh::Context& ctx) = 0;
    virtual std::shared_ptr<const ov::weight_sharing::Context> get_shared_ctx() const = 0;
};

/**
 * @brief File storage-based Implementation of ICacheManager
 *
 * Uses simple file for read/write cached models.
 *
 */
class FileStorageCacheManager final : public ICacheManager {
    std::filesystem::path m_cache_path;

    std::filesystem::path get_blob_file(const std::string& blob_hash) const {
        return m_cache_path / (blob_hash + ".blob");
    }

public:
    /**
     * @brief Constructor
     *
     */
    FileStorageCacheManager(std::filesystem::path cache_path) : m_cache_path(std::move(cache_path)) {
        util::create_directory_recursive(m_cache_path);
    }

private:
    void write_cache_entry(const std::string& id, StreamWriter writer) override {
        // Fix the bug caused by pugixml, which may return unexpected results if the locale is different from "C".
        ScopedLocale plocal_C(LC_ALL, "C");
        const auto blob_path = get_blob_file(id);
        std::ofstream stream(blob_path, std::ios_base::binary);
        writer(stream);
        stream.close();
        std::filesystem::permissions(blob_path,
                                     std::filesystem::perms::owner_read | std::filesystem::perms::group_read);
    }

    void read_cache_entry(const std::string& id, bool enable_mmap, StreamReader reader) override {
        // Fix the bug caused by pugixml, which may return unexpected results if the locale is different from "C".
        ScopedLocale plocal_C(LC_ALL, "C");
        const auto blob_path = get_blob_file(id);
        if (std::filesystem::exists(blob_path)) {
            if (enable_mmap) {
                CompiledBlobVariant compiled_blob{std::in_place_index<0>, ov::read_tensor_data(blob_path)};
                reader(compiled_blob);
            } else {
                std::ifstream stream(blob_path, std::ios_base::binary);
                CompiledBlobVariant compiled_blob{std::in_place_index<1>, std::ref(stream)};
                reader(compiled_blob);
            }
        }
    }

    void remove_cache_entry(const std::string& id) override {
        const auto blob_path = get_blob_file(id);

        if (std::filesystem::exists(blob_path)) {
            std::ignore = std::filesystem::remove(blob_path);
        }
    }
};

class SingleFileStorageCacheManager final : public ICacheManager, public ISharedCtxCacheManager {
    std::filesystem::path m_cache_file_path;
    std::unordered_map<std::string, std::tuple<size_t, size_t>> m_cache_index;  // blob_id -> (offset, size)
    std::shared_ptr<ov::weight_sharing::Context> m_shared_ctx;
    ov::wsh::Context m_ctx_diff;
    std::streampos m_ctx_end;

    bool has_blob_id(const std::string& blob_id) const {
        return m_cache_index.find(blob_id) != m_cache_index.end();
    }

    void populate_cache_index() {
        const auto f_size = ov::util::file_size(m_cache_file_path);
        if (std::ifstream blob_file(m_cache_file_path, std::ios_base::binary); blob_file.is_open()) {
            // Read shared context from the cache file
            int tag{};
            size_t size{};
            while (blob_file.good() && blob_file.tellg() < f_size) {
                blob_file.read(reinterpret_cast<char*>(&tag), sizeof(tag));
                blob_file.read(reinterpret_cast<char*>(&size), sizeof(size));
                if (blob_file.eof()) {
                    break;
                }
                if (tag == BLOB_TAG) {
                    std::string blob_id(24, '\0');
                    blob_file.read(blob_id.data(), blob_id.size());
                    blob_id.erase(std::find(blob_id.begin(), blob_id.end(), '\0'), blob_id.end());
                    size -= 24;
                    m_cache_index.emplace_hint(m_cache_index.end(),
                                               blob_id,
                                               std::make_tuple(static_cast<size_t>(blob_file.tellg()), size));
                }
                blob_file.seekg(size, std::ios::cur);
            }
        }
    }

public:
    /**
     * @brief Construct a new Single File Storage Cache Manager object
     *
     * @param cache_path
     */
    SingleFileStorageCacheManager(const std::filesystem::path& cache_path)
        : m_cache_file_path(std::move(cache_path)),
          m_cache_index(),
          m_shared_ctx(std::make_shared<ov::weight_sharing::Context>()),
          m_ctx_diff(),
          m_ctx_end(0) {
        util::create_directory_recursive(m_cache_file_path.parent_path());
        if (!util::file_exists(m_cache_file_path)) {
            std::ofstream stream(m_cache_file_path, std::ios_base::binary);
        } else {
            populate_cache_index();
            update_shared_ctx_from_file();
        }
    }

private:
    void update_shared_ctx(const ov::wsh::Context& new_ctx) {
        for (const auto& [src_id, consts] : new_ctx.m_weight_registry) {
            for (const auto& [const_id, props] : consts) {
                if (auto id_it = m_shared_ctx->m_weight_registry.find(src_id);
                    id_it != m_shared_ctx->m_weight_registry.end()) {
                    id_it->second[const_id] = props;
                } else {
                    m_shared_ctx->m_weight_registry[src_id] = {{const_id, props}};
                }
            }
        }
    }

    void update_shared_ctx_from_file() {
        if (std::ifstream blob_file(m_cache_file_path, std::ios_base::binary | std::ios_base::ate);
            blob_file.is_open() && (m_ctx_end < blob_file.tellg())) {
            blob_file.seekg(m_ctx_end);
            // Read shared context from the cache file
            ov::wsh::Context shared_ctx;
            SharedCtxCacheDecoder ctx_cache{&shared_ctx};
            blob_file >> ctx_cache;
            update_shared_ctx(shared_ctx);
            m_ctx_end = blob_file.tellg();
        }
    }

    void write_ctx_diff(std::ostream& stream) {
        if (!m_ctx_diff.m_weight_registry.empty()) {
            SharedCtxCacheDecoder ctx_cache{&m_ctx_diff};
            stream << ctx_cache;
            m_ctx_end = stream.tellp();
            update_shared_ctx(m_ctx_diff);
            m_ctx_diff.m_weight_registry.clear();
        }
    }

    void write_blob_entry(const std::string& id, StreamWriter& writer, std::ofstream& stream) {
        OPENVINO_ASSERT(!has_blob_id(id), "Blob with id ", id, " already exists in cache.");
        size_t blob_size = 0;
        stream.write(reinterpret_cast<const char*>(&BLOB_TAG), sizeof(BLOB_TAG));
        const auto tag_size_offset = stream.tellp();
        stream.write(reinterpret_cast<const char*>(&blob_size), sizeof(blob_size));

        const auto blob_id_offset = stream.tellp();
        auto ids = id;
        ids.resize(24, '\0');
        stream.write(ids.data(), 24);

        const auto blob_offset = stream.tellp();
        writer(stream);
        const auto end = stream.tellp();
        blob_size = end - blob_id_offset;
        stream.seekp(tag_size_offset);
        stream.write(reinterpret_cast<const char*>(&blob_size), sizeof(blob_size));
        stream.seekp(end);
        m_cache_index[id] = std::make_tuple(blob_offset, blob_size - ids.size());  // Update with actual offset and size
    }

    void write_cache_entry(const std::string& id, StreamWriter writer) override {
        // Fix the bug caused by pugixml, which may return unexpected results if the locale is different from "C".
        ScopedLocale plocal_C(LC_ALL, "C");
        std::ofstream stream(m_cache_file_path, std::ios_base::binary | std::ios_base::in | std::ios_base::ate);
        write_blob_entry(id, writer, stream);
    }

    void read_cache_entry(const std::string& id, bool enable_mmap, StreamReader reader) override {
        // Fix the bug caused by pugixml, which may return unexpected results if the locale is different from "C".
        ScopedLocale plocal_C(LC_ALL, "C");

        if (std::filesystem::exists(m_cache_file_path) && has_blob_id(id)) {
            const auto& [offset, size] = m_cache_index[id];
            if (enable_mmap) {
                CompiledBlobVariant compiled_blob{
                    std::in_place_index<0>,
                    ov::read_tensor_data(m_cache_file_path, element::u8, PartialShape{static_cast<int>(size)}, offset)};
                reader(compiled_blob);
            } else {
                std::ifstream stream(m_cache_file_path, std::ios_base::binary);
                stream.seekg(offset);
                CompiledBlobVariant compiled_blob{std::in_place_index<1>, std::ref(stream)};
                reader(compiled_blob);
            }
        }
    }

    void remove_cache_entry(const std::string&) override {}

    std::shared_ptr<const ov::weight_sharing::Context> get_shared_ctx() const override {
        return m_shared_ctx;
    }

    void write_context_entry(const ov::wsh::Context& ctx) override {
        for (auto&& [src_id, consts] : ctx.m_weight_registry) {
            for (auto&& [const_id, props] : consts) {
                if (auto id_it = m_ctx_diff.m_weight_registry.find(src_id);
                    id_it != m_ctx_diff.m_weight_registry.end()) {
                    id_it->second[const_id] = props;
                } else {
                    m_ctx_diff.m_weight_registry[src_id] = {{const_id, props}};
                }
            }
        }
        update_shared_ctx(ctx);
        std::ofstream stream(m_cache_file_path, std::ios_base::binary | std::ios_base::in | std::ios_base::ate);
        write_ctx_diff(stream);
    }
};

}  // namespace ov
