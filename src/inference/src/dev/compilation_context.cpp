// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sys/stat.h>
#include <sys/types.h>

#ifndef _WIN32
#    include <unistd.h>
#endif

#include "itt.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/compilation_context.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "transformations/hash.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

#ifdef _WIN32
#    define stat _stat
#endif

namespace ov {
template <typename T>
static uint64_t hash_combine(uint64_t seed, const T& a) {
    // Hash combine formula from boost
    return seed ^ (std::hash<T>()(a) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

std::string ModelCache::calculate_file_info(const std::string& filePath) {
    uint64_t seed = 0;
    auto absPath = filePath;
    if (filePath.size() > 0) {
        try {
            absPath = ov::util::get_absolute_file_path(filePath);
        } catch (std::runtime_error&) {
            // can't get absolute path, will use filePath for hash
        }
    }

    seed = hash_combine(seed, absPath);

    std::string res;
    struct stat result;
    if (stat(absPath.c_str(), &result) == 0) {
        seed = hash_combine(seed, result.st_mtime);
        seed = hash_combine(seed, result.st_size);
    }
    return std::to_string(seed);
}

std::string ModelCache::compute_hash(const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& compileOptions) {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ReadTime, "ModelCache::compute_hash - Model");
    return compute_hash(model, {}, compileOptions);
}

std::string ModelCache::compute_hash(const std::shared_ptr<const ov::Model>& model,
                                     const std::filesystem::path& model_path,
                                     const ov::AnyMap& compileOptions) {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ReadTime, "ModelCache::compute_hash - Model and path");

    OPENVINO_ASSERT(model);

    uint64_t seed = 0;
    // 1. Calculate hash on function, skipping weights if model path is provided
    ov::pass::Manager m;
    m.register_pass<ov::pass::Hash>(seed, !model_path.empty());
    m.run_passes(std::const_pointer_cast<ov::Model>(model));

    // 2. Compute hash on serialized data and options
    for (const auto& [name, option] : compileOptions) {
        seed = hash_combine(seed, name + option.as<std::string>());
    }

    // 3. Add runtime information which may not be serialized
    for (const auto& op : model->get_ordered_ops()) {
        // Skip runtime attributes which are not hash-able
        for (const auto& [name, attribute] : op->get_rt_info()) {
            if (!attribute.is<ov::RuntimeAttribute>() || attribute.as<ov::RuntimeAttribute>().is_deterministic()) {
                seed = hash_combine(seed, name);
                std::stringstream strm;
                attribute.print(strm);
                seed = hash_combine(seed, strm.str());
            }
        }
    }

    // 4. If model path is provided add file info to the hash
    if (!model_path.empty()) {
        seed = hash_combine(seed, compute_hash(model_path.string(), compileOptions));
    }

    return std::to_string(seed);
}

std::string ModelCache::compute_hash(const std::string& modelName, const ov::AnyMap& compileOptions) {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ReadTime, "ModelCache::compute_hash - Model");
    uint64_t seed = 0;
    try {
        seed = hash_combine(seed, ov::util::get_absolute_file_path(modelName));
    } catch (...) {
        // can't get absolute path, use modelName for hash calculation
        seed = hash_combine(seed, modelName);
    }
    for (const auto& kvp : compileOptions) {
        seed = hash_combine(seed, kvp.first + kvp.second.as<std::string>());
    }
    return std::to_string(seed);
}

std::string ModelCache::compute_hash(const std::string& modelStr,
                                     const ov::Tensor& tensor,
                                     const ov::AnyMap& compileOptions) {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ReadTime, "ModelCache::compute_hash - Model");
    uint64_t seed = 0;
    // model string
    seed = hash_combine(seed, modelStr);

    // tensor data
    if (tensor) {
        seed = hash_combine(seed, tensor.get_size());

        auto ptr = static_cast<const size_t*>(tensor.data());
        size_t size = tensor.get_size() / sizeof(size_t);

        // 10MB block size in size_t
        const size_t block_size = 10000000 / sizeof(size_t);
        size_t blocks_num = size / block_size;
        std::vector<uint64_t> block_hashes(blocks_num + 1, 0);

        ov::parallel_for(blocks_num, [&](size_t block_idx) {
            uint64_t local_hash = 0;
            auto local_ptr = ptr + block_size * block_idx;
            for (size_t i = 0; i < block_size; i++) {
                local_hash = hash_combine(local_hash, local_ptr[i]);
            }
            block_hashes[block_idx] = local_hash;
        });

        {
            uint64_t local_hash = 0;
            auto local_ptr = ptr + block_size * blocks_num;
            auto elements_left = size - block_size * blocks_num;
            for (size_t i = 0; i < elements_left; i++) {
                local_hash = hash_combine(local_hash, local_ptr[i]);
            }
            block_hashes[blocks_num] = local_hash;
        }

        for (auto hash : block_hashes) {
            seed = hash_combine(seed, hash);
        }

        auto size_done = size * sizeof(size_t);
        auto ptr_left = static_cast<const uint8_t*>(tensor.data()) + size_done;
        size_t size_left = tensor.get_size() - size_done;
        for (size_t i = 0; i < size_left; i++)
            seed = hash_combine(seed, ptr_left[i]);
    }

    // compile options
    for (const auto& kvp : compileOptions) {
        seed = hash_combine(seed, kvp.first + kvp.second.as<std::string>());
    }
    return std::to_string(seed);
}

//////////////////////////////////////////////////

CompiledBlobHeader::CompiledBlobHeader() {}

CompiledBlobHeader::CompiledBlobHeader(const std::string& ieVersion,
                                       const std::string& fileInfo,
                                       const std::string& runtimeInfo,
                                       const size_t headerSizeAlignment)
    : m_ieVersion(ieVersion),
      m_fileInfo(fileInfo),
      m_runtimeInfo(runtimeInfo),
      m_headerSizeAlignment(headerSizeAlignment) {}

std::istream& operator>>(std::istream& stream, CompiledBlobHeader& header) {
    std::string xmlStr;

    auto start = stream.tellg();
    std::getline(stream, xmlStr);
    auto end = stream.tellg();

    pugi::xml_document document;
    pugi::xml_parse_result res = document.load_string(xmlStr.c_str());
    OPENVINO_ASSERT(res.status == pugi::status_ok, "Error reading compiled blob header");

    pugi::xml_node compiledBlobNode = document.document_element();
    header.m_ieVersion = ov::util::pugixml::get_str_attr(compiledBlobNode, "ie_version");
    header.m_fileInfo = ov::util::pugixml::get_str_attr(compiledBlobNode, "file_info");
    header.m_runtimeInfo = ov::util::pugixml::get_str_attr(compiledBlobNode, "runtime_info");
    header.m_headerSizeAlignment = ov::util::pugixml::get_uint_attr(compiledBlobNode, "header_size_alignment");

    if (const size_t headerSizeAlignment = header.m_headerSizeAlignment; headerSizeAlignment) {
        size_t bytes_read = static_cast<size_t>(end - start);
        size_t pad =
            (headerSizeAlignment - (bytes_read % headerSizeAlignment)) % headerSizeAlignment;  // 0 if already aligned
        stream.seekg(static_cast<std::streamoff>(pad), std::ios::cur);
        OPENVINO_ASSERT(stream.good(), "Failed to seek over padding in compiled blob header");
    }

    return stream;
}

std::ostream& operator<<(std::ostream& stream, const CompiledBlobHeader& header) {
    pugi::xml_document document;
    auto compiledBlobNode = document.append_child("compiled_blob");
    compiledBlobNode.append_attribute("ie_version").set_value(header.m_ieVersion.c_str());
    compiledBlobNode.append_attribute("file_info").set_value(header.m_fileInfo.c_str());
    compiledBlobNode.append_attribute("runtime_info").set_value(header.m_runtimeInfo.c_str());
    compiledBlobNode.append_attribute("header_size_alignment")
        .set_value(std::to_string(header.m_headerSizeAlignment).c_str());

    auto start = stream.tellp();
    document.save(stream, nullptr, pugi::format_raw);
    document.reset();
    stream << std::endl;
    auto end = stream.tellp();

    // add padding
    if (const size_t headerSizeAlignment = header.m_headerSizeAlignment; headerSizeAlignment) {
        size_t bytes_written = static_cast<size_t>(end - start);
        size_t pad = (headerSizeAlignment - (bytes_written % headerSizeAlignment)) %
                     headerSizeAlignment;  // 0 if already aligned
        std::fill_n(std::ostream_iterator<char>(stream), pad, 0);
    }

    return stream;
}

namespace {
inline std::string getline_from_buffer(const char* buffer, size_t size, size_t& pos, char delim = '\n') {
    if (pos >= size) {
        return {};
    }

    const char* start = buffer + pos;
    const char* end = buffer + size;
    const char* newline = std::find(start, end, delim);

    size_t line_length = (newline == end) ? (end - start) : (newline - start);
    std::string line(start, line_length);

    // Update position (skip the delimiter if found)
    pos += line_length + (newline != end ? 1 : 0);

    return line;
}
}  // namespace

void CompiledBlobHeader::read_from_buffer(const char* buffer, size_t buffer_size, size_t& pos) {
    size_t start = pos;
    std::string xmlStr = ov::getline_from_buffer(buffer, buffer_size, pos);
    size_t end = pos;

    pugi::xml_document document;
    pugi::xml_parse_result res = document.load_string(xmlStr.c_str());
    OPENVINO_ASSERT(res.status == pugi::status_ok, "Error reading compiled blob header");

    pugi::xml_node compiledBlobNode = document.document_element();
    m_ieVersion = ov::util::pugixml::get_str_attr(compiledBlobNode, "ie_version");
    m_fileInfo = ov::util::pugixml::get_str_attr(compiledBlobNode, "file_info");
    m_runtimeInfo = ov::util::pugixml::get_str_attr(compiledBlobNode, "runtime_info");
    m_headerSizeAlignment = ov::util::pugixml::get_uint_attr(compiledBlobNode, "header_size_alignment");

    if (const size_t headerSizeAlignment = m_headerSizeAlignment; headerSizeAlignment) {
        size_t bytes_read = static_cast<size_t>(end - start);
        size_t pad =
            (headerSizeAlignment - (bytes_read % headerSizeAlignment)) % headerSizeAlignment;  // 0 if already aligned
        pos += pad;
    }
}
}  // namespace ov
