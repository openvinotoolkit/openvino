// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>

#include <fstream>
#include <string>

#include "graph_iterator_saved_model.hpp"
#include "tensor_bundle.pb.h"
#include "trackable_object_graph.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {

template <typename T>
static T smReadFixed(const char* ptr) {
    T result = 0;
    for (uint8_t i = 0; i < sizeof(T); ++i) {
        result |= ptr[i] << (i * 8);
    }
    return result;
}

template <typename T>
static T smUnpack(char** ptr, const char* ptr_end) {
    T result = 0;
    for (uint8_t i = 0; i < sizeof(T) * 7 && *ptr < ptr_end; i += 7) {
        T byte = *((*ptr)++);
        if (byte & 0x80) {
            result |= ((byte & 0x7F) << i);
        } else {
            result |= byte << i;
            return result;
        }
    }
    return 0;
}

struct VIBlock {
    uint64_t m_size;
    uint64_t m_offset;

    void read(char** ptr, const char* ptr_end) {
        m_offset = smUnpack<uint64_t>(ptr, ptr_end);
        m_size = smUnpack<uint64_t>(ptr, ptr_end);
    }
};

struct VIFooter {
    VIBlock m_metaIndex;
    VIBlock m_index;

    void read(char** ptr, const char* ptr_end) {
        m_index.read(ptr, ptr_end);
        m_metaIndex.read(ptr, ptr_end);
    }

    void Read(std::ifstream& fs) {
        fs.seekg(0, std::ios::end);
        size_t size = fs.tellg();
        char footerData[48] = {}, *ptr = &footerData[0];
        fs.seekg(size - sizeof(footerData));
        fs.read(ptr, sizeof(footerData));

        // https://github.com/tensorflow/tensorflow/blob/9659b7bdca80a8ef8240eb021d4da089034eeb00/tensorflow/tsl/lib/io/format.cc#L59
        ptr += sizeof(footerData) - 8;
        uint32_t magic_lo = *reinterpret_cast<const uint32_t*>(ptr);
        uint32_t magic_hi = *reinterpret_cast<const uint32_t*>(ptr + 4);
        uint64_t magic_no = (static_cast<uint64_t>(magic_hi) << 32) | static_cast<uint64_t>(magic_lo);

        FRONT_END_GENERAL_CHECK(magic_no == 0xdb4775248b80fb57ull, "Wrong index file, magic number mismatch detected");

        ptr = &footerData[0];
        m_metaIndex.read(&ptr, ptr + sizeof(footerData));
        m_index.read(&ptr, ptr + sizeof(footerData));
    }
};

void SavedModelVariablesIndex::read_variables_index_block(std::ifstream& fs,
                                                          const VIBlock* index,
                                                          std::vector<char>& data,
                                                          uint32_t* offset,
                                                          uint32_t* offset_end) {
    data.clear();
    data.resize(index->m_size + 5 /*kBlockTrailerSize*/);
    fs.seekg(index->m_offset, std::ios::beg);
    fs.read(data.data(), data.size());
    FRONT_END_GENERAL_CHECK(data[index->m_size] == 0, "Compressed files aren't supported");
    uint32_t numRestarts = smReadFixed<uint32_t>(data.data() + index->m_size - sizeof(uint32_t));
    size_t maxRestarts = (index->m_size - sizeof(uint32_t)) / sizeof(uint32_t);
    FRONT_END_GENERAL_CHECK(maxRestarts >= numRestarts, "Wrong restarts value");
    *offset_end = static_cast<uint32_t>(index->m_size) - ((numRestarts + 1) * sizeof(uint32_t));
    *offset = smReadFixed<uint32_t>(data.data() + *offset_end);
}

void SavedModelVariablesIndex::read_variables_index_pair(char** ptr,
                                                         const char* ptr_end,
                                                         std::string& key,
                                                         char** value,
                                                         uint32_t* val_length) {
    uint32_t shared, nonShared;
    shared = smUnpack<uint32_t>(ptr, ptr_end);
    nonShared = smUnpack<uint32_t>(ptr, ptr_end);
    *val_length = smUnpack<uint32_t>(ptr, ptr_end);

    // Key inherits last part of string (shared-size bytes) and appends new string
    // shared_part_key1 //resize(0) + append(shared_part_key1)
    // ............key2 //resize(12) + append(key2)
    // ............key3 //resize(12) + append(key3)
    // new_shared_key4 //resize(0) + append(new_shared_key4)
    // ...........key5 //resize(11) + append(key5)
    key.resize(shared);
    key.append(*ptr, nonShared);

    *value = *ptr + nonShared;
    *ptr = *value + *val_length;
}

void SavedModelVariablesIndex::read_variables_index(std::ifstream& fs,
                                                    std::map<std::string, std::vector<char>>& varIndex) {
    VIFooter footer;

    footer.Read(fs);

    std::vector<VIBlock> secondLevel;
    std::vector<char> blockData;

    uint32_t offset = 0, offset_end = 0;

    read_variables_index_block(fs, &footer.m_index, blockData, &offset, &offset_end);
    char *ptr = blockData.data() + offset, *ptr_end = blockData.data() + offset_end, *value = nullptr;
    std::string key = "";
    uint32_t valLength;

    while (ptr < ptr_end) {
        read_variables_index_pair(&ptr, ptr_end, key, &value, &valLength);

        VIBlock valBlock;
        valBlock.read(&value, value + valLength);
        secondLevel.push_back(valBlock);
        ptr = value + valLength;
    }

    for (auto& block : secondLevel) {
        read_variables_index_block(fs, &block, blockData, &offset, &offset_end);

        key = "";
        ptr = blockData.data() + offset;
        ptr_end = blockData.data() + offset_end;
        while (ptr < ptr_end) {
            read_variables_index_pair(&ptr, ptr_end, key, &value, &valLength);
            varIndex[key] = std::vector<char>(value, value + valLength);
        }
    }
}

void SavedModelVariablesIndex::read_bundle_header() {
    auto item = m_variables_index.find("");
    FRONT_END_GENERAL_CHECK(item != m_variables_index.end(), "Bundle Header isn't found in index");

    ::tensorflow::BundleHeaderProto bundleHeader;
    FRONT_END_GENERAL_CHECK(bundleHeader.ParseFromString(item->second.data()),
                            "Bundle Header: Cannot parse Bundle Header");
    FRONT_END_GENERAL_CHECK(bundleHeader.version().producer() == 1, "Bundle Header: Unsupported producer version");
    FRONT_END_GENERAL_CHECK(bundleHeader.version().min_consumer() == 0, "Bundle Header: Unsupported consumer version");
    FRONT_END_GENERAL_CHECK(bundleHeader.endianness() == 0, "Bundle Header: BIG endian isn't supported");

    m_total_shards = bundleHeader.num_shards();
}

void SavedModelVariablesIndex::read_checkpointable_object_graph() {
    m_variables_map.clear();

    auto item = m_variables_index.find("_CHECKPOINTABLE_OBJECT_GRAPH");
    FRONT_END_GENERAL_CHECK(item != m_variables_index.end(), "Checkpointable Object Graph isn't found in index");

    ::tensorflow::BundleEntryProto entry;
    FRONT_END_GENERAL_CHECK(entry.ParseFromArray(item->second.data(), static_cast<int>(item->second.size())),
                            "CMO: Cannot parse Bundle Entry");

    FRONT_END_GENERAL_CHECK(entry.slices().empty(), "CMO: Slices are not supported");

    auto shard = m_data_files.find(entry.shard_id());
    FRONT_END_GENERAL_CHECK(shard != m_data_files.end(), "CMO: data files isn't found");

    std::vector<char> data(entry.size());
    ::tensorflow::TrackableObjectGraph tog;

    // TODO: have to understand this offset
    // It looks like reinterpret_cast artifact
    // https://github.com/tensorflow/tensorflow/blob/d90f1947ebcf510b23c238f43c2191e5b3817cb3/tensorflow/cc/experimental/libexport/load.cc#L70
    int chg = 6;
    shard->second->seekg(entry.offset() + chg);
    shard->second->read(data.data(), entry.size() - chg);

    bool result = tog.ParseFromArray(data.data(), static_cast<int>(data.size()) - chg);

    // Might be need to remove this verification:
    // https://github.com/tensorflow/tensorflow/blob/d90f1947ebcf510b23c238f43c2191e5b3817cb3/tensorflow/cc/experimental/libexport/load.cc#L73
    // FRONT_END_GENERAL_CHECK(result, "CMO: Trackable Object Graph couldn't be read");

    for (const auto& node : tog.nodes()) {
        for (const auto& attr : node.attributes()) {
            m_variables_map[attr.full_name()] = attr.checkpoint_key();
        }
    }
}

bool GraphIteratorSavedModel::is_valid_signature(const ::tensorflow::SignatureDef& signature) {
    for (const auto& it : signature.inputs()) {
        if (it.second.name().empty()
            //			|| !isRefType(it.second.dtype())
        )
            return false;
    }
    for (const auto& it : signature.outputs()) {
        if (it.second.name().empty()
            //			|| !isRefType(it.second.dtype())
        )
            return false;
    }
    return true;
}

bool SavedModelVariablesIndex::read_variables(std::ifstream& vi_stream, const std::string& path) {
    m_variables_index.clear();
    read_variables_index(vi_stream, m_variables_index);
    read_bundle_header();

    std::vector<char> suffix(20);
    for (int32_t shard = 0; shard < m_total_shards; ++shard) {
        std::snprintf(suffix.data(), suffix.size(), "data-%05d-of-%05d", shard, m_total_shards);
        std::string fullPath = ov::util::path_join({path, "variables", std::string("variables.") + suffix.data()});
        m_data_files[shard] =
            std::shared_ptr<std::ifstream>(new std::ifstream(fullPath, std::ifstream::in | std::ifstream::binary));
        FRONT_END_GENERAL_CHECK(m_data_files[shard]->is_open(), "Saved Model's variable index file does not exist");
    }

    read_checkpointable_object_graph();
    return true;
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
bool SavedModelVariablesIndex::read_variables(std::ifstream& vi_stream, const std::wstring& path) {
    m_variables_index.clear();
    read_variables_index(vi_stream, m_variables_index);
    read_bundle_header();

    std::vector<wchar_t> suffix(20);
    for (int32_t shard = 0; shard < m_total_shards; ++shard) {
        swprintf_s(suffix.data(), suffix.size(), L"data-%05d-of-%05d", shard, m_total_shards);
        std::wstring fullPath =
            ov::util::path_join_w({path, L"variables", std::wstring(L"variables.") + suffix.data()});
        m_data_files[shard] =
            std::shared_ptr<std::ifstream>(new std::ifstream(fullPath, std::ifstream::in | std::ifstream::binary));
        FRONT_END_GENERAL_CHECK(m_data_files[shard]->is_open(), "Saved Model's variable index file does not exist");
    }

    read_checkpointable_object_graph();
    return true;
}
#endif

bool GraphIteratorSavedModel::is_supported(const std::string& path) {
    return ov::util::directory_exists(path) && ov::util::file_exists(ov::util::path_join({path, "saved_model.pb"}));
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
bool GraphIteratorSavedModel::is_supported(const std::wstring& path) {
    return ov::util::directory_exists(path) && ov::util::file_exists(ov::util::path_join_w({path, L"saved_model.pb"}));
}
#endif

template <>
std::basic_string<char> get_saved_model_name<char>() {
    return "/saved_model.pb";
}
template <>
std::basic_string<char> get_variables_index_name<char>() {
    return "/variables/variables.index";
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_saved_model_name<wchar_t>() {
    return L"/saved_model.pb";
}
template <>
std::basic_string<wchar_t> get_variables_index_name<wchar_t>() {
    return L"/variables/variables.index";
}
#endif

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
