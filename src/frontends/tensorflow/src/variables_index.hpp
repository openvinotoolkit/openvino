// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include "graph_iterator_proto.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"
#include "ov_tensorflow/saved_model.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {

using HashTableKeysValuesMap = std::unordered_map<std::string, std::shared_ptr<ov::op::v0::Constant>>;

struct VIBlock;

struct VariableStorage {
    std::shared_ptr<std::ifstream> stream;
    std::shared_ptr<ov::MappedMemory> mmap;
};

// Stores information about variables index
class VariablesIndex {
    // Contains file size for internal checks
    size_t m_variables_index_size;
    // Contains maximum amount of shards, used for creating corrext extension
    int32_t m_total_shards;
    // Contains BundleEntryProto variables list, readed from .index file
    std::map<std::string, std::vector<char>> m_variables_index;
    // List of opened data files for using with BundleEntryProto
    std::map<int32_t, VariableStorage> m_data_files;
    // List of mapped variables which could be read using TrackableObjectGraph
    std::map<std::string, std::string> m_variables_map;
    // Flag shows which file storage is using
    bool m_mmap_enabled;

public:
    VariablesIndex(bool mmap_enabled = false) : m_mmap_enabled(mmap_enabled) {}
    /// \brief Returns mmap_enabled state.
    /// \returns True if mmap is enabled, false otherwise
    bool is_mmap_enabled(void) const {
        return m_mmap_enabled;
    }
    /// \brief Reads variables from opened variable index file. Can cause an asserts in case of issues.
    /// \param vi_stream Opened stream file, file pointer doesn't matter, it will be rewind internally.
    /// \param path A path to file with variables data
    /// \param is_saved_model Flag shows variables index is a part of Saved Model format
    /// \returns Returns true in case of everything loads successfully, false otherwise
    bool read_variables(std::ifstream& vi_stream, const std::string& path, const bool is_saved_model = true);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    /// \brief Reads variables from opened variable index file. Can cause an asserts in case of issues.
    /// \param vi_stream Opened stream file, file pointer doesn't matter, it will be rewind internally.
    /// \param path A path to file with variables data
    /// \param is_saved_model Flag shows variables index is a part of Saved Model format
    /// \returns Returns true in case of everything loads successfully, false otherwise
    bool read_variables(std::ifstream& vi_stream, const std::wstring& path, const bool is_saved_model = true);
#endif

    /// \brief Returns data and size of data of stored variable
    /// \param name Name of variable
    /// \param data Pointer on a pointer where data pointer will be returned
    /// \param size Pointer on a variable which will stores data size
    /// \returns Returns true in case variable was found, false otherwise (data and size will be untouched)
    bool get_variable(const std::string& name, const char** data, size_t* size) const {
        auto varItem = m_variables_index.find(name);
        if (varItem == m_variables_index.end()) {
            return false;
        }
        if (data != nullptr) {
            *data = varItem->second.data();
        }
        if (size != nullptr) {
            *size = varItem->second.size();
        }
        return true;
    }

    /// \brief Returns data and size of data of mapped variable from trackable object graph to variables index
    /// \param name Name of a mapping variable
    /// \param data Pointer on a pointer where data pointer will be returned
    /// \param size Pointer on a variable which will stores data size
    /// \returns Returns true in case variable was found, false otherwise (data and size will be untouched)
    bool get_mapped_variable(const std::string& name, const char** data, size_t* size) const {
        auto mapItem = m_variables_map.find(name);
        if (mapItem == m_variables_map.end()) {
            return false;
        }
        return get_variable(mapItem->second, data, size);
    }

    /// \brief Checks if variable has a mapped pair
    /// \param name Name of variable for checking existence
    /// \returns True in case variable has mapped value and false otherwise
    bool has_mapped_variable(const std::string& name) const {
        auto mapItem = m_variables_map.find(name);
        return mapItem != m_variables_map.end();
    }

    /// \brief Returns shared pointer to a requested shard_id, or nullptr in case of shard_id isn't found
    /// \param shard_id Requested shard_id
    /// \returns Valid shared_ptr with ifstream or with nullptr if shard isn't found
    std::shared_ptr<std::ifstream> get_data_file(const int32_t shard_id) const {
        FRONT_END_GENERAL_CHECK(m_mmap_enabled == false,
                                "[TensorFlow Frontend] Requested ifstream, but mmap is enabled");
        auto result = m_data_files.find(shard_id);
        return result != m_data_files.end() ? result->second.stream : nullptr;
    }

    /// \brief Returns shared pointer to a requested shard_id, or nullptr in case of shard_id isn't found
    /// \param shard_id Requested shard_id
    /// \returns Valid shared_ptr with MappedMemory or with nullptr if shard isn't found
    std::shared_ptr<ov::MappedMemory> get_data_mmap(const int32_t shard_id) const {
        FRONT_END_GENERAL_CHECK(m_mmap_enabled == true,
                                "[TensorFlow Frontend] Requested MappedMemory, but mmap is disabled");
        auto result = m_data_files.find(shard_id);
        return result != m_data_files.end() ? result->second.mmap : nullptr;
    }

    /// \brief Adds variable mapping to the variables map
    /// \param var_name Variable full name (from .index file)
    /// \param map_name Mapped name
    /// \param rewrite Rewrite mapped value in case it exists
    /// \returns True if map updated. False if nothing changed (if variable exists and rewrite is false).
    bool map_variable(const std::string& var_name, const std::string& map_name, bool rewrite = false) {
        if (m_variables_map.find(var_name) != m_variables_map.end() && rewrite == false) {
            return false;
        }

        m_variables_map[var_name] = map_name;
        return true;
    }

    /// \brief Reads relationship between VarHandleOp - RestoreV2 - AssignVariableOp and
    /// stores this information in a provided key=value map. Where key - name of VarHandleOp,
    /// value - long variable name which is stored in RestoreV2.
    /// It needs to map VarHandleOp to right place in .index file.
    /// \param[in] graph_def GraphDef object for analysis
    /// \param[out] variables_map Map of variables found in graph_def
    static void map_assignvariable(const std::shared_ptr<::tensorflow::GraphDef> graph_def,
                                   std::map<std::string, std::string>& variables_map,
                                   HashTableKeysValuesMap& hash_table_keys_map,
                                   HashTableKeysValuesMap& hash_table_values_map);

private:
    /// \brief Reads block structure of .index file
    /// \param[in,out] fs Filestream of .index file, position in file will be updated
    /// \param[in] index Variables index block which stores information about block
    /// \param[out] data Block data will be readed
    /// \param[out] offset Offset of block start
    /// \param[out] offset_end Offset of block end
    void read_variables_index_block(std::ifstream& fs,
                                    const VIBlock& index,
                                    std::vector<char>& data,
                                    uint32_t& offset,
                                    uint32_t& offset_end);
    /// \brief Reads key=value pair from provided pointer
    /// \param[in,out] ptr Actual pointer, will be moved to the end of readed pair (to read next)
    /// \param[in] ptr_end End of memory which shouldn't be passed in case of broken structure
    /// \param[out] key Key name
    /// \param[out] value Stored value for key (isn't a pure string, data block)
    /// \param[out] val_lenght Length of readed value
    void read_variables_index_pair(char*& ptr,
                                   const char* ptr_end,
                                   std::string& key,
                                   char*& value,
                                   uint32_t& val_length);
    /// \brief Reads .index file and stores key=value map in provided varIndex
    /// \param[in,out] fs Filestream should be parsed. Position in file will be updated
    /// \param[out] varIndex Variables indx (key=value) from given filestream
    void read_variables_index(std::ifstream& fs, std::map<std::string, std::vector<char>>& varIndex);
    /// \brief Reads bundle header if it is available. Checks version and saves info about amount of shards
    void read_bundle_header();
    /// \brief Reads key=value map from storef _CHECKPOINTABLE_OBJECT_GRAPH variable
    void read_checkpointable_object_graph();
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
