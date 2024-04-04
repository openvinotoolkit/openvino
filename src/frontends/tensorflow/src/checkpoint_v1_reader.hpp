// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sys/stat.h>

#include <unordered_map>
#include <vector>

#include "checkpoint_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/frontend/exception.hpp"
#include "ov_tensorflow/saved_tensor_slice.pb.h"
#include "ov_tensorflow/tensor_shape.pb.h"
#include "ov_tensorflow/types.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {
// stores information about shape, type, and shard id for Variable
struct VariableInfo {
    ::tensorflow::TensorShapeProto variable_shape;
    ::tensorflow::DataType variable_type;
    int32_t shard_id;
    size_t offset;
    size_t size;
    std::vector<int64_t> starts;
    std::vector<int64_t> lenghts;
};

// reads checkpoints of v1 version
// it parses value, shape and type for Variable nodes
class CheckpointV1Reader {
    const std::string m_checkpoints;
    // a map from Variable name to its informations
    std::unordered_map<std::string, VariableInfo> m_variables_info_map;
    // a vector of streams for shards, where shard is one checkpoint file
    std::vector<std::shared_ptr<std::ifstream>> m_shards;
    // a vector of shard names
    std::vector<std::string> m_shard_names;

public:
    /// \brief constructs CheckpointV1Reader for a given directory of checkpoint files
    // CheckpointV1Reader(const std::string& checkpoints_dir);
    CheckpointV1Reader(const std::string& checkpoints);

    /// \brief initialize Checkpoint V1 reader
    void initialize();

    /// \brief Produces ov::Any object that wraps ov::Tensor for the requested variable
    /// it can also wraps string tensor
    /// \param variable_name the requested variable name
    /// \param a reference to the result
    void read_variable(const std::string& variable_name, ov::Any& data);

private:
    /// \brief finds non-master key entry that uses already cached offset and sizes of data blocks
    void find_entry(const std::shared_ptr<std::ifstream>& shard,
                    const std::string& shard_name,
                    const std::string& entry_key,
                    std::string& value);

    void seek_block(const std::string& shard_name,
                    const std::string& target,
                    const char* shard_data,
                    const uint32_t restarts,
                    std::string& value) const;

    void init_block(const std::shared_ptr<std::ifstream>& shard,
                    const std::string& shard_name,
                    uint64_t offset,
                    uint64_t size,
                    std::string& block,
                    uint64_t& restart_offset) const;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
