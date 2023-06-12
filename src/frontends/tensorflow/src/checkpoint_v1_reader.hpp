// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sys/stat.h>

#include <unordered_map>
#include <vector>

#include "checkpoint_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/frontend/exception.hpp"
#include "saved_tensor_slice.pb.h"
#include "tensor_shape.pb.h"
#include "types.pb.h"

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
    bool is_initialized;
};

// reads checkpoints of v1 version
// it parses value, shape and type for Variable nodes
class CheckpointV1Reader {
    // a map from Variable name to its informations
    std::unordered_map<std::string, VariableInfo> m_variables_info_map;
    // a vector of streams for shards, where shard is one checkpoint file
    std::vector<std::shared_ptr<std::ifstream>> m_shards;

public:
    /// \brief constructs CheckpointV1Reader for a given directory of checkpoint files
    // CheckpointV1Reader(const std::string& checkpoints_dir);
    CheckpointV1Reader(const std::string& checkpoints);

    /// \brief Produces ov::Any object that wraps ov::Tensor for the requested variable
    /// it can also wraps string tensor
    /// \param variable_name the requested variable name
    /// \param a reference to the result
    void read_variable(const std::string& variable_name, ov::Any& data);

private:
    void find_entry(const std::shared_ptr<std::ifstream>& shard, const std::string& entry_key, std::string& value);

    void seek_block(const std::string& target,
                    const char* shard_data,
                    const uint32_t restarts,
                    std::string& value) const;

    void init_block(const std::shared_ptr<std::ifstream>& shard,
                    uint64_t offset,
                    uint64_t size,
                    std::string& block,
                    uint64_t& restart_offset) const;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
