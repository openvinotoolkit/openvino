// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "checkpoint_v1_reader.hpp"

#include "checkpoint_utils.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/util/file_util.hpp"
#include "ov_tensorflow/saved_tensor_slice.pb.h"
#include "tf_utils.hpp"

#ifdef ENABLE_SNAPPY_COMPRESSION
#    include "snappy.h"
#endif

using namespace ov::frontend::tensorflow;

namespace {
std::vector<std::string> list_files_in_dir(const std::string& directory_path) {
    std::vector<std::string> res;
    try {
        ov::util::iterate_files(
            directory_path,
            [&res](const std::string& file_path, bool is_dir) {
                auto file = ov::util::get_file_name(file_path);
                if (!is_dir) {
                    res.push_back(file_path);
                }
            },
            false,
            true);
    } catch (...) {
        // Ignore exceptions
    }
    return res;
}
}  // namespace

CheckpointV1Reader::CheckpointV1Reader(const std::string& checkpoints) : m_checkpoints(checkpoints) {}

void CheckpointV1Reader::initialize() {
    // figure out if the input is a file or a directory of checkpoints
    std::vector<std::string> checkpoints_paths;
    if (ov::util::directory_exists(m_checkpoints)) {
        checkpoints_paths = list_files_in_dir(m_checkpoints);
    } else if (ov::util::file_exists(m_checkpoints)) {
        checkpoints_paths = {m_checkpoints};
    } else {
        FRONT_END_GENERAL_CHECK(false, "[TensorFlow Frontend] incorrect checkpoint: the checkpoint does not exist");
    }

    m_variables_info_map.clear();

    for (auto checkpoint_path : checkpoints_paths) {
        // create ifstream for each shard
        std::shared_ptr<std::ifstream> shard_stream =
            std::make_shared<std::ifstream>(checkpoint_path, std::ifstream::in | std::ifstream::binary);
        FRONT_END_GENERAL_CHECK(
            shard_stream && shard_stream->is_open(),
            "[TensorFlow Frontend] incorrect model: checkpoint file " + checkpoint_path + "does not exist");
        const int32_t shard_ind = static_cast<int32_t>(m_shards.size());
        m_shards.push_back(shard_stream);
        m_shard_names.push_back(checkpoint_path);
        std::string value;
        find_entry(shard_stream, checkpoint_path, SAVED_TENSOR_SLICES_KEY, value);

        // parse empty index block
        // This is only present at the first item of each checkpoint file and serves
        // as a table of contents, listing all the tensor slices saved in this file.
        ::tensorflow::SavedTensorSlices sts;
        FRONT_END_GENERAL_CHECK(sts.ParseFromArray(value.data(), static_cast<int>(value.size())),
                                "[TensorFlow Frontend] incorrect input checkpoint file or internal error: cannot parse "
                                "SavedTensorSlices entry");
        for (const auto& saved_slice_meta : sts.meta().tensor()) {
            // parse shapes and types for variables
            VariableInfo var_info;
            var_info.shard_id = shard_ind;
            auto variable_name = saved_slice_meta.name();  // original variable name (not encoded)
            var_info.variable_shape = saved_slice_meta.shape();
            var_info.variable_type = saved_slice_meta.type();

            // save starts and lenghts of slices for variable name encoding
            for (const auto& slice : saved_slice_meta.slice()) {
                // var_info.starts.push_back(slice.extent())
                for (const auto& extent : slice.extent()) {
                    var_info.starts.push_back(extent.start());
                    if (extent.has_length()) {
                        var_info.lenghts.push_back(extent.length());
                    } else {
                        var_info.lenghts.push_back(-1);
                    }
                }
            }
            m_variables_info_map[variable_name] = var_info;
        }
    }
}

void CheckpointV1Reader::seek_block(const std::string& shard_name,
                                    const std::string& target_key,
                                    const char* block_ptr,
                                    const uint32_t restarts,
                                    std::string& value) const {
    // parsing the next key starts at the end of value, so set value accordingly
    const char* curr_value_pos = block_ptr;
    const char* limit = block_ptr + restarts;  // restarts come right after data
    std::string key = "";

    bool is_found = false;
    while (true) {
        FRONT_END_GENERAL_CHECK(
            curr_value_pos < limit,
            "[TensorFlow Frontend] incorrect model: no more entries to return, invalid checkpoint file " + shard_name);

        // decode next entry
        // each entry looks as follows:
        // | shared (1 byte) | non-shared (1 byte) | value_length (1 byte) | key (non-shared bytes) |
        // | value (value_length bytes) |
        uint32_t shared, non_shared, value_length;
        curr_value_pos = decode_entry(curr_value_pos, limit, shared, non_shared, value_length);
        FRONT_END_GENERAL_CHECK(
            curr_value_pos && key.size() >= shared,
            "[TensorFlow Frontend] incorrect model: corruption error in checkpoint file " + shard_name);

        key.resize(shared);
        key.append(curr_value_pos, non_shared);
        value = std::string(curr_value_pos + non_shared, value_length);
        curr_value_pos += (non_shared + value_length);

        if (key.compare(target_key) >= 0) {
            is_found = true;
            break;
        }
    }
    FRONT_END_GENERAL_CHECK(
        is_found,
        "[TensorFlow Frontend] incorrect input model: checkpoint file " + shard_name + " can be incorrect");
}

void CheckpointV1Reader::init_block(const std::shared_ptr<std::ifstream>& shard,
                                    const std::string& shard_name,
                                    uint64_t offset,
                                    uint64_t size,
                                    std::string& block,
                                    uint64_t& restart_offset) const {
    // check a size of the shard
    FRONT_END_GENERAL_CHECK(shard,
                            "[TensorFlow Frontend] internal error: nullptr pointer to checkpoint file " + shard_name);
    shard->seekg(0, shard->end);
    uint64_t shard_size = static_cast<uint64_t>(shard->tellg());
    FRONT_END_GENERAL_CHECK(offset < shard_size,
                            "[TensorFlow Frontend] internal error or inconsistent checkpoint file: block offset is "
                            "out-of-range for checkpoint file " +
                                shard_name);
    auto n = size + BLOCK_TRAILER_SIZE;
    FRONT_END_GENERAL_CHECK(n < (shard_size - offset),
                            "[TensorFlow Frontend] internal error or inconsistent checkpoint file: block size is "
                            "out-of-range for checkpoint file " +
                                shard_name);

    // read a block and decompress if needed
    std::vector<char> buf(n);
    shard->seekg(offset);
    shard->read(buf.data(), n);
#ifndef ENABLE_SNAPPY_COMPRESSION
    FRONT_END_GENERAL_CHECK(buf[size] == 0,
                            "[TensorFlow Frontend] internal error: compression method for given block is not supported "
                            "for checkpoint file " +
                                shard_name);
    block = std::string(buf.data(), size);
#else
    FRONT_END_GENERAL_CHECK(buf[size] == 0 || buf[size] == 1,
                            "[TensorFlow Frontend] internal error: compression method for given block is not supported "
                            "for checkpoint file " +
                                shard_name);
    if (buf[size] == 1) {
        size_t uncompressed_length = 0;
        FRONT_END_GENERAL_CHECK(
            snappy::GetUncompressedLength(buf.data(), n, &uncompressed_length),
            "[TensorFlow Frontend] internal error: cannot retrieve uncompressed block length for checkpoint file " +
                shard_name);
        std::string uncompressed_string;
        block.clear();
        block.reserve(uncompressed_length);
        snappy::Uncompress(buf.data(), n, &block);
    } else {
        block = std::string(buf.data(), size);
    }
#endif
    const char* data = block.data();
    size = block.size();

    // find block characteristics: max_restarts_allowed, num_restarts and restart_offset
    FRONT_END_GENERAL_CHECK(
        size >= sizeof(uint32_t),
        "[TensorFlow Frontend] internal error: block size must be not less than 4 bytes in checkpoint file " +
            shard_name);
    size_t max_restarts_allowed = (size - sizeof(uint32_t)) / sizeof(uint32_t);
    uint32_t num_restarts = decode_fixed32(data + size - sizeof(uint32_t));
    FRONT_END_GENERAL_CHECK(
        num_restarts <= max_restarts_allowed,
        "[TensorFlow Frontend] internal error: num_restarts is greater than max_restarts_allowed in checkpoint file " +
            shard_name);
    restart_offset = size - (1 + num_restarts) * sizeof(uint32_t);
}

void CheckpointV1Reader::find_entry(const std::shared_ptr<std::ifstream>& shard,
                                    const std::string& shard_name,
                                    const std::string& entry_key,
                                    std::string& entry_value) {
    // read footer of the shard file to get offset and size of index block
    VIFooter footer;
    footer.read(*shard);
    uint64_t block_offset = footer.m_index.m_offset;
    uint64_t block_size = footer.m_index.m_size;
    std::string block;

    // initialize index block
    uint64_t restart_offset = 0;
    init_block(shard, shard_name, block_offset, block_size, block, restart_offset);

    // seek entry in the index block
    // this entry contains offset and size of the data block
    seek_block(shard_name, entry_key, block.data(), static_cast<uint32_t>(restart_offset), entry_value);

    // initialize the data block
    FRONT_END_GENERAL_CHECK(
        get_varint64(entry_value, &block_offset) && get_varint64(entry_value, &block_size),
        "[TensorFlow Frontend] incorrect input model: bad block handle in checkpoint file " + shard_name);
    init_block(shard, shard_name, block_offset, block_size, block, restart_offset);

    // seek the final entry in the data block
    seek_block(shard_name, entry_key, block.data(), static_cast<uint32_t>(restart_offset), entry_value);
}

void CheckpointV1Reader::read_variable(const std::string& variable_name, ov::Any& data) {
    FRONT_END_GENERAL_CHECK(m_variables_info_map.count(variable_name) > 0,
                            "[TensorFlow Frontend] incorrect input model: checkpoint files does not contain data for "
                            "the required variable " +
                                variable_name);
    auto var_info = m_variables_info_map[variable_name];
    auto shard_id = m_variables_info_map[variable_name].shard_id;
    FRONT_END_GENERAL_CHECK(shard_id < static_cast<int32_t>(m_shards.size()),
                            "[TensorFlow Frontend] internal error: shard_id is greater than a number of shards");
    FRONT_END_GENERAL_CHECK(
        m_shards.size() == m_shard_names.size(),
        "[TensorFlow Frontend] internal error: number of shards does not match a number of their names");
    auto shard_ptr = m_shards[shard_id];
    auto shard_name = m_shard_names[shard_id];
    auto encoded_name = encode_tensor_name_slice(variable_name, var_info.starts, var_info.lenghts);
    std::string raw_data;
    find_entry(shard_ptr, shard_name, encoded_name, raw_data);

    // This is only present at the first item of each checkpoint file and serves
    // as a table of contents, listing all the tensor slices saved in this file.
    ::tensorflow::SavedTensorSlices sts{};
    FRONT_END_GENERAL_CHECK(sts.ParseFromArray(raw_data.data(), static_cast<int>(raw_data.size())),
                            "[TensorFlow Frontend] incorrect input checkpoint file or internal error: cannot parse "
                            "SavedTensorSlices entry");
    data = unpack_tensor_proto(sts.data().data(), var_info.variable_shape, var_info.variable_type);
}
