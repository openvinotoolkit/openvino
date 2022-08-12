// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "utils.hpp"

struct InferenceResult {
    ov::TensorVector output_tensors;  // Inference request output
    std::string input_images;         // Multi image files are jointed with commas
};

struct OutputBuff {
    size_t size;
    void* buffer;
    size_t offset;
    size_t iteration;
    std::string binary_filename;
};

inline uint64_t get_bytes_from_MB(uint32_t size) {
    return size * 1024 * 1024LL;
}

/// @brief Responsible for dumping reference results
class ResultDump {
public:
    explicit ResultDump(const std::string& device_name,
                        const std::string& model_name,
                        const std::string& dump_dir,
                        const std::string& output_precision,
                        const uint32_t& output_max_num,
                        const uint32_t& binary_max_size)
        : _device_name(device_name),
          _model_name(model_name),
          _dump_dir(dump_dir),
          _output_precision(output_precision),
          _output_max_num(output_max_num) {
        _binary_file_max_size = get_bytes_from_MB(binary_max_size);
        _separator = get_path_separator();
        auto now_time = get_now_time_string();
        _file_prefix = dump_dir + _separator + "benchmark_output_" + now_time;
        _binary_filepath = _file_prefix + "_" + std::to_string(_binary_file_index) + ".bin";
        _index_filepath = _file_prefix + ".json";
        // If binary file exists, clear the file
        std::ofstream outfile(_binary_filepath, std::ios_base::binary | std::ios_base::out);
        if (outfile.is_open()) {
            outfile.close();
        }
    }

    ~ResultDump() {
        write_json_file();
        for (auto iter = _different_outputs.begin(); iter != _different_outputs.end(); iter++) {
            for (auto& output : iter->second) {
                if (output.buffer) {
                    free(output.buffer);
                    output.buffer = nullptr;
                }
            }
            iter->second.clear();
        }
    }

    void add_result(const std::string& input_image, const ov::TensorVector& outputs);

private:
    bool compare_output(const std::string& input_image, const ov::TensorVector& outputs);
    void write_binary_file(const std::string& input_image, const ov::TensorVector& outputs);
    void write_json_file();

private:
    std::string _device_name;  // Device config
    std::string _model_name;   // Model name
    std::string _dump_dir;     // Dump file path
    std::string _output_precision;

    int _output_max_num;            // For each image, the maximum number of distinct outputs saveds
    int64_t _binary_file_max_size;  // The maximum size of a single binary file
    int _binary_file_index = 0;
    std::string _separator;

    // If network has multiple output, the output datas are sequentially saved in a continuous memory
    std::map<std::string, std::vector<OutputBuff>> _different_outputs;

    std::string _file_prefix;
    std::string _binary_filepath;
    std::string _index_filepath;
};