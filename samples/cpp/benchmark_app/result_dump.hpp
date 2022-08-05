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

#define MAX_SAVE_INFERENCE_RESULT 10

struct InferenceResult {
    ov::TensorVector output_tensors;  // Inference request output
    std::string input_images;         // Multi image files are jointed with commas
};

struct OutputBuff {
    size_t size;
    void* buffer;
};

struct OutputIndex {
    size_t offset;
    size_t size;
};

/// @brief Responsible for dumping reference results
class ResultDump {
public:
    explicit ResultDump(const std::string& device_name, const std::string& model_name, const std::string& dump_dir)
        : _device_name(device_name),
          _model_name(model_name),
          _dump_dir(dump_dir) {
        _separator = get_path_separator();
        auto now_time = get_now_time_string();
        std::string file_prefix = dump_dir + _separator + "benchmark_output_" + now_time;
        _binary_filepath = file_prefix + ".bin";
        _index_filepath = file_prefix + ".json";
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
    int _different_max_num =
        MAX_SAVE_INFERENCE_RESULT;  // For each image, the maximum number of distinct outputs saveds
    std::string _separator;

    // If network has multiple output, the output datas are sequentially saved in a continuous memory
    std::map<std::string, std::vector<OutputBuff>> _different_outputs;

    std::map<std::string, std::vector<OutputIndex>> _outputs_index;
    std::string _binary_filepath;
    std::string _index_filepath;
};