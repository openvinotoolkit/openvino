// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "result_dump.hpp"

#include <nlohmann/json.hpp>

#include "samples/slog.hpp"

void ResultDump::add_result(const std::string& input_image, const ov::TensorVector& outputs) {
    // Compare with exist output, if it is different, save it to buffer and write it to binary file.
    if (compare_output(input_image, outputs)) {
        // Check if the number of outputs has reached the maximum value
        // Stop saving if the number of different outputs exceeds the maximum
        auto it = _different_outputs.find(input_image);
        if (it != _different_outputs.end()) {
            if (it->second.size() >= _output_max_num) {
                slog::warn << "The different output size exceeds the maximum for image:" << input_image << slog::endl;
                return;
            }
        }
        write_binary_file(input_image, outputs);
    }
}

/**
 * @brief      Compare output with previous output
 * @param[in]  inputimage Input image for inference request
 * @param[in]  outputs Output tensor for inference request
 * @return true if this output is different from previous output, otherwise false
 */
bool ResultDump::compare_output(const std::string& input_image, const ov::TensorVector& outputs) {
    auto iter = _different_outputs.find(input_image);
    if (iter == _different_outputs.end()) {
        _different_outputs[input_image].reserve(_output_max_num);
        return true;
    }

    auto output_size = outputs.size();
    bool different = true;
    for (auto& exist_outputs : iter->second) {
        size_t offset = 0;
        auto old_output = (char*)exist_outputs.buffer;
        bool same = true;
        for (size_t i = 0; i < output_size; i++) {
            auto buffsize = outputs[i].get_byte_size();
            auto new_output = outputs[i].data();
            if (offset + buffsize > exist_outputs.size) {
                slog::err << "New outputs size is different with the exist outputs size" << slog::endl;
                same = false;
                break;
            } else {
                if (0 != std::memcmp(new_output, old_output, buffsize)) {
                    same = false;
                    break;
                }
            }
            offset += buffsize;
            old_output += buffsize;
        }

        if (same) {
            different = false;
            exist_outputs.iteration++;
            break;
        }
    }

    return different;
}

/**
 * @brief      Write the output data to the binary file
 * @param[in]  inputimage input image for inference request
 * @param[in]  outputs output tensor for inference request
 */
void ResultDump::write_binary_file(const std::string& input_image, const ov::TensorVector& outputs) {
    // Write binary file and record the offset of the binary file
    std::ofstream binfile(_binary_filepath, std::ios_base::binary | std::ios_base::app);
    if (binfile.is_open()) {
        OutputBuff output_buff;
        binfile.seekp(0, std::ios_base::end);
        output_buff.offset = static_cast<std::size_t>(binfile.tellp());
        output_buff.size = 0;
        output_buff.iteration = 1;
        // If the file size exceeds the maximum, open a new file
        if (output_buff.offset >= _binary_file_max_size) {
            binfile.close();
            _binary_file_index++;
            _binary_filepath = _file_prefix + "_" + std::to_string(_binary_file_index) + ".bin";
            binfile.open(_binary_filepath, std::ios_base::binary | std::ios_base::out);
            if (binfile.is_open()) {
                output_buff.offset = 0;
            } else {
                slog::warn << "fail to open file " << _binary_filepath << slog::endl;
                return;
            }
        }
        for (auto& output : outputs) {
            auto size = output.get_byte_size();
            binfile.write(reinterpret_cast<const char*>(output.data()), size);
            output_buff.size += size;
        }
        binfile.close();

        // Record the index of binary file
        auto iter = _different_outputs.find(input_image);
        if (iter == _different_outputs.end()) {
            iter = _different_outputs.insert(std::make_pair(input_image, std::vector<OutputBuff>())).first;
        }

        output_buff.binary_filename = _binary_filepath;

        // Copy multiple output data sequentially to a continuous memory
        output_buff.buffer = (void*)malloc(output_buff.size);
        if (nullptr == output_buff.buffer) {
            slog::warn << "malloc buffer fail" << slog::endl;
        } else {
            auto output_dst = (char*)output_buff.buffer;
            for (auto& output : outputs) {
                std::memcpy(output_dst, output.data(), output.get_byte_size());
                output_dst += output.get_byte_size();
            }
            iter->second.push_back(output_buff);
        }
    } else {
        slog::warn << "fail to open file " << _binary_filepath << slog::endl;
    }
}

/**
 * @brief      Write the index files of the binary to a json file
 */
void ResultDump::write_json_file() {
    nlohmann::json json_array = nlohmann::json::array();
    for (auto iter = _different_outputs.begin(); iter != _different_outputs.end(); iter++) {
        nlohmann::json json_image_object = nlohmann::json::object();
        json_image_object["image"] = iter->first;
        for (auto& index : iter->second) {
            nlohmann::json json_object = nlohmann::json::object();
            json_object["offset"] = index.offset;
            json_object["size"] = index.size;
            json_object["binary_name"] = index.binary_filename;
            json_object["iteration"] = index.iteration;
            json_image_object["datas"].emplace_back(json_object);
        }
        json_array.emplace_back(json_image_object);
    }

    nlohmann::json json_file;
    json_file["device"] = _device_name;
    json_file["model"]["name"] = _model_name;
    json_file["model"]["output_precision"] = _output_precision;
    json_file["configure"]["binary_maxsize"] = _binary_file_max_size;
    json_file["configure"]["output_maxnum"] = _output_max_num;
    json_file["outputs"] = json_array;

    std::ofstream out(_index_filepath);
    if (!out.is_open()) {
        slog::warn << "Can't dump file " << _index_filepath << slog::endl;
    } else {
        out << json_file;
    }
}
