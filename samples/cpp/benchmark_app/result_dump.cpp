// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "result_dump.hpp"

#include <nlohmann/json.hpp>

#include "samples/slog.hpp"

void ResultDump::add_result(const std::string& input_image, const ov::TensorVector& outputs) {
    // Compare with exist output, if it is different, save it to buffer and write it to binary file.
    if (compare_output(input_image, outputs)) {
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
        _different_outputs[input_image].reserve(_different_max_num);
        return true;
    }

    auto output_size = outputs.size();
    bool different = false;
    for (auto& exist_outputs : iter->second) {
        size_t offset = 0;
        auto old_output = (char*)exist_outputs.buffer;
        for (size_t i = 0; i < output_size; i++) {
            auto buffsize = outputs[i].get_byte_size();
            auto new_output = outputs[i].data();
            if (offset + buffsize > exist_outputs.size) {
                slog::err << "New outputs size is different with the exist outputs size" << slog::endl;
                break;
            } else {
                if (0 != std::memcmp(new_output, old_output, buffsize)) {
                    different = true;
                    break;
                }
            }
            offset += buffsize;
            old_output += buffsize;
        }
        if (different)
            break;
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
        OutputIndex index;
        index.offset = static_cast<std::size_t>(binfile.tellp());
        index.size = 0;
        for (auto& output : outputs) {
            auto size = output.get_byte_size();
            binfile.write(reinterpret_cast<const char*>(output.data()), size);
            index.size += size;
        }
        binfile.close();

        // Record the index of binary file
        auto iter = _outputs_index.find(input_image);
        if (iter == _outputs_index.end()) {
            iter = _outputs_index.insert(std::make_pair(input_image, std::vector<OutputIndex>())).first;
        }

        iter->second.push_back(index);

        // Copy multiple output data sequentially to a continuous memory
        OutputBuff output_buff;
        output_buff.size = index.size;
        output_buff.buffer = (void*)malloc(index.size);
        if (nullptr == output_buff.buffer) {
            slog::warn << "malloc buffer fail" << slog::endl;
        } else {
            auto output_dst = (char*)output_buff.buffer;
            for (auto& output : outputs) {
                std::memcpy(output_dst, output.data(), output.get_byte_size());
                output_dst += output.get_byte_size();
            }
            // If the number of different output exceed the maximum for one image, override the first output
            if (_different_outputs[input_image].size() >= _different_max_num) {
                if (_different_outputs[input_image][_different_max_num - 1].buffer) {
                    free(_different_outputs[input_image][_different_max_num - 1].buffer);
                    _different_outputs[input_image][_different_max_num - 1].buffer = nullptr;
                }
                _different_outputs[input_image][_different_max_num - 1] = output_buff;
            } else {
                _different_outputs[input_image].emplace_back(output_buff);
            }
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
    for (auto iter = _outputs_index.begin(); iter != _outputs_index.end(); iter++) {
        nlohmann::json json_image_object = nlohmann::json::object();
        json_image_object["image"] = iter->first;
        for (auto& index : iter->second) {
            nlohmann::json jsonObject = nlohmann::json::object();
            jsonObject["offset"] = index.offset;
            jsonObject["size"] = index.size;
            json_image_object["datas"].emplace_back(jsonObject);
        }
        json_array.emplace_back(json_image_object);
    }

    nlohmann::json jsonFile;
    jsonFile["device"] = _device_name;
    jsonFile["model"]["name"] = _model_name;
    jsonFile["outputs"] = json_array;

    std::ofstream out(_index_filepath);
    if (!out.is_open()) {
        slog::warn << "Can't dump file " << _index_filepath << slog::endl;
    } else {
        out << jsonFile;
    }
}
