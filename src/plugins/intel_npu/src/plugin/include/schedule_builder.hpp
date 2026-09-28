// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstdint>
#include <map>
#include <regex>
#include <vector>

namespace intel_npu {
using elf_binary = std::vector<uint8_t>;
using io_map = std::map<uint32_t, uint32_t>;
using parameter_map = std::map<std::string, std::vector<uint32_t>>;

enum class BuilderOption { none, io_reuse, io_iterate, io_consolidate, io_global, io_name, 
    io_mapping,
    fuse_elf,
    memset,
    compress,
    branch,
    static_dma,
    version,
    chunk_dma,
    tiny_dma
};

static std::map<BuilderOption, std::string> builder_option_name{{BuilderOption::io_global, "IO_GLOBAL"},
                                                                {BuilderOption::io_consolidate, "IO_CONSOLIDATE"},
                                                                {BuilderOption::io_iterate, "IO_ITERATE"},
                                                                {BuilderOption::io_reuse, "IO_REUSE"},
                                                                {BuilderOption::io_name, "IO_NAME"},
                                                                {BuilderOption::io_mapping, "IO_MAPPING"},
                                                                {BuilderOption::fuse_elf, "FUSE_ELF"},
                                                                {BuilderOption::memset, "MEMSET"},
                                                                {BuilderOption::compress, "COMPRESS"},
                                                                {BuilderOption::branch, "BRANCH"},
                                                                {BuilderOption::static_dma, "STATIC_DMA"},
                                                                {BuilderOption::chunk_dma, "CHUNK_DMA"},
                                                                {BuilderOption::tiny_dma, "TINY_DMA"},
                                                                {BuilderOption::version, "VERSION"},
                                                                {BuilderOption::none, "NONE"}};

static uint32_t get_io_port_mapping(std::vector<uint8_t>& io_text,
                                    io_map& global_inputs,
                                    io_map& global_outputs,
                                    parameter_map& global_paramaters) {
    global_inputs.clear();
    global_outputs.clear();
    global_paramaters.clear();

    uint32_t global_parameter_count{};
    std::vector<std::string> input_ports;
    std::vector<std::string> output_ports;

    {
        std::string io_mapping_ports(std::begin(io_text), std::end(io_text));

        std::istringstream iss(io_mapping_ports);

        if (!iss.good())
            return 0u;

        std::string line_in;

        int32_t input_count{-1};
        int32_t output_count{-1};

        while (!iss.eof()) {
            getline(iss, line_in);

            if (input_count == -1) {
                input_count = atoi(line_in.c_str());
                input_ports.reserve(input_count);
            } else if (input_count > 0) {
                input_ports.push_back(line_in);
                input_count--;
            } else if (output_count == -1) {
                output_count = atoi(line_in.c_str());
                output_ports.reserve(output_count);
            } else if (output_count > 0) {
                output_ports.push_back(line_in);
                output_count--;
            }
        }
    }

    std::map<std::string, std::deque<uint32_t>> input_params{};

    {
        auto get_base_param_name = [](std::string& name) {
            std::regex pattern("s(\\d+).");
            std::smatch matches;

            if (std::regex_search(name, matches, pattern)) {
                const auto prefix_len{matches[0].length()};
                name = name.substr(prefix_len, name.length() - prefix_len);

                std::regex pattern_iter(".iter(\\d+)");
                std::smatch matches_iter;

                if (std::regex_search(name, matches_iter, pattern_iter)) {
                    name = name.substr(0, name.length() - matches_iter[0].length());
                }
            } 
            else {
                std::regex pattern_iter(".iter(\\d+)");
                std::smatch matches_iter;

                if (std::regex_search(name, matches_iter, pattern_iter)) {
                    auto name_clipped{name.substr(0, name.length() - matches_iter[0].length())};

                    const auto cat_length{name_clipped.length() + matches_iter[0].str().length()};

                    if (cat_length == name.length()) {
                        name = name_clipped;
                    }
                }
            }
        };

        uint32_t indx{};
        for (auto& name : input_ports) {
            auto io_marker_pos{name.find("#GI[")};

            if (io_marker_pos != std::string::npos) {
                auto param_name{name};
                get_base_param_name(param_name);
                auto end_marker{name.find("]_")};
                const auto start{io_marker_pos + 4ull};
                auto num_str{name.substr(start, end_marker - start)};
                global_inputs[static_cast<uint32_t>(atol(num_str.c_str()))] = indx;

                end_marker = param_name.find("]_");
                end_marker += 2ull;

                param_name = param_name.substr(end_marker, param_name.length() - end_marker);
                input_params[param_name].push_back(indx);
            } else {
                get_base_param_name(name);
                input_params[name].push_back(indx);
            }

            indx++;
        }

        for (auto& param : input_params) {
            auto& global_param{global_paramaters[param.first]};
            global_param.reserve(param.second.size());

            for (auto port : param.second) {
                global_param.push_back(port);
                global_parameter_count++;
            }
        }
    }
    {
        uint32_t indx{};
        for (auto& name : output_ports) {
            auto io_marker_pos{name.find("#GO[")};

            if (io_marker_pos != std::string::npos) {
                auto end_marker{name.find("]_")};
                const auto start{io_marker_pos + 4ull};
                auto num_str{name.substr(start, end_marker - start)};
                global_outputs[static_cast<uint32_t>(atol(num_str.c_str()))] = indx;
            }

            indx++;
        }
    }

    return global_parameter_count;
}
}  // namespace intel_npu
