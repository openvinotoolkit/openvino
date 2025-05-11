// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <map>
#include <vector>

#define SEPARATOR " - "

/**
 * @brief Class response for writing provided statistics
 *
 * Object of the class is writing provided statistics to a specified
 * file in YAML format.
 */
class StatisticsWriter {
private:
    std::ofstream statistics_file;

    std::map<std::string, std::vector<size_t>> mem_structure; // mem_counter_name, memory measurements
    std::vector<std::string> mem_struct_order;

    StatisticsWriter() = default;

    StatisticsWriter(const StatisticsWriter &) = delete;

    StatisticsWriter &operator=(const StatisticsWriter &) = delete;

public:
    /**
     * @brief Creates StatisticsWriter singleton object
     */
    static StatisticsWriter &Instance() {
        static StatisticsWriter writer;
        return writer;
    }

    /**
     * @brief Specifies, opens and validates statistics path for writing
     */
    void setFile(const std::string &statistics_path) {
        statistics_file.open(statistics_path);
        if (!statistics_file.good()) {
            std::stringstream err;
            err << "Statistic file \"" << statistics_path
                << "\" can't be used for writing";
            throw std::runtime_error(err.str());
        }
    }

    /**
     * @brief Creates counter structure
     */
    void addMemCounterToStructure(const std::pair<std::string, std::vector<size_t>> &record) {
        mem_struct_order.push_back(record.first);
        mem_structure[record.first] = record.second;
    }

    /**
     * @brief Writes provided statistics in YAML format.
     */
    void write() {
        if (!statistics_file)
            throw std::runtime_error("Statistic file path isn't set");
        for (auto &mem_counter: mem_struct_order) {
            statistics_file << mem_counter << ":" << '\n'
                            << SEPARATOR << "vmrss: " << mem_structure[mem_counter][0] << '\n'
                            << SEPARATOR << "vmhwm: " << mem_structure[mem_counter][1] << '\n'
                            << SEPARATOR << "vmsize: " << mem_structure[mem_counter][2] << '\n'
                            << SEPARATOR << "vmpeak: " << mem_structure[mem_counter][3] << '\n'
                            << SEPARATOR << "threads: " << mem_structure[mem_counter][4] << '\n';
        }
        statistics_file << "---" << '\n' << "measurement_unit: Kb";
    }
};
