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

#define TAB 2

/**
 * @brief Class response for writing provided statistics
 *
 * Object of the class is writing provided statistics to a specified
 * file in YAML format.
 */
class StatisticsWriter {
private:
    std::ofstream statistics_file;

    std::map<std::string, std::pair<int, float>> time_structure; // timer_name, <tab number, duration>
    std::vector<std::string> time_struct_order;
    int tab_count = 0;

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
 * @brief Compute order for statistics operations.
 */
void addTimer(const std::string name) {
    time_struct_order.push_back(name);
    tab_count++;
}

void deleteTimer(const std::pair<std::string, float> &record) {
    tab_count--;
    time_structure[record.first] = std::make_pair(tab_count, record.second);
}

/**
 * @brief Writes provided statistics in YAML format.
 */
void write() {
    if (!statistics_file)
        throw std::runtime_error("Statistic file path isn't set");
    for (auto& timer: time_struct_order) {
        std::string tabs = std::string(TAB* time_structure[timer].first, ' ');
        statistics_file << tabs << "- " << timer << ":" << '\n'
                        << tabs << "  " << "- " << time_structure[timer].second << '\n';
    }
    statistics_file << "---" << '\n' << "measurement_unit: microsecs";
}
};
