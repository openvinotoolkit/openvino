// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <map>

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
    mem_structure[record.first] = record.second;
  }

  /**
   * @brief Writes provided statistics in YAML format.
   */
  void write() {
    if (!statistics_file)
      throw std::runtime_error("Statistic file path isn't set");
    for (auto& mem_counter: mem_structure) {
      statistics_file << mem_counter.first << ":" << '\n'
                      << SEPARATOR << "vmrss: " << mem_counter.second[0] << '\n'
                      << SEPARATOR << "vmhwm: " << mem_counter.second[1] << '\n'
                      << SEPARATOR << "vmsize: " << mem_counter.second[2] << '\n'
                      << SEPARATOR << "vmpeak: " << mem_counter.second[3] << '\n'
                      << SEPARATOR << "threads: " << mem_counter.second[4] << '\n';
    }
    statistics_file << "---" << '\n' << "measurement_unit: Kb";
  }
};
