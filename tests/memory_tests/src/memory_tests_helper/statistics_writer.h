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

  std::map<std::string, std::pair<int, std::vector<size_t>>> mem_structure; // mem_counter_name, <tab number, memory measurements>
  std::vector<std::string> mem_struct_order;
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
  void addMemCounter(const std::string name) {
    mem_struct_order.push_back(name);
    tab_count++;
  }

  void deleteMemCounter(const std::pair<std::string, std::vector<size_t>> &record) {
    tab_count--;
    mem_structure[record.first] = std::make_pair(tab_count, record.second);
  }

  /**
   * @brief Writes provided statistics in YAML format.
   */
  void write() {
    if (!statistics_file)
      throw std::runtime_error("Statistic file path isn't set");
    for (auto& mem_counter: mem_struct_order) {
      std::string tabs = std::string(TAB* mem_structure[timer].first, ' ');
      statistics_file << tabs << "- " << mem_counter << ":" << '\n'
                      << tabs << "  " << "- " << mem_structure[mem_counter].second << '\n'; # TODO: update print of mem_structure[timer].second
    }
    statistics_file << "---" << '\n' << "measurement_unit: Kb";
  }
};
