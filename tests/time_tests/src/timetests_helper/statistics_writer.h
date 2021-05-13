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

/**
 * @brief Class response for writing provided statistics
 *
 * Object of the class is writing provided statistics to a specified
 * file in YAML format.
 */
class StatisticsWriter {
private:
  std::ofstream statistics_file;
  std::map<std::pair<int, std::string>, std::pair<std::string, float>> time_structure;
  std::map<std::string, int> order_structure;
  int tab_count = 0;
  int order = 0;

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
  void addOrderCount(const std::string name) {
    order++;
    order_structure.insert(std::make_pair(name, order));
    tab_count++;
  }

  void deleteOrderCount() {
    tab_count--;
  }

  /**
   * @brief Writes statistics in map structure.
   */
  void addToTimeStructure(const std::pair<std::string, float> &record) {
    const std::string tab_const = "  ";
    std::string tabs = "";
    for (int i = 0; i < tab_count - 1; ++i) {
      tabs += tab_const;
    }
    time_structure.insert({std::make_pair(order_structure[record.first], tabs), record});
  }

  /**
   * @brief Writes provided statistics in YAML format.
   */
  void write() {
    if (!statistics_file)
      throw std::runtime_error("Statistic file path isn't set");
    for (auto& x: time_structure) {
      statistics_file << (x.first).second << "- " << (x.second).first << ":" << '\n'
                      << (x.first).second << "  " << "- " << (x.second).second << '\n';
    }
    statistics_file << "---" << '\n' << "measurement_unit: microsecs";
  }
};
