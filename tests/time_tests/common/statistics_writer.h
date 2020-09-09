// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <cstdio>


/**
 * @brief Class response for writing provided statistics
 *
 * Object of the class is writing provided statistics to a specified
 * file in YAML format.
 */
class StatisticsWriter {
private:
    std::ofstream statistics_file;

    StatisticsWriter() = default;
    StatisticsWriter(const StatisticsWriter&) = delete;
    StatisticsWriter& operator=(const StatisticsWriter&) = delete;
public:
    /**
     * @brief Creates StatisticsWriter singleton object
     */
    static StatisticsWriter& Instance(){
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
            err << "Statistic file \"" << statistics_path << "\" can't be used for writing";
            throw std::runtime_error(err.str());
        }
    }

    /**
     * @brief Writes provided statistics in YAML format.
     */
    void write(const std::pair<std::string, float> &record) {
        if (!statistics_file)
            throw std::runtime_error("Statistic file path isn't set");
        statistics_file << record.first << ": " << record.second << "\n";
    }
};
