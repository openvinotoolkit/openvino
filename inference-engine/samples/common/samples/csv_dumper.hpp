// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>

#include <samples/slog.hpp>

/**
 * @class CsvDumper
 * @brief A CsvDumper class provides functionality for dumping the values in CSV files
 */
class CsvDumper {
    std::ofstream file;
    std::string filename;
    bool canDump = true;
    char delimiter = ';';

    std::string generateFilename() {
        std::stringstream filename;
        filename << "dumpfile-";
        filename << time(nullptr);
        filename << ".csv";
        return filename.str();
    }

public:
    /**
     * @brief A constructor. Disables dumping in case dump file cannot be created
     * @param enabled - True if dumping is enabled by default.
     * @param name - name of file to dump to. File won't be created if first parameter is false.
     */
    explicit CsvDumper(bool enabled = true, const std::string& name = "") : canDump(enabled) {
        if (!canDump) {
            return;
        }
        filename = (name == "" ? generateFilename() : name);
        file.open(filename, std::ios::out);
        if (!file) {
            slog::warn << "Cannot create dump file! Disabling dump." << slog::endl;
            canDump = false;
        }
    }

    /**
     * @brief Sets a delimiter to use in csv file
     * @param c - Delimiter char
     * @return
     */
    void setDelimiter(char c) {
        delimiter = c;
    }

    /**
     * @brief Overloads operator to organize streaming values to file. Does nothing if dumping is disabled
     *        Adds delimiter at the end of value provided
     * @param add - value to add to dump
     * @return reference to same object
     */
    template<class T>
    CsvDumper& operator<<(const T& add) {
        if (canDump) {
            file << add << delimiter;
        }
        return *this;
    }

    /**
     * @brief Finishes line in dump file. Does nothing if dumping is disabled
     */
    void endLine() {
        if (canDump) {
            file << "\n";
        }
    }

    /**
     * @brief Gets information if dump is enabled.
     * @return true if dump is enabled and file was successfully created
     */
    bool dumpEnabled() {
        return canDump;
    }

    /**
     * @brief Gets name of a dump file
     * @return name of a dump file
     */
    std::string getFilename() const {
        return filename;
    }
};
