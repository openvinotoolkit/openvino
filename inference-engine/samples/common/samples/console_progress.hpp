// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <iomanip>

/**
 * @class ConsoleProgress
 * @brief A ConsoleProgress class provides functionality for printing progress dynamics
 */
class ConsoleProgress {
    static const int DEFAULT_DETALIZATION = 20;

    size_t total;
    size_t current = 0;
    bool stream_output;
    size_t detalization;

public:
    /**
    * @brief A constructor of ConsoleProgress class
    * @param _total - maximum value that is correspondent to 100%
    * @param _detalization - number of symbols(.) to use to represent progress
    */
    explicit ConsoleProgress(size_t _total, bool _stream_output = false, size_t _detalization = DEFAULT_DETALIZATION) :
            total(_total), detalization(_detalization) {
        stream_output = _stream_output;
        if (total == 0) {
            total = 1;
        }
        std::cout << std::unitbuf;
    }

    /**
     * @brief Shows progress with current data. Progress is shown from the beginning of the current line.
     * @return
     */
    void showProgress() const {
        std::stringstream strm;
        if (!stream_output) {
            strm << '\r';
        }
        strm << "Progress: [";
        size_t i = 0;
        for (; i < detalization * current / total; i++) {
            strm << ".";
        }
        for (; i < detalization; i++) {
            strm << " ";
        }
        strm << "] " << std::fixed << std::setprecision(2) << 100 * static_cast<float>(current) / total << "% done";
        if (stream_output) {
            std::cout << strm.str() << std::endl;
        } else {
            std::cout << strm.str() << std::flush;
        }
    }

    /**
     * @brief Updates current value and progressbar
     * @param newProgress - new value to represent
     */
    void updateProgress(size_t newProgress) {
        current = newProgress;
        if (current > total) current = total;
        showProgress();
    }

    /**
     * @brief Adds value to currently represented and redraw progressbar
     * @param add - value to add
     */
    void addProgress(int add) {
        if (add < 0 && -add > static_cast<int>(current)) {
            add = -static_cast<int>(current);
        }
        updateProgress(current + add);
    }

    /**
     * @brief Output end line.
     * @return
     */
    void finish() {
        std::cerr << std::nounitbuf << "\n";
    }
};
