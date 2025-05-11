// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <iomanip>
#include <sstream>

/**
 * @class ConsoleProgress
 * @brief A ConsoleProgress class provides functionality for printing progress dynamics
 */
class ConsoleProgress {
    static const size_t DEFAULT_DETALIZATION = 20;
    static const size_t DEFAULT_PERCENT_TO_UPDATE_PROGRESS = 1;

    size_t total;
    size_t cur_progress = 0;
    size_t prev_progress = 0;
    bool stream_output;
    size_t detalization;
    size_t percent_to_update;

public:
    /**
     * @brief A constructor of ConsoleProgress class
     * @param _total - maximum value that is correspondent to 100%
     * @param _detalization - number of symbols(.) to use to represent progress
     */
    explicit ConsoleProgress(size_t _total,
                             bool _stream_output = false,
                             size_t _percent_to_update = DEFAULT_PERCENT_TO_UPDATE_PROGRESS,
                             size_t _detalization = DEFAULT_DETALIZATION)
        : total(_total),
          detalization(_detalization),
          percent_to_update(_percent_to_update) {
        stream_output = _stream_output;
        if (total == 0) {
            total = 1;
        }
    }

    /**
     * @brief Shows progress with current data. Progress is shown from the beginning of the current
     * line.
     */
    void showProgress() const {
        std::stringstream strm;
        if (!stream_output) {
            strm << '\r';
        }
        strm << "Progress: [";
        size_t i = 0;
        for (; i < detalization * cur_progress / total; i++) {
            strm << ".";
        }
        for (; i < detalization; i++) {
            strm << " ";
        }
        strm << "] " << std::setw(3) << 100 * cur_progress / total << "% done";
        if (stream_output) {
            strm << std::endl;
        }
        std::fputs(strm.str().c_str(), stdout);
        std::fflush(stdout);
    }

    /**
     * @brief Updates current value and progressbar
     */
    void updateProgress() {
        if (cur_progress > total)
            cur_progress = total;
        size_t prev_percent = 100 * prev_progress / total;
        size_t cur_percent = 100 * cur_progress / total;

        if (prev_progress == 0 || cur_progress == total || prev_percent + percent_to_update <= cur_percent) {
            showProgress();
            prev_progress = cur_progress;
        }
    }

    /**
     * @brief Adds value to currently represented and redraw progressbar
     * @param add - value to add
     */
    void addProgress(int add) {
        if (add < 0 && -add > static_cast<int>(cur_progress)) {
            add = -static_cast<int>(cur_progress);
        }
        cur_progress += add;
        updateProgress();
    }

    /**
     * @brief Output end line.
     * @return
     */
    void finish() {
        std::stringstream strm;
        strm << std::endl;
        std::fputs(strm.str().c_str(), stdout);
        std::fflush(stdout);
    }
};
