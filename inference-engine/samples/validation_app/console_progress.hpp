// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    size_t detalization;

public:
    /**
    * @brief A constructor of ConsoleProgress class
    * @param _total - maximum value that is correspondent to 100%
    * @param _detalization - number of symbols(.) to use to represent progress
    */
    explicit ConsoleProgress(size_t _total, size_t _detalization = DEFAULT_DETALIZATION) :
            total(_total), detalization(_detalization) {
        if (total == 0) {
            total = 1;
        }
    }

    /**
     * @brief Shows progress with current data. Progress is shown from the beginning of the current line.
     * @return
     */
    void showProgress() const {
        std::cout << "\rProgress: [";
        size_t i = 0;
        for (; i < detalization * current / total; i++) {
            std::cout << ".";
        }
        for (; i < detalization; i++) {
            std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(2) << 100 * static_cast<float>(current) / total << "% done    ";
        std::flush(std::cout);
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
        if (add < 0 && -add > current) {
            add = -static_cast<int>(current);
        }
        updateProgress(current + add);
    }

    /**
     * @brief Output end line.
     * @return
     */
    void finish() {
        std::cout << "\n";
    }
};
