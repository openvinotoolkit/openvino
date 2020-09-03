// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <chrono>
#include <fstream>
#include <memory>

using time_point = std::chrono::high_resolution_clock::time_point;

/**
* @brief Class response for encapsulating time measurements.
*
* Object of a class measures time at start and finish of object's life cycle.
* When deleting, reports duration.
*/
class Timer {
private:
    std::string name;
    time_point start_time;

public:
    /**
     * @brief Constructs Timer object and measures start time
     */
    Timer(const std::string &timer_name) {
        name = timer_name;
        start_time = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Destructs Timer object, measures duration and reports it
     */
    ~Timer(){
        float duration = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
        std::cout << name << ":" << duration << "\n"; // TODO: replace with writer
    }
};

#define SCOPED_TIMER(timer_name) Timer timer_name(#timer_name);
