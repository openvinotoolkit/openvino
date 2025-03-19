//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <numeric>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/gapi/own/assert.hpp>

namespace utils {

void createNDMat(cv::Mat& mat, const std::vector<int>& dims, int depth);
void generateRandom(cv::Mat& out);
cv::Mat createRandom(const std::vector<int>& dims, int depth);

template <typename duration_t>
typename duration_t::rep measure(std::function<void()> f) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    f();
    return duration_cast<duration_t>(high_resolution_clock::now() - start).count();
}

template <typename duration_t>
typename duration_t::rep timestamp() {
    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    return duration_cast<duration_t>(now.time_since_epoch()).count();
}

inline void busyWait(std::chrono::microseconds delay) {
    auto start_ts = timestamp<std::chrono::microseconds>();
    auto end_ts = start_ts;
    auto time_to_wait = delay.count();

    while (end_ts - start_ts < time_to_wait) {
        end_ts = timestamp<std::chrono::microseconds>();
    }
}

template <typename T>
double avg(const std::vector<T>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}

template <typename T>
T max(const std::vector<T>& vec) {
    return *std::max_element(vec.begin(), vec.end());
}

template <typename T>
T min(const std::vector<T>& vec) {
    return *std::min_element(vec.begin(), vec.end());
}

void readFromBinFile(const std::string& filepath, cv::Mat& mat);
void writeToBinFile(const std::string& filepath, const cv::Mat& mat);

}  // namespace utils
