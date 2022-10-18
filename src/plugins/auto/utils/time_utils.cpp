// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>
#include "time_utils.hpp"

namespace MultiDevicePlugin {
namespace TimeUtils {

bool localtimeSafe(const time_t* time, struct tm* result) {
    if (time && result) {
#if (defined(_WIN32) || defined(_WIN64))
        localtime_s(result, time);
#else
        localtime_r(time, result);
#endif
        return true;
    }
    return false;
}

std::string putTime(std::chrono::system_clock::time_point tp, const char* format) {
    struct tm t = {};
    time_t timeObj = std::chrono::system_clock::to_time_t(tp);

    localtimeSafe(&timeObj, &t);

    std::stringstream ss;

#if (defined(__GNUC__) && (__GNUC__ < 5))
    char time_str[24];
    strftime(time_str, sizeof(time_str), format, &t);
    ss << time_str;
#else
    ss << std::put_time(&t, format);
#endif

    return ss.str();
}

std::string getCurrentTime() {
    std::stringstream ss;

#ifdef VERBOSE_LOG
    const char* timeFormat = "%F %T";
#else
    const char* timeFormat = "%T";
#endif

    auto now = std::chrono::system_clock::now();

    auto microseconds = (std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000) / 100;

    ss << putTime(now, timeFormat) << '.' << std::setfill('0') << std::setw(4) << microseconds;

    return ss.str();
}
} // namespace TimeUtils
} // namespace MultiDevicePlugin
