// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#define CPU_DEBUG_CAP_ENABLE(_x) _x;
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) true

// OV_CPU_DEBUG_LOG controls DEBUG_LOGs to output
//
// positive filter: enables patterns in filter
//   [+]foo;bar:line2;  enables  "foo:*" and "bar:line2"
//   -                  enables all debug log
//
// negative filter: disable patterns in filter
//   -f1;f2:l;          disables  "foo:*" and "bar:line2"
//
class DebugLogEnabled {
    bool enabled;

public:
    DebugLogEnabled(const char* func, int line) {
        // check ENV
        const char* p_filters = std::getenv("OV_CPU_DEBUG_LOG");
        if (!p_filters) {
            enabled = false;
            return;
        }

        // check each filter patten:
        bool filter_match_action;
        if (p_filters[0] == '-') {
            p_filters++;
            filter_match_action = false;
        } else {
            filter_match_action = true;
        }

        std::string func_with_line(func);
        func_with_line += ":" + std::to_string(line);

        bool match = false;
        const char* p0 = p_filters;
        const char* p1;
        while (*p0 != 0) {
            p1 = p0;
            while (*p1 != ';' && *p1 != 0)
                ++p1;
            std::string patten(p0, p1 - p0);
            if (patten == func || patten == func_with_line) {
                match = true;
                break;
            }
            p0 = p1;
            if (*p0 == ';')
                ++p0;
        }

        if (match)
            enabled = filter_match_action;
        else
            enabled = !filter_match_action;
    }
    operator bool() const {
        return enabled;
    }
};

#define DEBUG_ENABLE_NAME debug_enable_##__LINE__

#define DEBUG_LOG(...)                                                                                     \
        do {                                                                                               \
            static DebugLogEnabled DEBUG_ENABLE_NAME(__func__, __LINE__);                                  \
            if (DEBUG_ENABLE_NAME) {                                                                       \
                ::std::stringstream ss___;                                                                 \
                ::ov::write_all_to_stream(ss___, "[ DEBUG ] ", __func__, ":", __LINE__, " ", __VA_ARGS__); \
                std::cout << ss___.str() << std::endl;                                                     \
            }                                                                                              \
        } while (0)

#else // !CPU_DEBUG_CAPS

#define CPU_DEBUG_CAP_ENABLE(_x)
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) x

#define DEBUG_LOG(...)

#endif // CPU_DEBUG_CAPS
