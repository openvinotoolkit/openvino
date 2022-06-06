// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#define CPU_DEBUG_CAP_ENABLE(_x) _x;
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) true

#include <string>
#include <iostream>

#include <onednn/dnnl.h>
#include <dnnl_debug.h>

namespace ov {
namespace intel_cpu {


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
    std::string tag;

public:
    DebugLogEnabled(const char* file, const char* func, int line);

    const std::string & get_tag() const { return tag; }
    operator bool() const { return enabled; }
    void break_at(const std::string & log);
};

static inline std::ostream& write_all_to_stream(std::ostream& os) {
    return os;
}
template <typename T, typename... TS>
static inline std::ostream& write_all_to_stream(std::ostream& os, const T& arg, TS&&... args) {
    return write_all_to_stream(os << arg, std::forward<TS>(args)...);
}

class NodeDesc;
class MemoryDesc;
class Node;
class Edge;

std::ostream & operator<<(std::ostream & os, const dnnl::memory::desc& desc);
std::ostream & operator<<(std::ostream & os, const NodeDesc& desc);
std::ostream & operator<<(std::ostream & os, const Node& node);
std::ostream & operator<<(std::ostream & os, const MemoryDesc& desc);
std::ostream & operator<<(std::ostream & os, const Edge& edge);

}   // namespace intel_cpu
}   // namespace ov

#define DEBUG_ENABLE_NAME debug_enable_##__LINE__

#define DEBUG_LOG(...)                                                                                     \
        do {                                                                                               \
            static DebugLogEnabled DEBUG_ENABLE_NAME(__FILE__, __func__, __LINE__);                        \
            if (DEBUG_ENABLE_NAME) {                                                                       \
                ::std::stringstream ss___;                                                                 \
                ov::intel_cpu::write_all_to_stream(ss___, "[ DEBUG ] ", DEBUG_ENABLE_NAME.get_tag(), " ", __VA_ARGS__); \
                std::cout << ss___.str() << std::endl;                                                     \
                DEBUG_ENABLE_NAME.break_at(ss___.str());                                                   \
            }                                                                                              \
        } while (0)

#else // !CPU_DEBUG_CAPS

#define CPU_DEBUG_CAP_ENABLE(_x)
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) x

#define DEBUG_LOG(...)

#endif // CPU_DEBUG_CAPS
