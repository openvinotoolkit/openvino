// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <oneapi/dnnl/dnnl.hpp>
#include <type_traits>
#include <vector>

#include "cpu_shape.h"
#ifdef CPU_DEBUG_CAPS

#    include <chrono>
#    include <iostream>
#    include <regex>
#    include <sstream>
#    include <string>
#    include <utility>

#    include "edge.h"
#    include "memory_control.hpp"
#    include "nodes/node_config.h"
#    include "onednn/iml_type_mapper.h"
#    include "openvino/core/model.hpp"
#    include "utils/general_utils.h"

namespace ov::intel_cpu {

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
    DebugLogEnabled(const char* file, const char* func, int line, const char* name = nullptr);

    [[nodiscard]] const std::string& get_tag() const {
        return tag;
    }
    operator bool() const {
        return enabled;
    }
    static void break_at(const std::string& log);
};

class NodeDesc;
class MemoryDesc;
class Node;
class Edge;
class Graph;
class IMemory;

class PrintableModel {
public:
    PrintableModel(const ov::Model& model, std::string tag = "", std::string prefix = "")
        : model(model),
          tag(std::move(tag)),
          prefix(std::move(prefix)) {}
    const ov::Model& model;
    const std::string tag;
    const std::string prefix;
};

template <typename T>
class PrintableVector {
public:
    PrintableVector(const std::vector<T>& values, int maxsize = 80) : values(values), maxsize(maxsize) {}
    const std::vector<T>& values;
    int maxsize;
};

template <typename T>
PrintableVector<T> printable(const std::vector<T>& values, int maxsize = 80) {
    return PrintableVector<T>(values, maxsize);
}
struct PrintableDelta {
    uint64_t us_last;
    uint64_t us_all;
};

class PrintableTimer {
public:
    PrintableTimer() : t0(std::chrono::high_resolution_clock::now()), t1(t0) {}

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;

    PrintableDelta delta() {
        auto now = std::chrono::high_resolution_clock::now();
        PrintableDelta ret{
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now - t1).count()),
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now - t0).count())};
        t1 = now;
        return ret;
    }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> vec) {
    for (const auto& element : vec) {
        os << element << "x";
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, const PortConfig& config);
std::ostream& operator<<(std::ostream& os, const NodeConfig& config);
std::ostream& operator<<(std::ostream& os, const NodeDesc& desc);
std::ostream& operator<<(std::ostream& os, const Node& node);
std::ostream& operator<<(std::ostream& os, const ov::intel_cpu::Graph& graph);
std::ostream& operator<<(std::ostream& os, const Shape& shape);
std::ostream& operator<<(std::ostream& os, const MemoryDesc& desc);
std::ostream& operator<<(std::ostream& os, const IMemory& mem);
std::ostream& operator<<(std::ostream& os, const PrintableModel& model);
std::ostream& operator<<(std::ostream& os, const PrintableDelta& d);
std::ostream& operator<<(std::ostream& os, Edge::ReorderStatus reorderStatus);
std::ostream& operator<<(std::ostream& os, const MemoryStatisticsRecord& record);

std::ostream& operator<<(std::ostream& os, const dnnl::primitive_desc& desc);
std::ostream& operator<<(std::ostream& os, const dnnl::memory::desc& desc);
std::ostream& operator<<(std::ostream& os, impl_desc_type impl_type);
std::ostream& operator<<(std::ostream& os, dnnl::memory::data_type dtype);
std::ostream& operator<<(std::ostream& os, dnnl::memory::format_tag format_tag);
std::ostream& operator<<(std::ostream& os, const dnnl::primitive_attr& attr);
std::ostream& operator<<(std::ostream& os, const dnnl::algorithm& alg);

template <typename T>
void print_dnnl_memory_as(const dnnl::memory& memory,
                          const size_t size,
                          const int id,
                          const std::string& message = {}) {
    const size_t s = memory.get_desc().get_size() / sizeof(T);
    std::cout << message << " ARG_ID: " << id << " size: " << s << ", values: ";
    auto m = static_cast<T*>(memory.get_data_handle());
    for (size_t i = 0; i < std::min(s, size); i++) {
        std::cout << std::to_string(*m) << " ";
        m++;
    }
    std::cout << "\n";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const PrintableVector<T>& vec) {
    std::stringstream ss;
    auto N = vec.values.size();
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            ss << ",";
        }
        if (ss.tellp() > vec.maxsize) {
            ss << "..." << N << "in total";
            break;
        }
        if (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            ss << static_cast<int>(vec.values[i]);
        } else {
            ss << vec.values[i];
        }
    }
    os << ss.str();
    return os;
}

template <typename... TS>
static inline std::ostream& _write_all_to_stream(std::ostream& os, TS&&... args) {
    return (os << ... << std::forward<TS>(args));
}

}  // namespace ov::intel_cpu

#    define DEBUG_ENABLE_NAME debug_enable_##__LINE__

#    define DEBUG_LOG_EXT(name, ostream, prefix, ...)                                                              \
        do {                                                                                                       \
            static DebugLogEnabled DEBUG_ENABLE_NAME(__FILE__, OV_CPU_FUNCTION_NAME, __LINE__, name);              \
            if (DEBUG_ENABLE_NAME || &ostream == &std::cerr) {                                                     \
                ::std::stringstream ss___;                                                                         \
                ov::intel_cpu::_write_all_to_stream(ss___, prefix, DEBUG_ENABLE_NAME.get_tag(), " ", __VA_ARGS__); \
                ostream << ss___.str() << '\n';                                                                    \
                DEBUG_ENABLE_NAME.break_at(ss___.str());                                                           \
            }                                                                                                      \
        } while (0)

#    define CPU_DEBUG_CAP_ENABLE(...) __VA_ARGS__

#    define DEBUG_LOG(...) DEBUG_LOG_EXT(nullptr, std::cout, "[ DEBUG ] ", __VA_ARGS__)
#    define ERROR_LOG(...) DEBUG_LOG_EXT(nullptr, std::cerr, "[ ERROR ] ", __VA_ARGS__)

#    define CREATE_DEBUG_TIMER(x) PrintableTimer x

#    define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) true

/*
 * important debugging tools for accuracy issues
 *   OV_CPU_INFER_PRC_POS_PATTERN : positive regex pattern to filter node type & orgname.
 *   OV_CPU_INFER_PRC_NEG_PATTERN : negative regex pattern to filter after pos-pattern was matched.
 *   OV_CPU_INFER_PRC_CNT   : number of nodes totally allowed to enforced
 * adjust these two settings until accuracy issue happens/disappears
 * from the log we can spot the node having issue when enabled bf16/f16
 */
struct EnforceInferPrcDebug {
    static std::string safe_getenv(const char* name, const char* default_value = "") {
        std::string value = default_value;
        const char* p = std::getenv(name);
        if (p) {
            value = p;
        }
        return value;
    }

    bool pattern_verbose;
    const char* str_pos_pattern;
    const char* str_neg_pattern;
    std::regex pos_pattern;
    std::regex neg_pattern;
    std::map<std::string, std::vector<std::string>> all_enabled_nodes;
    int count_limit = atoi(safe_getenv("OV_CPU_INFER_PRC_CNT", "9999999").c_str());
    int count = 0;

    EnforceInferPrcDebug()
        : str_pos_pattern(std::getenv("OV_CPU_INFER_PRC_POS_PATTERN")),
          str_neg_pattern(std::getenv("OV_CPU_INFER_PRC_NEG_PATTERN")) {
        pattern_verbose = (str_pos_pattern != nullptr) || (str_neg_pattern != nullptr);
        if (str_pos_pattern) {
            pos_pattern = std::regex(str_pos_pattern);
        }
        if (str_neg_pattern) {
            neg_pattern = std::regex(str_neg_pattern);
        }
    }

    ~EnforceInferPrcDebug() {
        if (pattern_verbose) {
            if (str_pos_pattern) {
                std::cout << "OV_CPU_INFER_PRC_POS_PATTERN=\"" << str_pos_pattern << "\"" << '\n';
            }
            if (str_neg_pattern) {
                std::cout << "OV_CPU_INFER_PRC_NEG_PATTERN=\"" << str_neg_pattern << "\"" << '\n';
            }
            std::cout << "infer precision enforced Types: ";
            size_t total_cnt = 0;
            for (auto& ent : all_enabled_nodes) {
                std::cout << ent.first << ",";
                total_cnt += ent.second.size();
            }
            std::cout << "  total number of nodes: " << total_cnt << '\n';
            for (auto& ent : all_enabled_nodes) {
                std::cout << ent.first << " : " << '\n';
                for (auto& name : ent.second) {
                    std::cout << "\t" << name << '\n';
                }
            }
            std::cout << '\n';
        }
    }

    bool enabled(const std::string& type, [[maybe_unused]] const std::string& name, const std::string& org_names) {
        std::string tag = type + "@" + org_names;
        std::smatch match;
        bool matched = true;
        // filter using pos pattern
        if (str_pos_pattern) {
            matched = std::regex_search(tag, match, pos_pattern);
        }
        // filter using neg pattern
        if (matched && str_neg_pattern) {
            matched = !std::regex_search(tag, match, neg_pattern);
        }

        // limit by CNT
        if (matched && count > count_limit) {
            matched = false;
        }

        if (matched) {
            auto it = all_enabled_nodes.find(type);
            if (it == all_enabled_nodes.end()) {
                all_enabled_nodes.insert({type, {tag}});
            } else {
                it->second.push_back(tag);
            }
            count++;
        }
        return matched;
    }
};

bool getEnvBool(const char* name);

#else  // !CPU_DEBUG_CAPS

#    define CPU_DEBUG_CAP_ENABLE(...)

#    define DEBUG_LOG(...)
#    define ERROR_LOG(...)
#    define DEBUG_LOG_EXT(name, ...)

#    define CREATE_DEBUG_TIMER(x)

#    define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) x

#endif  // CPU_DEBUG_CAPS
