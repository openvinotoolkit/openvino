// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/util/env_util.hpp"
#ifdef CPU_DEBUG_CAPS

#include <string>
#include <iostream>
#include <sstream>
#include <chrono>
#include <regex>

#include "onednn/dnnl.h"
#include "nodes/node_config.h"
#include <dnnl_debug.h>
#include "onednn/iml_type_mapper.h"
#include "openvino/core/model.hpp"
#include "edge.h"

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
    DebugLogEnabled(const char* file, const char* func, int line, const char* name = nullptr);

    const std::string & get_tag() const { return tag; }
    operator bool() const { return enabled; }
    void break_at(const std::string & log);
};

class NodeDesc;
class MemoryDesc;
class Node;
class Edge;
class Graph;
class IMemory;

class PrintableModel {
public:
    PrintableModel(const ov::Model& model, std::string tag = "", std::string prefix = "") : model(model), tag(tag), prefix(prefix) {}
    const ov::Model& model;
    const std::string tag;
    const std::string prefix;
};

template<typename T>
class PrintableVector {
public:
    PrintableVector(const std::vector<T>& values, int maxsize = 80) : values(values), maxsize(maxsize) {}
    const std::vector<T>& values;
    int maxsize;
};

template<typename T>
PrintableVector<T> printable(const std::vector<T>& values, int maxsize = 80) {
    return PrintableVector<T>(values, maxsize);
}
struct PrintableDelta {
    uint64_t us_last;
    uint64_t us_all;
};

class PrintableTimer {
public:
    PrintableTimer(): t0(std::chrono::high_resolution_clock::now()) {
        t1 = t0;
    }

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;

    PrintableDelta delta() {
        PrintableDelta ret;
        auto now = std::chrono::high_resolution_clock::now();
        ret.us_last = std::chrono::duration_cast<std::chrono::microseconds>(now - t1).count();
        ret.us_all = std::chrono::duration_cast<std::chrono::microseconds>(now - t0).count();
        t1 = now;
        return ret;
    }
};

std::ostream & operator<<(std::ostream & os, const PortConfig& desc);
std::ostream & operator<<(std::ostream & os, const NodeConfig& desc);
std::ostream & operator<<(std::ostream & os, const NodeDesc& desc);
std::ostream & operator<<(std::ostream & os, const Node& node);
std::ostream & operator<<(std::ostream & os, const ov::intel_cpu::Graph& graph);
std::ostream & operator<<(std::ostream & os, const Shape& shape);
std::ostream & operator<<(std::ostream & os, const MemoryDesc& desc);
std::ostream & operator<<(std::ostream & os, const IMemory& mem);
std::ostream & operator<<(std::ostream & os, const Edge& edge);
std::ostream & operator<<(std::ostream & os, const PrintableModel& model);
std::ostream & operator<<(std::ostream & os, const PrintableDelta& us);
std::ostream & operator<<(std::ostream & os, const Edge::ReorderStatus reorderStatus);

std::ostream & operator<<(std::ostream & os, const dnnl::primitive_desc& desc);
std::ostream & operator<<(std::ostream & os, const dnnl::memory::desc& desc);
std::ostream & operator<<(std::ostream & os, const impl_desc_type impl_type);
std::ostream & operator<<(std::ostream & os, const dnnl::memory::data_type dtype);
std::ostream & operator<<(std::ostream & os, const dnnl::memory::format_tag dtype);
std::ostream & operator<<(std::ostream & os, const dnnl::primitive_attr& attr);
std::ostream & operator<<(std::ostream & os, const dnnl::algorithm& alg);

void print_dnnl_memory(const dnnl::memory& memory, const size_t size, const int id, const char* message = "");

template<typename T>
std::ostream & operator<<(std::ostream & os, const PrintableVector<T>& vec) {
    std::stringstream ss;
    auto N = vec.values.size();
    for (size_t i = 0; i < N; i++) {
        if (i > 0)
            ss << ",";
        if (ss.tellp() > vec.maxsize) {
            ss << "..." << N << "in total";
            break;
        }
        if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value)
            ss << static_cast<int>(vec.values[i]);
        else
            ss << vec.values[i];
    }
    os << ss.str();
    return os;
}

static inline std::ostream& _write_all_to_stream(std::ostream& os) {
    return os;
}
template <typename T, typename... TS>
static inline std::ostream& _write_all_to_stream(std::ostream& os, const T& arg, TS&&... args) {
    return ov::intel_cpu::_write_all_to_stream(os << arg, std::forward<TS>(args)...);
}

}   // namespace intel_cpu
}   // namespace ov

#define DEBUG_ENABLE_NAME debug_enable_##__LINE__

#define DEBUG_LOG_EXT(name, ostream, prefix, ...)                        \
        do {                                                                                               \
            static DebugLogEnabled DEBUG_ENABLE_NAME(__FILE__, __func__, __LINE__, name);                  \
            if (DEBUG_ENABLE_NAME) {                                                                       \
                ::std::stringstream ss___;                                                                 \
                ov::intel_cpu::_write_all_to_stream(ss___, prefix, DEBUG_ENABLE_NAME.get_tag(), " ", __VA_ARGS__); \
                ostream << ss___.str() << std::endl;                                                     \
                DEBUG_ENABLE_NAME.break_at(ss___.str());                                                   \
            }                                                                                              \
        } while (0)

#define CPU_DEBUG_CAP_ENABLE(...) __VA_ARGS__
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) true

#define DEBUG_LOG(...) DEBUG_LOG_EXT(nullptr, std::cout, "[ DEBUG ] ", __VA_ARGS__)
#define ERROR_LOG(...) DEBUG_LOG_EXT(nullptr, std::cerr, "[ ERROR ] ", __VA_ARGS__)

#define CREATE_DEBUG_TIMER(x) PrintableTimer x

/*
 * important debugging tools for accuracy issues
 *   OV_CPU_INFER_PRC_POS_PATTERN : positive regex pattern to filter node type & orgname.
 *   OV_CPU_INFER_PRC_NEG_PATTERN : negative regex pattern to filter after pos-pattern was matched.
 *   OV_CPU_INFER_PRC_CNT   : number of nodes totally allowed to enforced
 * adjust these two settings until accuracy issue happens/disappears
 * from the log we can spot the node having issue when enabled bf16/f16
 */
struct EnforceInferPrcDebug {
    std::string safe_getenv(const char* name, const char* default_value = "") {
        std::string value = default_value;
        const char* p = std::getenv(name);
        if (p)
            value = p;
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

    EnforceInferPrcDebug() {
        str_pos_pattern = std::getenv("OV_CPU_INFER_PRC_POS_PATTERN");
        str_neg_pattern = std::getenv("OV_CPU_INFER_PRC_NEG_PATTERN");
        if (str_pos_pattern || str_neg_pattern) {
            pattern_verbose = true;
        } else {
            pattern_verbose = false;
        }
        if (str_pos_pattern)
            pos_pattern = std::regex(str_pos_pattern);
        if (str_neg_pattern)
            neg_pattern = std::regex(str_neg_pattern);
    }

    ~EnforceInferPrcDebug() {
        if (pattern_verbose) {
            if (str_pos_pattern)
                std::cout << "OV_CPU_INFER_PRC_POS_PATTERN=\"" << str_pos_pattern << "\"" << std::endl;
            if (str_neg_pattern)
                std::cout << "OV_CPU_INFER_PRC_NEG_PATTERN=\"" << str_neg_pattern << "\"" << std::endl;
            std::cout << "infer precision enforced Types: ";
            size_t total_cnt = 0;
            for (auto& ent : all_enabled_nodes) {
                std::cout << ent.first << ",";
                total_cnt += ent.second.size();
            }
            std::cout << "  total number of nodes: " << total_cnt << std::endl;
            for (auto& ent : all_enabled_nodes) {
                std::cout << ent.first << " : " << std::endl;
                for (auto& name : ent.second) {
                    std::cout << "\t" << name << std::endl;
                }
            }
            std::cout << std::endl;
        }
    }

    bool enabled(std::string type, std::string name, std::string org_names) {
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
#else // !CPU_DEBUG_CAPS

#define CPU_DEBUG_CAP_ENABLE(...)
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) x

#define DEBUG_LOG(...)
#define ERROR_LOG(...)
#define DEBUG_LOG_EXT(name, ...)

#define CREATE_DEBUG_TIMER(x)

#endif // CPU_DEBUG_CAPS

// To avoid "unused variable" warnings `when debug caps
// need more information than non-debug caps version
#define CPU_DEBUG_CAPS_MAYBE_UNUSED(x) (void)x
