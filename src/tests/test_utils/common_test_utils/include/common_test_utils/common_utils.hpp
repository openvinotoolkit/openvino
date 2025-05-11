// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace test {
namespace utils {

enum class OpType { SCALAR, VECTOR };

std::ostream& operator<<(std::ostream& os, OpType type);

template <typename vecElementType>
inline std::string vec2str(const std::vector<vecElementType>& vec) {
    if (!vec.empty()) {
        std::ostringstream result;
        result << "(";
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<vecElementType>(result, "."));
        result << vec.back() << ")";
        return result.str();
    }
    return std::string("()");
}

template <>
inline std::string vec2str(const std::vector<int64_t>& vec) {
    if (!vec.empty()) {
        std::ostringstream result;
        result << "(";
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<int64_t>(result, "."));
        result << vec.back() << ")";
        auto ret = result.str();
        std::replace(ret.begin(), ret.end(), '-', '_');
        return ret;
    }
    return std::string("()");
}

inline void replaceSubstringInString(std::string& str, const std::string& from, const std::string& to) {
    size_t pos;
    while ((pos = str.find(from)) != std::string::npos) {
        str.replace(pos, 1, to);
    }
}

inline std::string partialShape2str(const std::vector<ov::PartialShape>& partialShapes) {
    std::ostringstream result;
    for (const auto& partialShape : partialShapes) {
        result << partialShape;
    }
    auto retStr = result.str();
    std::replace(retStr.begin(), retStr.end(), ',', '.');
    return retStr;
}

inline std::string pair2str(const std::pair<size_t, size_t>& p) {
    std::ostringstream result;
    result << "(" << p.first << "." << p.second << ")";
    return result.str();
}

inline std::string vec2str(const std::vector<std::pair<size_t, size_t>>& vec) {
    std::ostringstream result;
    for (const auto& p : vec) {
        result << pair2str(p);
    }
    return result.str();
}

inline std::string vec2str(const std::vector<std::vector<std::pair<size_t, size_t>>>& vec) {
    std::ostringstream result;
    for (const auto& v : vec) {
        result << vec2str(v);
    }
    return result.str();
}

template <typename vecElementType>
inline std::string vec2str(const std::vector<std::vector<vecElementType>>& vec) {
    std::ostringstream result;
    for (const auto& v : vec) {
        result << vec2str<vecElementType>(v);
    }
    return result.str();
}

template <typename vecElementType>
inline std::string vec2str(const std::vector<std::vector<std::vector<vecElementType>>>& vec) {
    std::ostringstream result;
    for (const auto& v : vec) {
        result << vec2str<vecElementType>(v);
    }
    return result.str();
}

template <typename vecElementType>
inline std::string set2str(const std::set<vecElementType>& set) {
    if (!set.empty()) {
        std::ostringstream result;
        result << "(";
        std::copy(set.begin(), std::prev(set.end()), std::ostream_iterator<vecElementType>(result, "."));
        result << *set.rbegin() << ")";
        return result.str();
    }
    return std::string("()");
}

inline std::string bool2str(const bool val) {
    return val ? "True" : "False";
}

template <typename master, typename slave>
std::vector<std::pair<master, slave>> combineParams(const std::map<master, std::vector<slave>>& keyValueSets) {
    std::vector<std::pair<master, slave>> resVec;
    for (auto& keyValues : keyValueSets) {
        if (keyValues.second.empty()) {
            resVec.push_back({keyValues.first, {}});
        }
        for (auto& item : keyValues.second) {
            resVec.push_back({keyValues.first, item});
        }
    }
    return resVec;
}

inline bool endsWith(const std::string& source, const std::string& expectedSuffix) {
    return expectedSuffix.size() <= source.size() &&
           source.compare(source.size() - expectedSuffix.size(), expectedSuffix.size(), expectedSuffix) == 0;
}

template <std::size_t... I>
struct Indices {
    using next = Indices<I..., sizeof...(I)>;
};

template <std::size_t Size>
struct MakeIndices {
    using value = typename MakeIndices<Size - 1>::value::next;
};

template <>
struct MakeIndices<0> {
    using value = Indices<>;
};

template <class Tuple>
constexpr typename MakeIndices<std::tuple_size<typename std::decay<Tuple>::type>::value>::value makeIndices() {
    return {};
}

template <class Tuple, std::size_t... I>
std::vector<typename std::tuple_element<0, typename std::decay<Tuple>::type>::type> tuple2Vector(Tuple&& tuple,
                                                                                                 Indices<I...>) {
    using std::get;
    return {{get<I>(std::forward<Tuple>(tuple))...}};
}

template <class Tuple>
inline auto tuple2Vector(Tuple&& tuple) -> decltype(tuple2Vector(std::declval<Tuple>(), makeIndices<Tuple>())) {
    return tuple2Vector(std::forward<Tuple>(tuple), makeIndices<Tuple>());
}

template <class T>
inline T getTotal(const std::vector<T>& shape) {
    return shape.empty() ? 0 : std::accumulate(shape.cbegin(), shape.cend(), static_cast<T>(1), std::multiplies<T>());
}

inline std::string GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch);
    return std::to_string(ns.count());
}

inline std::ostream& operator<<(std::ostream& os, const std::map<std::string, std::string>& config) {
    os << "(";
    for (const auto& configItem : config) {
        os << configItem.first << "=" << configItem.second << "_";
    }
    os << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const ov::AnyMap& config) {
    os << "(";
    for (const auto& configItem : config) {
        os << configItem.first << "=" << configItem.second.as<std::string>() << "_";
    }
    os << ")";
    return os;
}

std::string generateTestFilePrefix();

size_t getVmSizeInKB();

size_t getVmRSSInKB();
}  // namespace utils
}  // namespace test
}  // namespace ov
