// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <string>
#include <sstream>
#include <iterator>
#include <vector>
#include <set>
#include <chrono>
#include <ostream>
#include <memory>

#include <cpp/ie_cnn_network.h>
#include <ngraph/function.hpp>

namespace ngraph {
::std::ostream& operator << (::std::ostream &, const Function&);
}

namespace InferenceEngine {
class CNNLayer;
}

namespace CommonTestUtils {

enum class OpType {
    SCALAR,
    VECTOR
};

std::ostream& operator<<(std::ostream & os, OpType type);

IE_SUPPRESS_DEPRECATED_START
std::shared_ptr<InferenceEngine::CNNLayer>
getLayerByName(const InferenceEngine::CNNNetwork & network, const std::string & layerName);
IE_SUPPRESS_DEPRECATED_END

template<typename vecElementType>
inline std::string vec2str(const std::vector<vecElementType> &vec) {
    if (!vec.empty()) {
        std::ostringstream result;
        result << "(";
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<vecElementType>(result, "."));
        result << vec.back() << ")";
        return result.str();
    }
    return std::string("()");
}

template<typename vecElementType>
inline std::string vec2str(const std::vector<std::vector<vecElementType>> &vec) {
    std::ostringstream result;
    for (const auto &v : vec) {
        result << vec2str<vecElementType>(v);
    }
    return result.str();
}

template<typename vecElementType>
inline std::string set2str(const std::set<vecElementType> &set) {
    if (!set.empty()) {
        std::ostringstream result;
        result << "(";
        std::copy(set.begin(), std::prev(set.end()), std::ostream_iterator<vecElementType>(result, "."));
        result << *set.rbegin() << ")";
        return result.str();
    }
    return std::string("()");
}

template <typename master, typename slave>
std::vector<std::pair<master, slave>> combineParams(
    const std::map<master, std::vector<slave>>& keyValueSets) {
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
    return expectedSuffix.size() <= source.size() && source.compare(source.size() - expectedSuffix.size(), expectedSuffix.size(), expectedSuffix) == 0;
}

template<std::size_t... I>
struct Indices {
    using next = Indices<I..., sizeof...(I)>;
};

template<std::size_t Size>
struct MakeIndices {
    using value = typename MakeIndices<Size - 1>::value::next;
};

template<>
struct MakeIndices<0> {
    using value = Indices<>;
};

template<class Tuple>
constexpr typename MakeIndices<std::tuple_size<typename std::decay<Tuple>::type>::value>::value makeIndices() {
    return {};
}

template<class Tuple, std::size_t... I>
std::vector<typename std::tuple_element<0, typename std::decay<Tuple>::type>::type> tuple2Vector(Tuple&& tuple, Indices<I...>) {
    using std::get;
    return {{ get<I>(std::forward<Tuple>(tuple))... }};
}

template<class Tuple>
inline auto tuple2Vector(Tuple&& tuple) -> decltype(tuple2Vector(std::declval<Tuple>(), makeIndices<Tuple>())) {
    return tuple2Vector(std::forward<Tuple>(tuple), makeIndices<Tuple>());
}

template<class T>
inline T getTotal(const std::vector<T>& shape) {
    return shape.empty() ? 0 : std::accumulate(shape.cbegin(), shape.cend(), static_cast<T>(1), std::multiplies<T>());
}

inline std::string GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch);
    return std::to_string(ns.count());
}

}  // namespace CommonTestUtils
