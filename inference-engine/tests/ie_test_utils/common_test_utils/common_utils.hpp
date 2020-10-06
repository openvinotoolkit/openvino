// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <string>
#include <sstream>
#include <iterator>
#include <vector>
#include <set>

#include <cpp/ie_cnn_network.h>
#include <legacy/details/ie_cnn_network_iterator.hpp>

namespace CommonTestUtils {
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

inline InferenceEngine::CNNLayerPtr getLayerByName(const InferenceEngine::ICNNNetwork * icnnnetwork,
                                                   const std::string & layerName) {
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::details::CNNNetworkIterator i(icnnnetwork), end;
    while (i != end) {
        auto layer = *i;
        if (layer->name == layerName)
            return layer;
        ++i;
    }

    std::stringstream stream;
    stream << "Layer " << layerName << " not found in network";
    throw InferenceEngine::NotFound(stream.str());
    IE_SUPPRESS_DEPRECATED_END
}

inline InferenceEngine::CNNLayerPtr getLayerByName(const InferenceEngine::CNNNetwork & network,
                                                   const std::string & layerName) {
    const InferenceEngine::ICNNNetwork & icnnnetwork = static_cast<const InferenceEngine::ICNNNetwork&>(network);
    return getLayerByName(&icnnnetwork, layerName);
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

}  // namespace CommonTestUtils
