// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/openvino.hpp"
#include "utils/utils.hpp"

namespace ov {
namespace npuw {
namespace online {

// Mainly used as a shared_ptr to tag a particular repeated block
struct Repeated {
    struct Archetype {
        std::string metadesc;
        ov::npuw::online::detail::Reptrack reptrack;
        bool operator==(const Archetype& other) const;
    };

    void exclude();
    void resetExclude();
    bool openForMerge() const;

    bool m_excluded = false;
};

struct Interconnect {
    detail::OVNodePtr input_node;
    size_t input_port;

    detail::OVNodePtr output_node;
    size_t output_port;

    bool operator==(const Interconnect& other) const;
};

struct MetaInterconnect {
    std::string input_meta;
    detail::Reptrack input_reptrack;
    size_t input_port;

    std::string output_meta;
    detail::Reptrack output_reptrack;
    size_t output_port;

    bool operator==(const MetaInterconnect& other) const;
    bool operator<(const MetaInterconnect& other) const;
};

}  // namespace online
}  // namespace npuw
}  // namespace ov

namespace std {
template <>
struct hash<std::pair<ov::npuw::online::detail::OVNodePtr, ov::npuw::online::detail::OVNodePtr>> {
    inline size_t operator()(
        const std::pair<ov::npuw::online::detail::OVNodePtr, ov::npuw::online::detail::OVNodePtr>& p) const {
        return (std::hash<ov::npuw::online::detail::OVNodePtr>()(p.first) + 0x9e3779b9) ^
               (std::hash<ov::npuw::online::detail::OVNodePtr>()(p.second) + 0x9e3779b9);
    }
};

template <>
struct hash<std::vector<ov::element::Type>> {
    inline size_t operator()(const std::vector<ov::element::Type>& vec) const {
        std::size_t seed = vec.size();
        for (const auto& s : vec) {
            seed ^= s.hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

template <>
struct hash<std::tuple<std::string, std::set<std::string>, std::string>> {
    inline size_t operator()(const std::tuple<std::string, std::set<std::string>, std::string>& t) const {
        std::size_t seed = std::hash<std::string>()(std::get<0>(t)) + 0x9e3779b9;
        seed ^= std::hash<std::string>()(std::get<2>(t)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        for (const auto& s : std::get<1>(t)) {
            seed ^= std::hash<std::string>()(s) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

template <>
struct hash<ov::npuw::online::detail::Reptrack> {
    inline size_t operator()(const ov::npuw::online::detail::Reptrack& vec) const {
        std::size_t seed = vec.size();
        for (const auto& rep : vec) {
            seed ^=
                std::hash<std::shared_ptr<ov::npuw::online::Repeated>>()(rep) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

template <>
struct hash<ov::npuw::online::Repeated::Archetype> {
    inline size_t operator()(const ov::npuw::online::Repeated::Archetype& act) const {
        return (std::hash<std::string>()(act.metadesc) + 0x9e3779b9) ^
               (std::hash<ov::npuw::online::detail::Reptrack>()(act.reptrack) + 0x9e3779b9);
    }
};

template <>
struct hash<ov::npuw::online::Interconnect> {
    inline size_t operator()(const ov::npuw::online::Interconnect& mic) const {
        return (std::hash<ov::npuw::online::detail::OVNodePtr>()(mic.input_node) + 0x9e3779b9) ^
               (std::hash<ov::npuw::online::detail::OVNodePtr>()(mic.output_node) + 0x9e3779b9) ^
               (std::hash<size_t>()(mic.input_port) + 0x9e3779b9) ^ (std::hash<size_t>()(mic.output_port) + 0x9e3779b9);
    }
};

template <>
struct hash<ov::npuw::online::MetaInterconnect> {
    inline size_t operator()(const ov::npuw::online::MetaInterconnect& mic) const {
        return (std::hash<std::string>()(mic.input_meta) + 0x9e3779b9) ^
               (std::hash<std::string>()(mic.output_meta) + 0x9e3779b9) ^
               (std::hash<size_t>()(mic.input_port) + 0x9e3779b9) ^
               (std::hash<size_t>()(mic.output_port) + 0x9e3779b9) ^
               (std::hash<ov::npuw::online::detail::Reptrack>()(mic.input_reptrack) + 0x9e3779b9) ^
               (std::hash<ov::npuw::online::detail::Reptrack>()(mic.output_reptrack) + 0x9e3779b9);
    }
};

// FIXME: hash<MetaInterconnect> defined above. This hash should be available by default
template <>
struct hash<std::vector<ov::npuw::online::MetaInterconnect>> {
    inline size_t operator()(const std::vector<ov::npuw::online::MetaInterconnect>& vec) const {
        std::size_t seed = vec.size();
        for (const auto& mic : vec) {
            seed ^= std::hash<ov::npuw::online::MetaInterconnect>()(mic) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
}  // namespace std
