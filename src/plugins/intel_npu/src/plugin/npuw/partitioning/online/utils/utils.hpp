// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <unordered_map>

#include "attribute_visitor.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace npuw {
namespace online {

enum class PatternType { OP = 0, PATTERN = 1 };

struct Avoid {
    PatternType type;
    std::string pattern;
    std::string device;
};

struct Isolate {
    PatternType type;
    std::string pattern;
    std::string tag;
};

struct PassContext {
    size_t min_graph_size = 10;
    size_t keep_blocks = 10;
    size_t keep_block_size = 10;
    std::vector<Avoid> avoids;
    std::vector<Isolate> isolates;
    std::vector<std::string> nofolds;
};

// Forward declaration
class Group;
struct Repeated;
struct Interconnect;
struct MetaInterconnect;

namespace detail {
using OVNodePtr = std::shared_ptr<ov::Node>;
using OVNodeSet = std::unordered_set<OVNodePtr>;
using OVNodeSetPair = std::pair<OVNodeSet, OVNodeSet>;
using OVNodeMap = std::unordered_map<OVNodePtr, OVNodeSetPair>;
using OVNodeMapPtr = std::shared_ptr<OVNodeMap>;
using OVNodeToGroupMap = std::unordered_map<std::shared_ptr<ov::Node>, std::shared_ptr<Group>>;
using OVNodeToGroupMapPtr = std::shared_ptr<OVNodeToGroupMap>;
using GPtrSet = std::unordered_set<std::shared_ptr<Group>>;
using OVPortsMap = std::unordered_map<std::pair<OVNodePtr, OVNodePtr>, std::pair<size_t, size_t>>;
using Reptrack = std::vector<std::shared_ptr<Repeated>>;
using ReptrackMap = std::unordered_map<OVNodePtr, Reptrack>;
using Uniques = std::unordered_map<std::tuple<std::string, std::set<std::string>, std::string>, GPtrSet>;
using Pass = std::function<void(void)>;
}  // namespace detail

namespace util {
// FIXME: metadesc should be hash of layer's meta, not string
std::string getMetaDesc(const std::shared_ptr<ov::Node>& ov_node);
std::string repeated_id(const std::shared_ptr<Repeated>& ptr);
std::optional<Avoid> parseAvoid(const std::string& s);
std::optional<Isolate> parseIsolate(const std::string& s);
std::tuple<PatternType, std::string, std::string> parse(const std::string& s);
}  // namespace util

}  // namespace online
}  // namespace npuw
}  // namespace ov
