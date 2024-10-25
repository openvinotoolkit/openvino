// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "graph.hpp"
#include "openvino/openvino.hpp"
#include "utils/utils.hpp"

namespace ov {
namespace npuw {

struct Group;  // forward declaration

namespace online {

class Snapshot;  // forward declaration

// Partitioning operates with groups to prepare graph structure for the plugin
// Initally we assign each layer (excluding Parameters, Results, Constants and Converts)
// to it's own group. After some rounds of passes groups with several layers are formed and used in the plugin.
class Group : public std::enable_shared_from_this<Group> {
public:
    using GPtr = std::shared_ptr<Group>;

    Group() = delete;
    Group(const std::shared_ptr<ov::Node>& node,
          size_t gid,
          own::ade::NodeHandle nh,
          const std::shared_ptr<own::ade::Graph>& g,
          const std::weak_ptr<Snapshot>& snapshot);
    Group(size_t gid,
          own::ade::NodeHandle nh,
          const std::shared_ptr<own::ade::Graph>& g,
          const std::weak_ptr<Snapshot>& snapshot);

    // After we formed a final structure of partitioning,
    // we append excluded Convert layers to properly link submodels
    // Convert this representation to plugin-compatible one
    ov::npuw::Group toGroup() const;
    std::vector<own::ade::NodeHandle> srcNodes() const;
    std::vector<own::ade::NodeHandle> dstNodes() const;
    own::ade::NodeHandle getHandle() const;
    // Note: can only be used during initial group initialization
    std::shared_ptr<ov::Node> getInitialNode() const;
    void addInput(const std::shared_ptr<ov::Node>& node);
    void addOutput(const std::shared_ptr<ov::Node>& node);
    void addContent(const std::shared_ptr<ov::Node>& node);
    size_t getId() const;
    // This group consumes its producer
    void fuse(const Group::GPtr& gptr_prod);
    // This group consumes its consumer
    void fuseWith(const Group::GPtr& gptr_cons);
    // The inputs are merged together, this group is unchanged
    void fuseInputs(const std::pair<Group::GPtr, Group::GPtr>& gptr_inputs);
    // Check if there is indirect path from cons to this group
    bool hasCycle(const Group::GPtr& gptr_cons) const;
    size_t size() const;
    void freeze();
    void noFold();
    bool isFrozen() const;
    bool isNoFold() const;
    const detail::OVNodeSet& getContent() const;

    // Below is repeated blocks functionality
    const detail::Reptrack& getReptrack(const detail::OVNodePtr& node_ptr) const;
    void setRepeated(const std::shared_ptr<Repeated>& rep);
    std::shared_ptr<Repeated> repeated() const;
    std::unordered_set<MetaInterconnect> metaInterconnect(const Group::GPtr& gptr_prod) const;
    std::unordered_set<Interconnect> interconnect(const Group::GPtr& gptr_prod) const;
    // FIXME: unify avoid and isolate
    void avoid(const std::string& device);
    void isolate(const std::string& tag);
    void dontIsolate();
    const std::set<std::string>& avoidedTargets() const;
    const std::string& isolatedTag() const;
    std::string specialTags() const;
    void addWeightsPrecision(const std::vector<ov::element::Type>& prec);
    const std::vector<ov::element::Type>& getConstsPrecision() const;

private:
    void includeExtraLayers(detail::OVNodeSet& input_layers,
                            detail::OVNodeSet& output_layers,
                            detail::OVNodeSet& content) const;
    void updateInputLayers(const GPtr& gptr_other);
    void updateOutputLayers(const GPtr& gptr_other);
    // This group takes extra info of other group (such as reptrack, avoids, etc)
    void takeFlags(const Group::GPtr& gptr_other);
    void relinkGraph(const GPtr& gptr_other);

    detail::OVNodeSet m_input_layers;
    detail::OVNodeSet m_content;
    detail::OVNodeSet m_output_layers;

    own::ade::NodeHandle m_nh;
    size_t m_id;  // used for utility prints only
    std::shared_ptr<own::ade::Graph> m_graph;
    std::weak_ptr<Snapshot> m_snapshot;
    bool m_frozen = false;
    bool m_nofold = false;
    std::set<std::string> m_avoided_devices;
    std::string m_isol_tag = "";

    // Structure to keep track of mixed precision within initial model
    // Note: partitioning is stable so keep it in a single vector
    std::vector<ov::element::Type> m_consts_precision;

    // Unique repeated tag
    std::shared_ptr<Repeated> m_repeated = nullptr;
    // For each layer inside group, store it's history of repeated groups
    detail::ReptrackMap m_reptrack;
};

}  // namespace online
}  // namespace npuw
}  // namespace ov
