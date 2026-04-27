// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group.hpp"

#include <sstream>

#include "../../logging.hpp"
#include "../partitioning.hpp"  // ov::npuw::Group
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/util/common_util.hpp"
#include "repeated.hpp"
#include "snapshot.hpp"

using ov::npuw::online::Group;
using ov::npuw::online::Interconnect;
using ov::npuw::online::MetaInterconnect;
using ov::npuw::online::MetaInterconnectIO;
using ov::npuw::online::Repeated;
using ov::npuw::online::detail::isOp;
using ov::npuw::online::detail::MICSet;
using ov::npuw::online::detail::PairMICSetIO;

Group::Group(const std::shared_ptr<ov::Node>& node,
             size_t gid,
             own::ade::NodeHandle nh,
             const std::weak_ptr<own::ade::Graph>& g,
             const std::weak_ptr<Snapshot>& snapshot)
    : m_nh(std::move(nh)),
      m_id(gid),
      m_graph(g),
      m_snapshot(snapshot) {
    m_input_layers.insert(node);
    m_output_layers.insert(node);
    m_content.insert(node);
}

Group::Group(size_t gid,
             own::ade::NodeHandle nh,
             const std::weak_ptr<own::ade::Graph>& g,
             const std::weak_ptr<Snapshot>& snapshot)
    : m_nh(std::move(nh)),
      m_id(gid),
      m_graph(g),
      m_snapshot(snapshot) {}

// Include Parameters, Outputs, Converts, etc to content's layers for proper linking at the plugin level
void Group::includeExtraLayers(detail::OVNodeSet& input_layers,
                               detail::OVNodeSet& output_layers,
                               detail::OVNodeSet& content) const {
    detail::OVNodeSet extra_content;
    // Including writers of parameter layers and results' reader layers
    for (const auto& layer : content) {
        for (size_t i = 0; i < layer->inputs().size(); ++i) {
            auto target_input = layer->get_input_source_output(i);
            auto layer_parent = target_input.get_node()->shared_from_this();
            if (ov::op::util::is_parameter(layer_parent)) {
                input_layers.insert(layer);
            }
            // Also include Converts
            if (!isOp(layer_parent)) {
                if (!ov::op::util::is_constant(layer_parent) && !ov::op::util::is_parameter(layer_parent) &&
                    !ov::op::util::is_output(layer_parent)) {
                    NPUW_ASSERT(ov::is_type<ov::op::v0::Convert>(layer_parent));
                    extra_content.insert(layer_parent);
                }
            }
        }
        for (size_t i = 0; i < layer->outputs().size(); ++i) {
            const auto target_outputs = layer->get_output_target_inputs(i);
            for (const auto& target_output : target_outputs) {
                auto layer_child = target_output.get_node()->shared_from_this();
                if (ov::op::util::is_output(layer_child)) {
                    output_layers.insert(layer);
                }
            }
        }
    }

    for (const auto& layer : extra_content) {
        content.insert(layer);
    }
}

// Convert this representation to plugin-compatible one
ov::npuw::Group Group::toGroup() const {
    auto input_copy = m_input_layers;
    auto output_copy = m_output_layers;
    auto content_copy = m_content;
    includeExtraLayers(input_copy, output_copy, content_copy);

    ov::npuw::Group g;
    for (auto&& node : input_copy) {
        g.input_layers.push_back(node->get_friendly_name());
    }
    for (auto&& node : output_copy) {
        g.output_layers.push_back(node->get_friendly_name());
    }
    for (auto&& node : content_copy) {
        g.all_layers.push_back(node->get_friendly_name());
    }

    // Sort layers to stabilize the partitioning
    std::sort(g.input_layers.begin(), g.input_layers.end());
    std::sort(g.output_layers.begin(), g.output_layers.end());
    std::sort(g.all_layers.begin(), g.all_layers.end());

    g.gflops = 0.0001f;  // FIXME: calculate proper flops

    if (m_repeated && !isNoFold()) {
        g.repeated_id = m_repeated->id();
    }

    if (!m_avoided_devices.empty()) {
        auto iter = m_avoided_devices.begin();
        g.avoid_list += *iter;
        while (++iter != m_avoided_devices.end()) {
            g.avoid_list += ',' + *iter;
        }
    }

    g.settag(m_isol_tag);

    return g;
}

// Note: can only be used after initial group initialization
std::shared_ptr<ov::Node> Group::getInitialNode() const {
    if (m_content.size() != 1) {
        OPENVINO_THROW("Online partitioning initial group ", m_id, " doesn't consist of exactly 1 layer!");
    }

    return *(m_content.begin());
}

const std::unordered_set<std::shared_ptr<ov::Node>>& Group::getInputs() const {
    return m_input_layers;
}

const std::unordered_set<std::shared_ptr<ov::Node>>& Group::getOutputs() const {
    return m_output_layers;
}

void Group::addInput(const std::shared_ptr<ov::Node>& node) {
    m_input_layers.insert(node);
    m_mic_io_valid = false;
}

void Group::addOutput(const std::shared_ptr<ov::Node>& node) {
    m_output_layers.insert(node);
    m_mic_io_valid = false;
}

void Group::addContent(const std::shared_ptr<ov::Node>& node) {
    m_content.insert(node);
}

size_t Group::getId() const {
    return m_id;
}

std::vector<own::ade::NodeHandle> Group::srcNodes() const {
    return m_nh->srcNodes();
}

std::vector<own::ade::NodeHandle> Group::dstNodes() const {
    return m_nh->dstNodes();
}

own::ade::NodeHandle Group::getHandle() const {
    return m_nh;
}

// Not every input should be included - those layers which contained inside merging group are not inputs anymore
void Group::updateInputLayers(const Group::GPtr& gptr_other) {
    m_mic_io_valid = false;
    detail::OVNodeSet combined_input;
    combined_input.insert(m_input_layers.begin(), m_input_layers.end());
    combined_input.insert(gptr_other->m_input_layers.begin(), gptr_other->m_input_layers.end());
    m_input_layers.clear();

    auto locked_snapshot = m_snapshot.lock();

    for (const auto& layer : combined_input) {
        detail::OVNodeSet selected;
        auto node_prod = locked_snapshot->getNodeProducers(layer);
        for (const auto& prod : node_prod) {
            if (m_content.find(prod) == m_content.end()) {
                selected.insert(prod);
            }
        }
        for (const auto& l : selected) {
            if (isOp(l)) {
                m_input_layers.insert(layer);
            }
        }
    }
}

// Not every output should be included - those layers which are consumed _ONLY_ by the merged group are not outputs
// anymore
void Group::updateOutputLayers(const Group::GPtr& gptr_other) {
    m_mic_io_valid = false;
    detail::OVNodeSet combined_output;
    combined_output.insert(m_output_layers.begin(), m_output_layers.end());
    combined_output.insert(gptr_other->m_output_layers.begin(), gptr_other->m_output_layers.end());
    m_output_layers.clear();

    auto locked_snapshot = m_snapshot.lock();

    for (const auto& layer : combined_output) {
        auto node_cons = locked_snapshot->getNodeConsumers(layer);
        bool reject = true;
        for (const auto& cons : node_cons) {
            if (m_content.find(cons) == m_content.end()) {
                reject = false;
            }
        }
        if (!reject) {
            m_output_layers.insert(layer);
        }
    }
}

void Group::relinkGraph(const Group::GPtr& gptr_other) {
    auto producers = gptr_other->srcNodes();
    auto consumers = gptr_other->dstNodes();

    // Remove gptr_other node from the graph. Note: also removes all it's edges
    auto&& graph = m_graph.lock();
    NPUW_ASSERT(graph);
    graph->remove(gptr_other->getHandle());
    for (const auto& nh : producers) {
        if (m_nh == nh) {
            continue;
        }
        // relink the graph
        if (!graph->linked(nh, m_nh)) {
            graph->link(nh, m_nh);
        }
    }
    for (const auto& nh : consumers) {
        if (m_nh == nh) {
            continue;
        }
        // relink the graph
        if (!graph->linked(m_nh, nh)) {
            graph->link(m_nh, nh);
        }
    }
}

// This group absorbs the producer
void Group::fuse(const Group::GPtr& gptr_prod) {
    fuseWith(gptr_prod);
}

// This group absorbs the consumer
void Group::fuseWith(const Group::GPtr& gptr_cons) {
    if (ov::npuw::debug_groups()) {
        LOG_DEBUG("Fusing...");
        LOG_BLOCK();
        {
            LOG_DEBUG("Merger: " << this->specialTags());
            dump();
        }
        {
            LOG_DEBUG("Mergee: " << gptr_cons->specialTags());
            gptr_cons->dump();
        }
    }

    auto locked_snapshot = m_snapshot.lock();
    auto node_to_gr = locked_snapshot->getNodeToGroupMap();
    for (const auto& layer : gptr_cons->m_content) {
        node_to_gr->at(layer) = shared_from_this();  // layers of consumer group are assigned to this group
    }

    // Merge 2 contents together
    for (const auto& layer : gptr_cons->m_content) {
        m_content.insert(layer);
    }

    takeFlags(gptr_cons);

    updateInputLayers(gptr_cons);
    updateOutputLayers(gptr_cons);
    relinkGraph(gptr_cons);
}

// This group is unchanged, the inputs are merged together into the first one
void Group::fuseInputs(const std::pair<Group::GPtr, Group::GPtr>& gptr_inputs) {
    Group::GPtr absorbing_group = gptr_inputs.first;
    Group::GPtr absorbed_group = gptr_inputs.second;

    auto locked_snapshot = m_snapshot.lock();
    auto node_to_gr = locked_snapshot->getNodeToGroupMap();

    // Update ov::node to own::ade::NodeHandle map and merge all contents together
    for (const auto& layer : absorbed_group->m_content) {
        node_to_gr->at(layer) = absorbing_group;
        absorbing_group->m_content.insert(layer);
    }
    absorbing_group->takeFlags(absorbed_group);
    absorbing_group->updateInputLayers(absorbed_group);
    absorbing_group->updateOutputLayers(absorbed_group);
    absorbing_group->relinkGraph(absorbed_group);
}

// This group takes extra info of other group (such as reptrack, avoids, etc)
// FIXME: unify managing all those tags, e.g. via a map string->string
void Group::takeFlags(const Group::GPtr& gptr_other) {
    // Update reptrack
    for (const auto& layer_to_track : gptr_other->m_reptrack) {
        auto layer = layer_to_track.first;
        auto track = layer_to_track.second;

        for (const auto& rep : track) {
            m_reptrack[layer].push_back(rep);
        }
    }
    // Update weights precisions
    for (const auto& wp : gptr_other->m_consts_precision) {
        m_consts_precision.push_back(wp);
    }
    // Update avoids
    for (const auto& device : gptr_other->avoidedTargets()) {
        avoid(device);
    }
    // Update nofold
    m_nofold = gptr_other->isNoFold();
    // Update isolate tag
    m_isol_tag = gptr_other->isolatedTag();
}

// Check if there is indirect path from this to gptr_cons
bool Group::hasCycle(const Group::GPtr& gptr_cons) const {
    // Fast-path: if this group has at most 1 consumer, there is no alternative path
    // to gptr_cons and therefore no cycle.
    if (m_nh->dstNodes().size() <= 1) {
        return false;
    }

    std::unordered_set<own::ade::NodeHandle> visited;
    std::stack<own::ade::NodeHandle> st;

    for (const auto& prod : gptr_cons->srcNodes()) {
        // skip self during this iter
        if (!(m_nh == prod)) {
            st.push(prod);
            visited.insert(prod);  // mark at push time to avoid duplicate pushes
        }
    }

    while (!st.empty()) {
        auto nh = st.top();
        st.pop();

        if (nh == m_nh) {
            // Found another path from self to gptr_cons
            return true;
        }

        for (const auto& prod : nh->srcNodes()) {
            if (!visited.count(prod)) {
                st.push(prod);
                visited.insert(prod);  // mark at push time
            }
        }
    }

    return false;
}

size_t Group::size() const {
    return m_content.size();
}

void Group::freeze() {
    m_frozen = true;
}

void Group::noFold() {
    m_nofold = true;
}

bool Group::isFrozen() const {
    return m_frozen;
}

bool Group::isNoFold() const {
    return m_nofold;
}

const ov::npuw::online::detail::OVNodeSet& Group::getContent() const {
    return m_content;
}

const ov::npuw::online::detail::Reptrack& Group::getReptrack(
    const ov::npuw::online::detail::OVNodePtr& node_ptr) const {
    if (m_reptrack.find(node_ptr) == m_reptrack.end()) {
        OPENVINO_THROW("Online partitioning repeated track doesn't contain ", node_ptr->get_friendly_name());
    }
    return m_reptrack.at(node_ptr);
}

std::shared_ptr<Repeated> Group::repeated() const {
    return m_repeated;
}

void Group::setRepeated(const std::shared_ptr<Repeated>& rep) {
    m_repeated = rep;

    if (!rep) {
        return;
    }

    for (const auto& layer : m_content) {
        m_reptrack[layer].push_back(m_repeated);
    }
}

PairMICSetIO Group::metaInterconnect(const Group::GPtr& gptr_prod) const {
    MICSet mics;

    auto locked_snapshot = m_snapshot.lock();
    auto ics = interconnect(gptr_prod);
    for (const auto& ic : ics) {
        const auto prod_track_it = gptr_prod->m_reptrack.find(ic.input_node);
        const auto cons_track_it = m_reptrack.find(ic.output_node);
        if (prod_track_it == gptr_prod->m_reptrack.end() || cons_track_it == m_reptrack.end()) {
            LOG_WARN("Online partitioning: missing reptrack while building metaInterconnect (prod="
                     << gptr_prod->getId() << ", cons=" << getId() << ", input_node="
                     << ic.input_node->get_friendly_name() << ", output_node=" << ic.output_node->get_friendly_name()
                     << "). This candidate will be skipped.");
            return {};
        }
        mics.insert({locked_snapshot->getMetaDesc(ic.input_node),
                     prod_track_it->second,
                     ic.input_port,
                     locked_snapshot->getMetaDesc(ic.output_node),
                     cons_track_it->second,
                     ic.output_port});
    }

    // Cache the MetaInterconnectIO part — it only depends on m_input_layers /
    // m_output_layers which are invalidated (m_mic_io_valid = false) whenever
    // those sets change (addInput / addOutput / updateInputLayers / updateOutputLayers).
    if (!m_mic_io_valid) {
        m_cached_mic_io = MetaInterconnectIO{};
        for (const auto& oi : m_input_layers) {
            m_cached_mic_io.output_imeta.insert(locked_snapshot->getMetaDesc(oi));
        }
        for (const auto& oo : m_output_layers) {
            m_cached_mic_io.output_ometa.insert(locked_snapshot->getMetaDesc(oo));
        }
        m_mic_io_valid = true;
    }

    return {mics, m_cached_mic_io};
}

std::unordered_set<Interconnect> Group::interconnect(const Group::GPtr& gptr_prod) const {
    std::unordered_set<Interconnect> ics;

    auto locked_snapshot = m_snapshot.lock();
    const auto& ports_map = locked_snapshot->getPortsMap();

    for (const auto& layer : m_content) {
        for (const auto& input_layer : locked_snapshot->getNodeProducers(layer)) {
            if (gptr_prod->m_content.find(input_layer) != gptr_prod->m_content.end()) {
                const auto ports_it = ports_map.find({input_layer, layer});
                if (ports_it == ports_map.end()) {
                    LOG_WARN("Online partitioning: ports map is missing an interconnect entry (prod="
                             << gptr_prod->getId() << ", cons=" << getId() << ", input_layer="
                             << input_layer->get_friendly_name() << ", layer=" << layer->get_friendly_name()
                             << "). This edge will be ignored.");
                    continue;
                }
                const auto& ports = ports_it->second;
                ics.insert({input_layer, ports.first, layer, ports.second});
            }
        }
    }

    return ics;
}

void Group::addWeightsPrecision(const std::vector<ov::element::Type>& prec) {
    m_consts_precision.insert(m_consts_precision.end(), prec.begin(), prec.end());
}

const std::vector<ov::element::Type>& Group::getConstsPrecision() const {
    return m_consts_precision;
}

std::string Group::specialTags() const {
    std::string tags = "";

    if (m_nofold) {
        tags += "nofold";
    }

    if (!m_isol_tag.empty()) {
        tags += m_isol_tag;
    }

    return tags;
}

void Group::avoid(const std::string& device) {
    m_avoided_devices.insert(device);
}

const std::set<std::string>& Group::avoidedTargets() const {
    return m_avoided_devices;
}

void Group::isolate(const std::string& tag) {
    m_isol_tag = tag;
}

void Group::dontIsolate() {
    m_isol_tag = "";
}

const std::string& Group::isolatedTag() const {
    return m_isol_tag;
}

void Group::dump() const {
    LOG_BLOCK();
    for (auto&& layer : m_content) {
        LOG_DEBUG(layer);
    }
}
