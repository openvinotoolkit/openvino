// Copyright (C) 2024 Intel Corporationov::npuw::
// SPDX-License-Identifier: Apache-2.0
//

#include "snapshot.hpp"

#include "../../logging.hpp"
#include "../patterns/avoid.hpp"  // RMSNormPattern
#include "group.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/util/common_util.hpp"
#include "utils/utils.hpp"

using ov::npuw::online::Group;
using ov::npuw::online::Repeated;
using ov::npuw::online::Snapshot;
using ov::npuw::online::detail::GPtrSet;
using ov::npuw::online::detail::OVNodePtr;
using ov::npuw::online::detail::OVNodeSet;
using ov::npuw::online::detail::OVPortsMap;
using ov::npuw::online::detail::Uniques;

namespace ov {
namespace npuw {
namespace online {
namespace detail {
bool isOp(const std::shared_ptr<ov::Node>& node) {
    if (ov::op::util::is_constant(node) || ov::op::util::is_parameter(node) || ov::op::util::is_output(node)) {
        return false;
    }
    if (ov::is_type<ov::opset1::Convert>(node)) {
        if (node->inputs().size() != 1) {
            // can occur only in Const->Convert->Node case
            return false;
        }
        auto target_input = node->get_input_source_output(0);
        auto parent_node = target_input.get_node()->shared_from_this();
        if (ov::op::util::is_constant(parent_node)) {
            return false;
        }
    }
    return true;
}
}  // namespace detail
}  // namespace online
}  // namespace npuw
}  // namespace ov

using ov::npuw::online::detail::isOp;

void Snapshot::buildGraph() {
    LOG_INFO("Online partitioning: parsing OV Model to initial groups...");
    LOG_BLOCK();

    size_t gid = 0;  // unique group id

    // Traverse OV layers
    for (const auto& ov_node : m_model->get_ordered_ops()) {
        if (!isOp(ov_node)) {
            continue;
        }

        m_node_to_prod_cons->insert({ov_node, {}});

        auto nh = m_graph->create();
        auto group = std::make_shared<Group>(ov_node, gid, nh, m_graph, shared_from_this());
        m_graph->meta(nh).set(group);
        m_node_to_gr->emplace(std::make_pair(ov_node, group));
        ++gid;
    }

    for (const auto& nh : m_graph->sorted()) {
        auto gptr = m_graph->meta(nh).get<Group::GPtr>();
        auto ov_node = gptr->getInitialNode();

        for (size_t i = 0; i < ov_node->outputs().size(); ++i) {
            const auto target_outputs = ov_node->get_output_target_inputs(i);

            for (const auto& target_output : target_outputs) {
                auto ov_node_child = target_output.get_node()->shared_from_this();

                // Insert readers from other layers
                m_node_to_prod_cons->at(ov_node).second.insert(ov_node_child);

                // Save ports for repeated blocks pipeline
                m_ports_map.insert({{ov_node, ov_node_child}, {i, target_output.get_index()}});

                if (!isOp(ov_node_child)) {
                    continue;
                }

                if (!m_graph->linked(nh, m_node_to_gr->at(ov_node_child)->getHandle())) {
                    m_graph->link(nh, m_node_to_gr->at(ov_node_child)->getHandle());
                }
            }
        }  // for(outputs)

        for (size_t i = 0; i < ov_node->inputs().size(); ++i) {
            auto target_input = ov_node->get_input_source_output(i);
            auto ov_node_parent = target_input.get_node()->shared_from_this();

            // Insert writers from other layers
            m_node_to_prod_cons->at(ov_node).first.insert(ov_node_parent);

            // Save ports for repeated blocks pipeline
            m_ports_map.insert({{ov_node_parent, ov_node}, {target_input.get_index(), i}});

            if (!isOp(ov_node_parent)) {
                continue;
            }

            if (!m_graph->linked(m_node_to_gr->at(ov_node_parent)->getHandle(), nh)) {
                m_graph->link(m_node_to_gr->at(ov_node_parent)->getHandle(), nh);
            }
        }  // for(inputs)
    }      // for(get_ordered_ops)

    LOG_DEBUG("Initial number of groups: " << graphSize());
    LOG_INFO("DONE.");
}

void Snapshot::collectLHF() {
    LOG_INFO("Online partitioning: executing collectLHF pass...");
    LOG_BLOCK();

    // iterate it topological order
    auto graph = getGraph();
    for (const auto& nh : graph->sorted()) {
        // skip if removed by fuse
        if (m_graph->contains(nh)) {
            Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
            auto producers = group->srcNodes();
            if (producers.size() == 1) {
                auto prod = producers.at(0);
                if (m_graph->contains(prod) && prod->dstNodes().size() == 1) {
                    Group::GPtr prod_group = m_graph->meta(prod).get<Group::GPtr>();
                    if (group->isFrozen() || prod_group->isFrozen()) {
                        continue;
                    }
                    // stop merging groups if the graph is already small enough
                    if (graphSize() <= m_ctx.min_graph_size) {
                        break;
                    }
                    group->fuse(prod_group);
                }
            }
        }
    }

    LOG_INFO("DONE");
}

void Snapshot::fuseRemnantsExtended() {
    LOG_INFO("Online partitioning: executing fuseRemnantsExtended pass...");
    LOG_BLOCK();

    repeat([&] {
        fuseRemnants();
    });
    repeat([&] {
        fuseInputs();
    });

    LOG_INFO("DONE");
}

void Snapshot::fuseRemnants() {
    LOG_INFO("Online partitioning: executing fuseRemnants pass...");
    LOG_BLOCK();

    // iterate it topological order
    auto graph = getGraph();
    for (const auto& nh : graph->sorted()) {
        // skip if removed by fuseWith
        if (m_graph->contains(nh)) {
            Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
            if (group->isFrozen()) {
                continue;
            }
            auto consumers = group->dstNodes();
            if (!consumers.empty()) {
                std::sort(consumers.begin(),
                          consumers.end(),
                          [&](const ade::NodeHandle& nh1, const ade::NodeHandle& nh2) {
                              if (!m_graph->contains(nh1) || !m_graph->contains(nh2)) {
                                  return false;
                              }
                              Group::GPtr g1 = m_graph->meta(nh1).get<Group::GPtr>();
                              Group::GPtr g2 = m_graph->meta(nh2).get<Group::GPtr>();
                              return g1->size() < g2->size();
                          });
                for (const auto& cons : consumers) {  // FIXME: pick the smallest flops
                    if (!m_graph->contains(cons)) {
                        continue;
                    }
                    Group::GPtr cons_group = m_graph->meta(cons).get<Group::GPtr>();
                    if (!group->hasCycle(cons_group)) {
                        if (!cons_group->isFrozen()) {
                            group->fuseWith(cons_group);
                            break;
                        }
                    }
                }
                // stop merging groups if the graph is already small enough
                if (graphSize() <= m_ctx.min_graph_size) {
                    break;
                }
            }
        }
    }

    LOG_INFO("DONE");
}

void Snapshot::fuseInputs() {
    LOG_INFO("Online partitioning: executing fuseInputs pass...");
    LOG_BLOCK();

    // iterate it topological order
    auto graph = getGraph();
    for (const auto& nh : graph->sorted()) {
        // skip if removed by fuseInputs
        if (!m_graph->contains(nh)) {
            continue;
        }
        Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();

        std::pair<Group::GPtr, Group::GPtr> inputs_to_fuse{nullptr, nullptr};
        auto src_nodes = group->srcNodes();
        for (size_t i = 0; i < src_nodes.size(); ++i) {
            auto prod_nh = src_nodes[i];
            if (!m_graph->contains(prod_nh)) {  // should be there, but check just in case
                continue;
            }
            Group::GPtr group_prod = m_graph->meta(prod_nh).get<Group::GPtr>();
            if (group_prod->isFrozen()) {
                continue;
            }
            inputs_to_fuse.first = group_prod;  // set the first candidate

            // Double loop here since we need to consider every pair of inputs
            for (size_t j = i + 1; j < src_nodes.size(); ++j) {
                auto prod_nh_other = src_nodes[j];
                if (!m_graph->contains(prod_nh_other)) {  // should be there, but check just in case
                    continue;
                }
                Group::GPtr group_prod_other = m_graph->meta(prod_nh_other).get<Group::GPtr>();
                if (group_prod_other->isFrozen()) {
                    continue;
                }
                if (!group_prod->hasCycle(group_prod_other) && !group_prod_other->hasCycle(group_prod)) {
                    // no cycles -> fusion allowed
                    inputs_to_fuse.second = std::move(group_prod_other);
                    break;
                }
            }
            // Found 2 inputs to fuse
            if (inputs_to_fuse.first && inputs_to_fuse.second) {
                group->fuseInputs(inputs_to_fuse);
                break;
            }
        }

        // stop merging groups if the graph is already small enough
        if (graphSize() <= m_ctx.min_graph_size) {
            break;
        }
    }

    LOG_INFO("DONE");
}

void Snapshot::earlyAvoids() {
    ov::pass::GraphRewrite rewr;
    bool handle_patterns = false;

    for (const auto& avoid : m_ctx.avoids) {
        switch (avoid.type) {
        case AvoidType::OP: {
            for (const auto& nh : m_graph->sorted()) {
                Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
                // This pass should only be called at the very beginning,
                // thus check and match only the single initial layer
                if (group->getInitialNode()->description() == avoid.pattern) {
                    group->avoid(avoid.device);
                }
            }
            break;
        }
        case AvoidType::PATTERN: {
            // FIXME: refactor as more patterns are supported
            if (avoid.pattern != "RMSNorm") {
                LOG_WARN("OPENVINO_NPUW_AVOID only supports RMSNorm as a pattern (don't confuse with operations)."
                         << " Avoid pattern " << avoid.pattern << " is skipped!");
                break;
            }
            handle_patterns = true;

            rewr.add_matcher<ov::npuw::patterns::RMSNormPattern>(shared_from_this(), avoid.device);
            break;
        }
        }
    }

    if (handle_patterns) {
        // Check the model for all specified patterns
        rewr.run_on_model(m_model);
    }
}

void Snapshot::repeatedBlocks() {
    LOG_INFO("Online partitioning: executing repeatedBlocks pass group...");
    LOG_BLOCK();

    identifyUniques();
    repeat([&] {
        mergeUniques();
    });
    cleanUpUniques();

    LOG_INFO("DONE");
}

void Snapshot::identifyUniques() {
    LOG_INFO("Online partitioning: executing identifyUniques pass...");
    LOG_BLOCK();

    Uniques uniques;

    for (const auto& nh : m_graph->sorted()) {
        Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
        // This pass should only be called at the very beginning,
        // thus check and use only the single initial layer
        auto ov_node = group->getInitialNode();
        auto metadesc = ov::npuw::online::util::getMetaDesc(ov_node);
        const auto& avoids = group->avoidedTargets();
        uniques[{metadesc, avoids}].insert(group);
    }

    for (const auto& elem : uniques) {
        if (elem.second.size() > 1) {
            std::shared_ptr<Repeated> rep = std::make_shared<Repeated>();

            for (const auto& gptr : elem.second) {
                gptr->setRepeated(rep);
            }
        }
    }

    LOG_INFO("DONE");
}

void Snapshot::mergeUniques() {
    LOG_INFO("Online partitioning: executing mergeUniques pass...");
    LOG_BLOCK();

    std::unordered_set<std::shared_ptr<Repeated>> merged_this_time;

    auto graph = getGraph();
    for (const auto& nh : graph->sorted()) {
        if (m_graph->contains(nh)) {
            Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
            auto rep = group->repeated();

            detail::GPtrSet repeating_groups;

            if (rep && rep->openForMerge() && merged_this_time.count(rep) == 0) {
                for (const auto& nh_other : graph->sorted()) {
                    if (m_graph->contains(nh_other)) {
                        Group::GPtr group_other = m_graph->meta(nh_other).get<Group::GPtr>();
                        auto rep_other = group_other->repeated();

                        if (rep_other && (rep_other.get() == rep.get())) {
                            repeating_groups.insert(group_other);
                        }
                    }
                }
            }

            if (!repeating_groups.empty()) {
                auto new_rep = tryGrowRepeatingGroups(repeating_groups);
                if (new_rep) {
                    merged_this_time.insert(new_rep);
                }
            }
        }
    }

    LOG_INFO("DONE");
}

std::shared_ptr<Repeated> Snapshot::tryGrowRepeatingGroups(const detail::GPtrSet& repeating_groups) {
    auto this_rep_tag = (*(repeating_groups.begin()))->repeated();  // should be the same for each group inside
    const auto& this_avoided = (*(repeating_groups.begin()))->avoidedTargets();

    std::unordered_map<std::vector<MetaInterconnect>, std::vector<std::pair<Group::GPtr, Group::GPtr>>> mics;

    std::vector<Group::GPtr> repeating_groups_sorted(repeating_groups.begin(), repeating_groups.end());

    // FIXME: this was introduced to make the partitioning
    // the same every run when created the same way.
    // This std::sort allows to prioritize the groups from the tail
    // of the original model. It's possible due to preservation of
    // group IDs in topological order throughout the whole partitioning process.
    // In the networks we're looking at, ensuring the merge order from the bottom
    // of the network gives a better generalization for the identified repeated blocks,
    // e.g. we can guarantee we can find one more, which otherwise would fuse into
    // head or tail (depending on the topology).
    std::sort(repeating_groups_sorted.begin(),
              repeating_groups_sorted.end(),
              [&](const Group::GPtr& gptr_a, const Group::GPtr& gptr_b) {
                  return gptr_a->getId() > gptr_b->getId();
              });

    for (const auto& group : repeating_groups_sorted) {
        auto producers = group->srcNodes();
        for (const auto& prod_nh : producers) {
            if (m_graph->contains(prod_nh)) {
                Group::GPtr prod_group = m_graph->meta(prod_nh).get<Group::GPtr>();
                if (prod_group->repeated() && !prod_group->hasCycle(group) && prod_group->repeated() != this_rep_tag &&
                    prod_group->avoidedTargets() == this_avoided) {
                    auto meta_interconnect = group->metaInterconnect(prod_group);

                    // FIXME: find a better way to reduce time complexity
                    // Need to align interconnects in the same format via sort, so they could be compared later
                    std::vector<MetaInterconnect> mic_sorted_key(meta_interconnect.begin(), meta_interconnect.end());
                    std::sort(mic_sorted_key.begin(), mic_sorted_key.end());
                    mics[mic_sorted_key].push_back({prod_group, group});
                }
            }
        }
    }

    // FIXME: find a better way to reduce time complexity
    // Below we sort meta interconnects by size, so we could try to merge the bigger ones first
    std::vector<std::vector<std::pair<Group::GPtr, Group::GPtr>>> mics_vec;
    for (const auto& mic : mics) {
        mics_vec.push_back(mic.second);
    }

    std::sort(mics_vec.begin(), mics_vec.end(), [](const auto& a, const auto& b) {
        if (a.size() == b.size()) {
            if (a.empty()) {
                return false;  // doesn't matter for stability - no groups are fused
            }
            // This std::sort allows to prioritize groups from the tail
            // of the original model. It's possible due to preservation of
            // group IDs in topological order throughout the whole partitioning process.
            // In the networks we're looking at, ensuring the merge order from the bottom
            // of the network gives a better structure of a repeated block which can be
            // later optimized by the plugin.
            return a.at(0).first->getId() > b.at(0).first->getId();
        }
        // Generally we prefer bigger blocks (in terms of number of layers)
        // to be merged first. For other cases check the comment above
        return a.size() > b.size();
    });

    for (const auto& mic : mics_vec) {
        std::vector<Group::GPtr> prods;
        std::vector<Group::GPtr> conss;

        for (const auto& el : mic) {
            prods.push_back(el.first);
            conss.push_back(el.second);
        }

        auto new_rep = tryMergeRepeating(prods, conss);
        if (new_rep) {
            return new_rep;
        }
    }

    // No merges happened at all? Exclude this group from the merge procedure and indicate via return value.
    this_rep_tag->exclude();

    return nullptr;
}

std::shared_ptr<Repeated> Snapshot::tryMergeRepeating(const std::vector<Group::GPtr>& prods,
                                                      const std::vector<Group::GPtr>& conss) {
    if (prods.size() != conss.size()) {
        // FIXME: it's actually possible to merge under certain circumstances
        OPENVINO_THROW(
            "Online partitioning tried to merge repeated groups with different sizes of producers and consumers!");
    }

    if (conss.size() == 1) {
        return nullptr;
    }

    std::unordered_set<Group::GPtr> prods_set;
    for (const auto& prod : prods) {
        prods_set.insert(prod);
    }

    if (prods_set.size() != conss.size()) {
        return nullptr;
    }

    for (const auto& cons : conss) {
        if (std::find(prods.begin(), prods.end(), cons) != prods.end()) {
            OPENVINO_THROW("Online partitioning tried to merge repeated groups which overlap!");
        }
    }

    std::shared_ptr<Repeated> new_rep = std::make_shared<Repeated>();

    for (size_t i = 0; i < conss.size(); ++i) {
        conss.at(i)->fuse(prods.at(i));
        conss.at(i)->setRepeated(new_rep);  // producer is consumed, no need to setRepeated() it
    }

    for (const auto& cons : conss) {
        auto prod_nhs = cons->srcNodes();
        for (const auto& nh : prod_nhs) {
            Group::GPtr prod_group = m_graph->meta(nh).get<Group::GPtr>();
            if (prod_group == cons) {
                OPENVINO_THROW(
                    "Online partitioning have merged repeated groups incorrectly: producers/consumers overlap!");
            }
        }
    }

    return new_rep;
}

std::unordered_map<std::shared_ptr<Repeated>, ov::npuw::online::detail::GPtrSet> Snapshot::repeating() const {
    std::unordered_map<std::shared_ptr<Repeated>, detail::GPtrSet> repeating;
    for (const auto& nh : m_graph->sorted()) {
        Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
        auto rep = group->repeated();
        if (rep) {
            repeating[rep].insert(group);
        }
    }

    return repeating;
}

void Snapshot::cleanUpUniques() {
    LOG_INFO("Online partitioning: executing cleanUpUniques pass...");
    LOG_BLOCK();

    for (auto& reptag_to_gset : repeating()) {
        bool keep = cleanUpUniquesImpl(reptag_to_gset.second);

        if (!keep) {
            continue;  // If we dropped repeated blocks in cleanUpUniquesImpl, skip the next section
        }

        completeRepeating(reptag_to_gset.first, reptag_to_gset.second);
    }

    LOG_INFO("DONE");
}

bool Snapshot::cleanUpUniquesImpl(const GPtrSet& gptrs) {
    for (const auto& gptr : gptrs) {
        if (!gptr->avoidedTargets().empty()) {
            // Special case - keep it
            for (const auto& g : gptrs) {
                g->freeze();
            }
            return true;
        }
    }

    // FIXME: in ensemble we check threshold of 10% of the total compute to discard a repeated block
    // Since we don't have flops here yet, let's just check the size of the group
    auto block_layer_size = (*(gptrs.begin()))->size();
    if (gptrs.size() < 2 || block_layer_size < 10 || gptrs.size() * block_layer_size < 100) {
        for (const auto& gptr : gptrs) {
            gptr->setRepeated(nullptr);
        }
        LOG_DEBUG("Repeated block of " << gptrs.size() << " groups with " << block_layer_size << " layers is dropped.");
        return false;
    }

    LOG_DEBUG("Keeping a repeated block of " << gptrs.size() << " groups with " << block_layer_size << " layers.");

    // Repeated block is good enough to keep
    for (const auto& gptr : gptrs) {
        gptr->freeze();
    }

    return true;
}

void Snapshot::completeRepeating(const std::shared_ptr<Repeated>& reptag, const detail::GPtrSet& gset) {
    std::unordered_map<Repeated::Archetype, std::unordered_set<ov::npuw::online::detail::OVNodePtr>> matches;

    for (const auto& gptr : gset) {
        for (const auto& layer : gptr->getContent()) {  // FIXME: should it be a part of group's API instead?
            const auto& metadesc = ov::npuw::online::util::getMetaDesc(layer);
            const auto& archetype = gptr->getReptrack(layer);
            matches[{std::move(metadesc), std::move(archetype)}].insert(layer);
        }
    }

    // Sanity check:
    // 1. For every node archetype, there must be the same number of instances:
    //    equal to the number of groups.
    // 2. Total count of archetypes must be equal to size of every individual group
    for (const auto& elem : matches) {
        const auto& node_set = elem.second;
        if (node_set.size() != gset.size()) {
            OPENVINO_THROW("Online partitioning couldn't match properly "
                           "during repeated blocks pass (node archetype). "
                           "Got ",
                           node_set.size(),
                           ", expected ",
                           gset.size());
        }
    }
    for (const auto& gptr : gset) {
        if (matches.size() != gptr->getContent().size()) {
            OPENVINO_THROW("Online partitioning couldn't match properly "
                           "during repeated blocks pass (count of archetypes). "
                           "Got ",
                           matches.size(),
                           ", expected ",
                           gptr->getContent().size());
        }
    }

    std::vector<std::set<std::string>> layer_matches;
    for (const auto& elem : matches) {
        layer_matches.push_back({});
        for (const auto& layer : elem.second) {
            layer_matches.back().insert(layer->get_friendly_name());
        }
    }

    std::string tag = ov::npuw::online::util::repeated_id(reptag);
    m_layer_matches.insert({tag, layer_matches});
}

const OVNodeSet& Snapshot::getNodeProducers(const OVNodePtr& node) const {
    return m_node_to_prod_cons->at(node).first;
}

const OVNodeSet& Snapshot::getNodeConsumers(const OVNodePtr& node) const {
    return m_node_to_prod_cons->at(node).second;
}

const ov::npuw::online::detail::OVNodeToGroupMapPtr& Snapshot::getNodeToGroupMap() const {
    return m_node_to_gr;
}

std::shared_ptr<ade::Graph> Snapshot::getGraph() const {
    return m_graph;
}

size_t Snapshot::graphSize() const {
    return m_graph->nodes().size();
}

const OVPortsMap& Snapshot::getPortsMap() const {
    return m_ports_map;
}

const std::map<std::string, std::vector<std::set<std::string>>>& Snapshot::getMatches() const {
    return m_layer_matches;
}

void Snapshot::repeat(detail::Pass&& pass) {
    size_t prev_graph_size = 0;
    size_t curr_graph_size = graphSize();

    while (graphSize() > m_ctx.min_graph_size && curr_graph_size != prev_graph_size) {
        prev_graph_size = graphSize();
        pass();
        curr_graph_size = graphSize();
    }

    LOG_INFO("Number of groups after compiler pass: " << graphSize());
}

void Snapshot::setCtx(const ov::npuw::online::PassContext& ctx) {
    m_ctx = ctx;
}
