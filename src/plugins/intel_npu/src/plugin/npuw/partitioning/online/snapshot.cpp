// Copyright (C) 2024 Intel Corporationov::npuw::
// SPDX-License-Identifier: Apache-2.0
//

#include "snapshot.hpp"

#include "../../logging.hpp"
#include "../../util.hpp"
#include "../patterns/avoid.hpp"
#include "../patterns/compute.hpp"
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

std::vector<ov::element::Type> getConstsPrecision(const std::shared_ptr<ov::Node>& node) {
    NPUW_ASSERT(!ov::op::util::is_constant(node) && !ov::op::util::is_parameter(node) &&
                !ov::op::util::is_output(node));

    std::vector<ov::element::Type> precisions;

    for (size_t i = 0; i < node->inputs().size(); ++i) {
        auto target_input = node->get_input_source_output(i);
        auto ov_node_parent = target_input.get_node()->shared_from_this();

        if (ov::is_type<ov::opset1::Convert>(ov_node_parent)) {
            auto target_op_input = ov_node_parent->get_input_source_output(0);
            auto parent_op_node = target_op_input.get_node()->shared_from_this();

            if (ov::op::util::is_constant(parent_op_node)) {
                precisions.push_back(parent_op_node->get_element_type());
            }
        }
    }

    return precisions;
}
}  // namespace detail
}  // namespace online
}  // namespace npuw
}  // namespace ov

using ov::npuw::online::detail::getConstsPrecision;
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
        group->addWeightsPrecision(getConstsPrecision(ov_node));
        m_graph->meta(nh).set(group);
        m_node_to_gr->emplace(std::make_pair(ov_node, group));
        ++gid;
    }

    using namespace ov::npuw::util::at;

    for (const auto& nh : m_graph->sorted()) {
        auto gptr = m_graph->meta(nh).get<Group::GPtr>();
        auto ov_node = gptr->getInitialNode();

        for (size_t i = 0; i < ov_node->outputs().size(); ++i) {
            const auto target_outputs = ov_node->get_output_target_inputs(i);

            for (const auto& target_output : target_outputs) {
                auto ov_node_child = target_output.get_node()->shared_from_this();

                // Insert readers from other layers
                _(m_node_to_prod_cons).at(ov_node).second.insert(ov_node_child);

                // Save ports for repeated blocks pipeline
                m_ports_map.insert({{ov_node, ov_node_child}, {i, target_output.get_index()}});

                if (!isOp(ov_node_child)) {
                    continue;
                }
                Group::GPtr gr_child = _(m_node_to_gr).at(ov_node_child);
                if (!m_graph->linked(nh, gr_child->getHandle())) {
                    m_graph->link(nh, gr_child->getHandle());
                }
            }
        }  // for(outputs)

        for (size_t i = 0; i < ov_node->inputs().size(); ++i) {
            auto target_input = ov_node->get_input_source_output(i);
            auto ov_node_parent = target_input.get_node()->shared_from_this();

            // Insert writers from other layers
            _(m_node_to_prod_cons).at(ov_node).first.insert(ov_node_parent);

            // Save ports for repeated blocks pipeline
            m_ports_map.insert({{ov_node_parent, ov_node}, {target_input.get_index(), i}});

            if (!isOp(ov_node_parent)) {
                continue;
            }

            Group::GPtr gr_parent = _(m_node_to_gr).at(ov_node_parent);
            if (!m_graph->linked(gr_parent->getHandle(), nh)) {
                m_graph->link(gr_parent->getHandle(), nh);
            }
        }  // for(inputs)
    }      // for(get_ordered_ops)

    LOG_DEBUG("Initial number of groups: " << graphSize());
    LOG_INFO("DONE.");
}

void Snapshot::splitMixedPrecision() {
    LOG_INFO("Online partitioning: executing splitMixedPrecision pass...");
    LOG_BLOCK();

    auto reptag_to_gset = repeating();
    // Iterate over repeated blocks
    for (const auto& elem : reptag_to_gset) {
        auto reptag = elem.first;
        auto gset = elem.second;

        // Fill a map of ordered consts precisions to a Group
        std::unordered_map<std::vector<ov::element::Type>, GPtrSet> prec_to_new_gset;
        for (const auto& gptr : gset) {
            prec_to_new_gset[gptr->getConstsPrecision()].insert(gptr);
        }

        // In case all precisions match - skip
        if (prec_to_new_gset.size() == 1) {
            continue;
        }

        // Otherwise need to split repeated block based on consts precisions
        for (const auto& elem : prec_to_new_gset) {
            // Assign new reptags - basically create a new repeated block
            std::shared_ptr<Repeated> rep = std::make_shared<Repeated>();

            LOG_VERB("Identified mixed precision, splitting a new repeated block of " << elem.second.size()
                                                                                      << " groups.");

            for (const auto& gptr : elem.second) {
                gptr->setRepeated(rep);
            }
        }
    }

    LOG_INFO("DONE");
}

void Snapshot::singleGroup() {
    LOG_INFO("Online partitioning: executing singleGroup pass...");
    LOG_BLOCK();

    auto nh = m_graph->create();
    auto group = std::make_shared<Group>(0, nh, m_graph, shared_from_this());
    m_graph->meta(nh).set(group);

    for (const auto& node : m_model->get_ordered_ops()) {
        if (ov::op::util::is_parameter(node)) {
            auto readers = node->output(0).get_target_inputs();
            for (auto&& r : readers) {
                group->addInput(r.get_node()->shared_from_this());
            }
        } else if (ov::op::util::is_output(node)) {
            group->addOutput(node->input(0).get_source_output().get_node_shared_ptr());
        } else if (isOp(node)) {
            group->addContent(node);
        }
    }  // for (get_ordered_ops)

    NPUW_ASSERT(graphSize() == 1);

    LOG_INFO("DONE.");
}

void Snapshot::collectLHF() {
    LOG_INFO("Online partitioning: executing collectLHF pass...");
    LOG_BLOCK();

    // iterate it topological order
    for (const auto& nh : m_graph->sorted()) {
        // skip if removed by fuse
        if (!m_graph->contains(nh)) {
            continue;
        }
        Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
        auto producers = group->srcNodes();
        if (producers.size() == 1) {
            auto prod = producers.at(0);
            if (prod->dstNodes().size() == 1) {
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
    for (const auto& nh : m_graph->sorted()) {
        // skip if removed by fuseWith
        if (!m_graph->contains(nh)) {
            continue;
        }
        Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
        if (group->isFrozen()) {
            continue;
        }
        auto consumers = group->dstNodes();
        if (!consumers.empty()) {
            std::sort(consumers.begin(),
                      consumers.end(),
                      [&](const own::ade::NodeHandle& nh1, const own::ade::NodeHandle& nh2) {
                          Group::GPtr g1 = m_graph->meta(nh1).get<Group::GPtr>();
                          Group::GPtr g2 = m_graph->meta(nh2).get<Group::GPtr>();
                          return g1->size() < g2->size();
                      });
            for (const auto& cons : consumers) {  // FIXME: pick the smallest flops
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

    LOG_INFO("DONE");
}

void Snapshot::fuseInputs() {
    LOG_INFO("Online partitioning: executing fuseInputs pass...");
    LOG_BLOCK();

    // iterate it topological order
    for (const auto& nh : m_graph->sorted()) {
        // skip if removed by fuseInputs
        if (!m_graph->contains(nh)) {
            continue;
        }
        Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();

        std::pair<Group::GPtr, Group::GPtr> inputs_to_fuse{nullptr, nullptr};
        auto src_nodes = group->srcNodes();
        for (size_t i = 0; i < src_nodes.size(); ++i) {
            auto prod_nh = src_nodes[i];
            Group::GPtr group_prod = m_graph->meta(prod_nh).get<Group::GPtr>();
            if (group_prod->isFrozen()) {
                continue;
            }
            inputs_to_fuse.first = group_prod;  // set the first candidate

            // Double loop here since we need to consider every pair of inputs
            for (size_t j = i + 1; j < src_nodes.size(); ++j) {
                auto prod_nh_other = src_nodes[j];
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

void Snapshot::markInternalCompute() {
    LOG_INFO("Online partitioning: executing markInternalCompute pass...");
    LOG_BLOCK();

    // Iterate over groups and drop all "fake" tags.
    // It's done for markInternalCompute pass to work properly if
    // there are multiple tags used to not split internal patterns.
    for (const auto& nh : m_graph->sorted()) {
        Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
        if (!group->isolatedTag().empty() && group->isolatedTag() == "fake") {
            group->dontIsolate();
        }
    }

    // iterate it topological order
    for (const auto& nh : m_graph->sorted()) {
        Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
        if (!group->specialTags().empty() || !group->repeated()) {  // we only need repeated groups with empty tags
            continue;
        }

        // We need to filter out compute group
        // with all of it's producers and consumers have the same tags
        std::unordered_set<std::string> prod_cons_tags;
        for (const auto& prod_nh : group->srcNodes()) {
            Group::GPtr group_prod = m_graph->meta(prod_nh).get<Group::GPtr>();
            prod_cons_tags.insert(group_prod->specialTags());
        }
        for (const auto& cons_nh : group->dstNodes()) {
            Group::GPtr group_cons = m_graph->meta(cons_nh).get<Group::GPtr>();
            prod_cons_tags.insert(group_cons->specialTags());
        }
        if (prod_cons_tags.size() == 1 && !(*prod_cons_tags.begin()).empty()) {
            NPUW_ASSERT(!group->srcNodes().empty());
            auto prod_nh = group->srcNodes().at(0);  // all tags are the same, pick either group
            Group::GPtr group_prod = m_graph->meta(prod_nh).get<Group::GPtr>();
            NPUW_ASSERT(!group_prod->isolatedTag().empty());
            if (group_prod->isolatedTag() !=
                "compute") {  // this pass only operates with "compute" tag set by COMPUTE pipeline
                continue;
            }
            group->isolate(group_prod->isolatedTag());
        }
    }

    LOG_INFO("DONE");
}

void Snapshot::resetExcludedRep() {
    for (const auto& nh : m_graph->sorted()) {
        Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
        auto rep = group->repeated();
        if (rep) {
            rep->resetExclude();
        }
    }
}

void Snapshot::earlyAvoids() {
    LOG_INFO("Online partitioning: executing earlyAvoids pass...");
    LOG_BLOCK();

    ov::pass::GraphRewrite rewr;
    bool handle_patterns = false;

    for (const auto& avoid : m_ctx.avoids) {
        switch (avoid.type) {
        case PatternType::OP: {
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
        case PatternType::PATTERN: {
            // FIXME: refactor as more patterns are supported
            if (avoid.pattern != "RMSNorm") {
                LOG_WARN("OPENVINO_NPUW_AVOID only supports RMSNorm as a pattern (don't confuse with operations)."
                         << " Avoid pattern " << avoid.pattern << " is skipped!");
                break;
            }
            handle_patterns = true;

            rewr.add_matcher<ov::npuw::patterns::avoid::RMSNorm>(shared_from_this(), avoid.device);
            break;
        }
        }
    }

    if (handle_patterns) {
        // Check the model for all specified patterns
        rewr.run_on_model(m_model);
    }

    LOG_INFO("DONE.");
}

void Snapshot::earlyRegroup() {
    LOG_INFO("Online partitioning: executing earlyRegroup pass...");
    LOG_BLOCK();

    ov::pass::GraphRewrite rewr;
    ov::pass::GraphRewrite rewr_fake;
    bool handle_patterns = false;

    for (const auto& isolate : m_ctx.isolates) {
        switch (isolate.type) {
        case PatternType::OP: {
            for (const auto& nh : m_graph->sorted()) {
                Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
                // This pass should only be called at the very beginning,
                // thus check and match only the single initial layer
                if (group->getInitialNode()->description() == isolate.pattern) {
                    group->isolate(isolate.tag);
                }
            }
            break;
        }
        case PatternType::PATTERN: {
#define HNDL(p)                                                                            \
    if (isolate.pattern == #p) {                                                           \
        rewr.add_matcher<ov::npuw::patterns::compute::p>(shared_from_this(), isolate.tag); \
        handle_patterns = true;                                                            \
    }
#define HNDL_FAKE(p)                                                                            \
    if (isolate.pattern == #p) {                                                                \
        rewr_fake.add_matcher<ov::npuw::patterns::compute::p>(shared_from_this(), isolate.tag); \
        handle_patterns = true;                                                                 \
    }
            HNDL(RMSNorm);
            HNDL(RMSNorm2);
            HNDL(DQMatMulCWu4);
            HNDL(DQMatMulGQu4);
            HNDL(DQMatMulCWi4);
            HNDL(DQMatMulGQi4);
            HNDL(DQMatMulConv);
            HNDL(VocabMatMul);
            HNDL(VariadicSplit);
            HNDL_FAKE(FakeConvert);
            HNDL_FAKE(FakeQuantize);
#undef HNDL_FAKE
#undef HNDL
        }
        }
    }

    if (handle_patterns) {
        // Check the model for all specified patterns
        // Note: it's important to run Fake patterns first so it won't mix with the compute ones
        rewr_fake.run_on_model(m_model);
        rewr.run_on_model(m_model);
    }

    LOG_INFO("DONE.");
}

void Snapshot::repeatedBlocks(Snapshot::CB&& on_done) {
    LOG_INFO("Online partitioning: executing repeatedBlocks pass group...");
    LOG_BLOCK();

    identifyUniques();
    repeat([&] {
        repeat([&] {
            repeat([&] {
                mergeUniques();
            });
            mergeTriangles();
            markInternalCompute();
            resetExcludedRep();
        });
        // While the current process is entirely done, let the caller
        // influence the partitioning - so the algorithm could continue.
        if (on_done) {
            on_done();
        } else {
            return;  // FROM top-level repeat!
        }
    });
    splitMixedPrecision();
    cleanUpUniques();

    LOG_INFO("Number of groups after compiler pass: " << graphSize());

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
        const auto& special_tags = group->specialTags();
        uniques[{metadesc, avoids, special_tags}].insert(group);
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

void Snapshot::mergeTriangles() {
    LOG_INFO("Online partitioning: executing mergeTriangles pass...");
    LOG_BLOCK();

    // Handle a special case where one repeating group can be a producer
    // to multiple other repeating groups at once, like in
    //
    //       A1             A2            A3
    //    .  .  .        .  .  .       .  .  .
    //    :  :  :        :  :  :       :  :  :
    //    B1 B2 B3       B4 B5 B6      B7 B8 B9
    //
    // mergeUniques doesn't handle this case - when two candidate vectors
    // (producers + consumers) are selected for this merge,
    // 1. We'll get the two vectors as [ A1 A1 A1 A2 A2 A2 A3 A3 A3 ] x
    //    [ B1 B2 B3 B4 B5 B6 B7 B8 B9 ]
    // 2. We'll 'squash' the A vector to set to check the inconsistency,
    //    will get a [ A1 A2 A3 ] set which won't match the original one,
    //    and fail the test to merge

    std::unordered_set<std::shared_ptr<Repeated>> merged_this_time;

    for (const auto& nh : m_graph->sorted()) {
        if (!m_graph->contains(nh)) {
            continue;
        }

        Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
        auto rep = group->repeated();

        GPtrSet repeating_groups;

        // Note: openForMerge is not used here
        if (rep && !group->isFrozen() && merged_this_time.count(rep) == 0) {
            repeating_groups = getRepGroups(group);
        }

        if (!repeating_groups.empty()) {
            auto new_rep = tryMergeTriangles(repeating_groups);
            if (new_rep) {
                merged_this_time.insert(new_rep);
            }
        }
    }

    LOG_INFO("Number of groups after compiler pass: " << graphSize());
    LOG_INFO("DONE");
}

// FIXME: At this point, it is almost a full duplicate of tryGrowRepeatingGroups
std::shared_ptr<Repeated> Snapshot::tryMergeTriangles(const GPtrSet& repeating_groups) {
    const auto& first_rep_group = *(repeating_groups.begin());
    // Those 3 should be the same for each group inside
    auto this_rep_tag = first_rep_group->repeated();
    const auto& this_avoided = first_rep_group->avoidedTargets();
    const auto& this_special = first_rep_group->specialTags();

    if (repeating_groups.size() < 2) {
        return {};
    }

    std::unordered_map<std::vector<MetaInterconnect>, std::unordered_map<Group::GPtr, std::unordered_set<Group::GPtr>>>
        mics;

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
    // FIXME: might not be needed for triangles at all
    std::sort(repeating_groups_sorted.begin(),
              repeating_groups_sorted.end(),
              [&](const Group::GPtr& gptr_a, const Group::GPtr& gptr_b) {
                  return gptr_a->getId() > gptr_b->getId();
              });

    for (const auto& group : repeating_groups_sorted) {
        auto consumers = group->dstNodes();
        for (const auto& cons_nh : consumers) {
            Group::GPtr cons_group = m_graph->meta(cons_nh).get<Group::GPtr>();
            if (cons_group->repeated() && !group->hasCycle(cons_group) && cons_group->repeated() != this_rep_tag &&
                cons_group->avoidedTargets() == this_avoided && cons_group->specialTags() == this_special) {
                auto meta_interconnect = cons_group->metaInterconnect(group);

                // FIXME: find a better way to reduce time complexity
                // Need to align interconnects in the same format via sort, so they could be compared later
                std::vector<MetaInterconnect> mic_sorted_key(meta_interconnect.begin(), meta_interconnect.end());
                std::sort(mic_sorted_key.begin(), mic_sorted_key.end());

                auto& triangle = mics[mic_sorted_key];
                triangle[group].insert(cons_group);
            }
        }
    }

    // FIXME: find a better way to reduce time complexity
    // Below we sort meta interconnects by size, so we could try to merge the bigger ones first
    // Wrapping as:
    // 0. Meta interconnect
    // 1. Repeated triangle
    // 2. Pair of apex + base
    std::vector<std::vector<std::pair<Group::GPtr, std::vector<Group::GPtr>>>> mics_vec;
    for (const auto& mic : mics) {
        mics_vec.push_back({});
        for (const auto& apex_n_base : mic.second) {
            std::vector<Group::GPtr> base(apex_n_base.second.begin(), apex_n_base.second.end());
            mics_vec.back().push_back({apex_n_base.first, base});
        }
        // FIXME: this was introduced to make the partitioning
        // the same every run when created the same way.
        // Worsens time complexity
        std::sort(mics_vec.back().begin(), mics_vec.back().end(), [](const auto& a, const auto& b) {
            return a.first->getId() > b.first->getId();
        });
    }

    std::sort(mics_vec.begin(), mics_vec.end(), [](const auto& a, const auto& b) {
        if (a.size() == b.size()) {
            if (a.empty()) {
                return false;  // doesn't matter for stability - no groups are fused
            }
            return a.at(0).first->getId() > b.at(0).first->getId();
        }
        return a.size() > b.size();
    });

    for (const auto& mic : mics_vec) {
        std::vector<Group::GPtr> prods;
        std::vector<std::vector<Group::GPtr>> conss;

        for (const auto& el : mic) {
            prods.push_back(el.first);
            conss.push_back(el.second);
        }

        auto new_rep = tryMergeTriangles(prods, conss);
        if (new_rep) {
            return new_rep;
        }
    }

    // As this set of passes ignores `excluded` groups, dont exclude here too

    return {};
}

std::shared_ptr<Repeated> Snapshot::tryMergeTriangles(const std::vector<Group::GPtr>& prods,
                                                      const std::vector<std::vector<Group::GPtr>>& conss) {
    if (prods.size() != conss.size()) {
        // FIXME: it's actually possible to merge under certain circumstances?
        OPENVINO_THROW(
            "Online partitioning tried to merge repeated triangles with different sizes of producers and consumers!");
    }

    if (prods.size() < 2) {
        return {};
    }

    if (prods.size() < m_ctx.keep_blocks) {
        // In some cases (specifically mixed precision) during MergeUniques() pass we could be left with
        // E.g. 10 repeated blocks with tag AAA and 2 repeated blocks with tag BBB
        // TryMergeTriangles() pass checks that producer and consumer have a different tag to be merged further.
        // Let's say in our example 10 AAA blocks are finalized and cannot be merged further due to above check.
        // However we will proceed to merge 3 BBB blocks with 3 AAA blocks since the tags are different.
        // This will create a new tag CCC for the merged blocks and the merge will continue until those 3 blocks
        // consume a large amount of legit AAA blocks.
        // Later in CleanUpUniques() pass those repeated blocks will be stripped off repeated tag due to the same check
        // in this "if". To prevent such cases where we would end up with small number of huge blocks this check was
        // introduced.
        return {};
    }

    // In this special case we only assume
    // our vector of N repeating consumer groups
    // 1. has the same size
    // 2. All consumers have a single consumer itself
    for (const auto& cons : conss) {
        if (cons.size() != conss.front().size()) {
            return {};
        }
        for (const auto& el : cons) {
            if (el->dstNodes().size() > 1 || el->srcNodes().size() > 1) {
                return {};
            }
        }
    }

    // We will try to merge the triangle base (formed by each vector in conss) into
    // the prods, but we need to make it in the right order. Remember, our conss
    // vectors are all the same so we need to distinguish it somehow. A reliable way do it is to
    // look at the conss's own metaInterconnect descriptors with their own consumers.
    // There must be difference, and we can use this difference to pick the right candidates at time.
    // This mic2 metaInterconnect is of 2nd oreder in this case.
    std::unordered_map<std::vector<MetaInterconnect>, std::vector<Group::GPtr>> mic2;
    for (const auto& cons : conss) {
        for (const auto& gptr : cons) {
            Group::GPtr group_cons = m_graph->meta(gptr->dstNodes().front()).get<Group::GPtr>();
            auto meta_interconnect = group_cons->metaInterconnect(gptr);

            // FIXME: find a better way to reduce time complexity
            // Need to align interconnects in the same format via sort, so they could be compared later
            std::vector<MetaInterconnect> mic_sorted_key(meta_interconnect.begin(), meta_interconnect.end());
            std::sort(mic_sorted_key.begin(), mic_sorted_key.end());

            mic2[mic_sorted_key].push_back(gptr);
        }
    }

    // Note: mic2.size() and conss.front().size() might not be equal here

    // Cache cons->prod pairs
    std::unordered_map<Group::GPtr, Group::GPtr> cons_prod_cache;
    for (size_t i = 0; i < prods.size(); ++i) {
        for (const auto& con : conss.at(i)) {
            cons_prod_cache[con] = prods.at(i);
        }
    }

    // Fuse bases step by step into apexes
    std::shared_ptr<Repeated> new_rep = nullptr;
    for (const auto& mic : mic2) {
        new_rep = std::make_shared<Repeated>();
        for (const auto& same_cons : mic.second) {
            auto prod = cons_prod_cache[same_cons];
            prod->fuseWith(same_cons);
            prod->setRepeated(new_rep);  // consumer is consumed, no need to setRepeated() it
        }
    }

    return new_rep;
}

void Snapshot::mergeUniques() {
    LOG_INFO("Online partitioning: executing mergeUniques pass...");
    LOG_BLOCK();

    std::unordered_set<std::shared_ptr<Repeated>> merged_this_time;

    for (const auto& nh : m_graph->sorted()) {
        if (!m_graph->contains(nh)) {
            continue;
        }
        Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
        auto rep = group->repeated();

        GPtrSet repeating_groups;

        if (rep && rep->openForMerge() && merged_this_time.count(rep) == 0) {
            repeating_groups = getRepGroups(group);
        }

        if (!repeating_groups.empty()) {
            auto new_rep = tryGrowRepeatingGroups(repeating_groups);
            if (new_rep) {
                merged_this_time.insert(new_rep);
            }
        }
    }

    LOG_INFO("Number of groups after compiler pass: " << graphSize());
    LOG_INFO("DONE");
}

std::shared_ptr<Repeated> Snapshot::tryGrowRepeatingGroups(const GPtrSet& repeating_groups) {
    const auto& first_rep_group = *(repeating_groups.begin());
    // Those 3 should be the same for each group inside
    auto this_rep_tag = first_rep_group->repeated();
    const auto& this_avoided = first_rep_group->avoidedTargets();
    const auto& this_special = first_rep_group->specialTags();

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
            Group::GPtr prod_group = m_graph->meta(prod_nh).get<Group::GPtr>();
            if (prod_group->repeated() && !prod_group->hasCycle(group) && prod_group->repeated() != this_rep_tag &&
                prod_group->avoidedTargets() == this_avoided && prod_group->specialTags() == this_special) {
                auto meta_interconnect = group->metaInterconnect(prod_group);

                // FIXME: find a better way to reduce time complexity
                // Need to align interconnects in the same format via sort, so they could be compared later
                std::vector<MetaInterconnect> mic_sorted_key(meta_interconnect.begin(), meta_interconnect.end());
                std::sort(mic_sorted_key.begin(), mic_sorted_key.end());
                mics[mic_sorted_key].push_back({prod_group, group});
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

    return {};
}

std::shared_ptr<Repeated> Snapshot::tryMergeRepeating(const std::vector<Group::GPtr>& prods,
                                                      const std::vector<Group::GPtr>& conss) {
    if (prods.size() != conss.size()) {
        // FIXME: it's actually possible to merge under certain circumstances
        OPENVINO_THROW(
            "Online partitioning tried to merge repeated groups with different sizes of producers and consumers!");
    }

    if (conss.size() == 1) {
        return {};
    }

    std::unordered_set<Group::GPtr> prods_set;
    for (const auto& prod : prods) {
        prods_set.insert(prod);
    }

    if (prods_set.size() != conss.size()) {
        // Unintentionally this is also a check which prevents repeating producer/consumer
        // triangles to be merged. For a configuration like
        //
        //  A1     A2
        // .  .   .  .
        // B1 B2  B3 B4
        //
        // In this method we get [ A1, A1, A2, A2 ] as prods what is not very correct
        // but this check using std::set reverts it back to the proper [ A1, A2 ] form and the check fails
        return {};
    }

    for (const auto& cons : conss) {
        if (std::find(prods.begin(), prods.end(), cons) != prods.end()) {
            OPENVINO_THROW("Online partitioning tried to merge repeated groups which overlap!");
        }
    }

    if (prods.size() < m_ctx.keep_blocks) {
        // In some cases (specifically mixed precision) during MergeUniques() pass we could be left with
        // E.g. 10 repeated blocks with tag AAA and 2 repeated blocks with tag BBB
        // TryMergeRepeating() pass checks that producer and consumer have a different tag to be merged further.
        // Let's say in our example 10 AAA blocks are finalized and cannot be merged further due to above check.
        // However we will proceed to merge 3 BBB blocks with 3 AAA blocks since the tags are different.
        // This will create a new tag CCC for the merged blocks and the merge will continue until those 3 blocks
        // consume a large amount of legit AAA blocks.
        // Later in CleanUpUniques() pass those repeated blocks will be stripped off repeated tag due to the same check
        // in this "if". To prevent such cases where we would end up with small number of huge blocks this check was
        // introduced.
        return {};
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

std::unordered_map<std::shared_ptr<Repeated>, GPtrSet> Snapshot::repeating() const {
    std::unordered_map<std::shared_ptr<Repeated>, GPtrSet> repeating;
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

    afterUniques();

    LOG_INFO("Number of groups after compiler pass: " << graphSize());
    LOG_INFO("DONE");
}

void Snapshot::afterUniques() {
    LOG_INFO("Online partitioning: executing afterUniques pass...");
    LOG_BLOCK();

    for (const auto& nh : m_graph->sorted()) {
        Group::GPtr group = m_graph->meta(nh).get<Group::GPtr>();
        const auto& tag = group->isolatedTag();

        // Not expecting thousands of tags here, thus std::find on a vector
        if (!tag.empty() && std::find(m_ctx.nofolds.begin(), m_ctx.nofolds.end(), tag) != m_ctx.nofolds.end()) {
            group->noFold();
        }
    }

    LOG_INFO("DONE");
}

bool Snapshot::cleanUpUniquesImpl(const GPtrSet& gptrs) {
    for (const auto& gptr : gptrs) {
        if (!gptr->avoidedTargets().empty() || gptr->isNoFold()) {
            auto block_layer_size = (*(gptrs.begin()))->size();
            LOG_DEBUG("Keeping a repeated block of " << gptrs.size() << " groups with " << block_layer_size
                                                     << " layers - has AVOIDs");
            // Special case - keep it
            for (const auto& g : gptrs) {
                g->freeze();
            }
            return true;
        }
    }

    // Another special case, actually a workaround. Keep it
    // FIXME: slightly different from Ensemble since we don't check flops and keep it by size only
    auto block_layer_size = (*(gptrs.begin()))->size();
    if (gptrs.size() >= m_ctx.keep_blocks && block_layer_size >= m_ctx.keep_block_size) {
        LOG_DEBUG("Keeping a repeated block of " << gptrs.size() << " groups with " << block_layer_size << " layers.");
        for (const auto& g : gptrs) {
            g->freeze();
        }
        return true;
    }

    // Not good enough to keep
    for (const auto& gptr : gptrs) {
        gptr->setRepeated(nullptr);
    }

    LOG_DEBUG("Repeated block of " << gptrs.size() << " groups with " << block_layer_size << " layers is dropped.");

    return false;
}

void Snapshot::completeRepeating(const std::shared_ptr<Repeated>& reptag, const GPtrSet& gset) {
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

GPtrSet Snapshot::getRepGroups(const Group::GPtr& group) const {
    auto rep = group->repeated();

    GPtrSet repeating_groups;

    for (const auto& nh_other : m_graph->sorted()) {
        if (!m_graph->contains(nh_other)) {
            continue;
        }
        Group::GPtr group_other = m_graph->meta(nh_other).get<Group::GPtr>();
        auto rep_other = group_other->repeated();

        if (rep_other && !group_other->isFrozen() && (rep_other.get() == rep.get())) {
            repeating_groups.insert(group_other);
        }
    }

    return repeating_groups;
}

const OVNodeSet& Snapshot::getNodeProducers(const OVNodePtr& node) const {
    return ov::npuw::util::at::_(m_node_to_prod_cons).at(node).first;
}

const OVNodeSet& Snapshot::getNodeConsumers(const OVNodePtr& node) const {
    return ov::npuw::util::at::_(m_node_to_prod_cons).at(node).second;
}

// Updated within a group during fusion
const ov::npuw::online::detail::OVNodeToGroupMapPtr& Snapshot::getNodeToGroupMap() const {
    return m_node_to_gr;
}

std::shared_ptr<own::ade::Graph> Snapshot::getGraph() const {
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

void Snapshot::stripTag(const std::string& tag) {
    for (auto&& nh : m_graph->nodes()) {
        auto gptr = m_graph->meta(nh).get<Group::GPtr>();
        if (gptr->isolatedTag() == tag) {
            gptr->dontIsolate();
        }
    }
}
