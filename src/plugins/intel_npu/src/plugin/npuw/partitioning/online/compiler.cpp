// Copyright (C) 2024 Intel Corporationov::npuw::
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler.hpp"

#include <iostream>
#include <sstream>

#include "../../logging.hpp"
#include "../../util.hpp"
#include "group.hpp"
#include "intel_npu/al/config/config.hpp"
#include "intel_npu/al/config/npuw.hpp"
#include "pugixml.hpp"
#include "snapshot.hpp"

namespace ov {
namespace npuw {
namespace online {

namespace detail {
// For missing declaration warning
size_t getMinGraphSize(const std::shared_ptr<ov::Model>& model, ::intel_npu::Config& cfg);
std::vector<Avoid> getAvoids(const std::shared_ptr<ov::Model>& model, ::intel_npu::Config& cfg);
void dump_partitioning(const ov::npuw::Ensemble& ens, const std::shared_ptr<ov::Model>& model, const std::string& to);

size_t getMinGraphSize(const std::shared_ptr<ov::Model>& model, ::intel_npu::Config& cfg) {
    std::size_t min_size = cfg.get<::intel_npu::NPUW_ONLINE_MIN_SIZE>();

    // Sanity check
    if (min_size < 10) {
        LOG_WARN("Minimum possible partitioning size is too small: " << min_size << ", using a default value of 10.");
        min_size = 10;
    }

    LOG_INFO("Online partitioning will continue until there are " << min_size << " or less subgraphs.");

    return min_size;
}

std::vector<Avoid> getAvoids(const std::shared_ptr<ov::Model>& model, ::intel_npu::Config& cfg) {
    std::vector<Avoid> avoids;

    std::string avoids_opt = cfg.getString<::intel_npu::NPUW_ONLINE_AVOID>();
    if (avoids_opt.empty()) {
        LOG_WARN(::intel_npu::NPUW_ONLINE_AVOID().key()
                 << " property is not set! NPU device will be prioritized for every subgraph.");
        return {};
    }

    std::string s = std::move(avoids_opt);

    size_t pos = 0;
    size_t start = 0;
    std::string token;

    while ((pos = s.find(',', start)) != std::string::npos) {
        token = s.substr(start, pos - start);
        auto avoid_opt = util::parseAvoid(token);
        // Check that parsing was a success
        if (avoid_opt) {
            avoids.push_back(*avoid_opt);
        }
        start = pos + 1;
    }

    // Parse the tail
    auto avoid_opt = util::parseAvoid(s.substr(start, s.size() - start));
    // Check that parsing was a success
    if (avoid_opt) {
        avoids.push_back(*avoid_opt);
    }

    if (!avoids.empty()) {
        LOG_INFO("Online partitioning will avoid running subgraphs containing specified patterns on their respective "
                 "devices.");
    } else {
        LOG_WARN("Incorect pattern in OPENVINO_NPUW_AVOID!"
                 << " Please, follow the example: Op:Select/NPU,P:RMSNorm/NPU."
                 << " No avoid rules will be taken into account during execution!");
    }

    return avoids;
}

void dump_partitioning(const ov::npuw::Ensemble& ens, const std::shared_ptr<ov::Model>& model, const std::string& to) {
    pugi::xml_document doc;

    pugi::xml_node node = doc.append_child("ensemble");
    node.append_attribute("gflops") = std::to_string(ens.gflops).data();

    pugi::xml_node part = node.append_child("partitioning");
    pugi::xml_node rep;
    if (!ens.repeated.empty()) {
        rep = node.append_child("repeated");
    }

    size_t gid = 0;
    for (const auto& group : ens.groups) {
        pugi::xml_node gr = part.append_child("group");
        gr.append_attribute("id") = std::to_string(gid++).data();
        gr.append_attribute("gflops") = std::to_string(group.gflops).data();
        if (!group.repeated_id.empty()) {
            gr.append_attribute("repeated") = group.repeated_id.data();
        }
        if (!group.avoid_list.empty()) {
            gr.append_attribute("avoid") = group.avoid_list.data();
        }

        // Note: Ensemble also add "id" attribute but it's not used by the plugin
        for (const auto& input : group.input_layers) {
            pugi::xml_node in = gr.append_child("input");
            in.append_attribute("name") = input.data();
        }
        for (const auto& output : group.output_layers) {
            pugi::xml_node out = gr.append_child("output");
            out.append_attribute("name") = output.data();
        }
        for (const auto& layer : group.all_layers) {
            pugi::xml_node l = gr.append_child("layer");
            l.append_attribute("name") = layer.data();
        }
    }

    for (const auto& repeated : ens.repeated) {
        pugi::xml_node bl = rep.append_child("block");
        bl.append_attribute("id") = repeated.first.data();

        for (const auto& match : repeated.second.matches) {
            // Note: Ensemble also add "like" attribute but it's not used by the plugin
            pugi::xml_node mt = bl.append_child("match");

            for (const auto& layer : match) {
                pugi::xml_node l = mt.append_child("layer");
                l.append_attribute("name") = layer.data();
            }
        }
    }

    doc.save_file(to.data());
}
}  // namespace detail

// Interface to get online partitioning from the model
class Compiler {
    enum class Pipeline {
        NONE,  // Do nothing, partitioning will be empty
        INIT,  // Initialize only. The hardest mode, every group has just 1 layer inside
        JUST,  // "justParitioning" - combination of LHF + Remnants
        REP,   // Repeated blocks pipeline - combination of repeatedBlocks and Remnants - default configuration
    };

    Pipeline currentPipeline(const std::shared_ptr<ov::Model>& model) {
        std::string pipeline_opt = m_cfg.getString<::intel_npu::NPUW_ONLINE_PIPELINE>();
        if (pipeline_opt == "NONE") {
            return Pipeline::NONE;
        } else if (pipeline_opt == "INIT") {
            return Pipeline::INIT;
        } else if (pipeline_opt == "JUST") {
            return Pipeline::JUST;
        } else if (pipeline_opt == "REP") {
            return Pipeline::REP;
        } else {
            LOG_WARN("Unknown partitioning compiler pipeline " << pipeline_opt << ", switching to REP");
            return Pipeline::REP;
        }
    }

    // Compiler pipelines
    void init() {
        LOG_INFO("Online partitioning: compiling initial pipeline...");
        LOG_BLOCK();

        // Literally there's nothing to do for now.

        LOG_INFO("Done");
    }

    void just() {
        LOG_INFO("Online partitioning: compiling fixed pipeline...");
        LOG_BLOCK();

        m_snapshot->repeat([&] {
            m_snapshot->collectLHF();
        });
        m_snapshot->repeat([&] {
            m_snapshot->fuseRemnants();
        });

        LOG_INFO("Done");
    }

    void rep() {
        LOG_INFO("Online partitioning: compiling repeated blocks pipeline...");
        LOG_BLOCK();

        m_snapshot->earlyAvoids();
        m_snapshot->repeatedBlocks();
        m_snapshot->repeat([&] {
            m_snapshot->fuseRemnantsExtended();
        });

        LOG_INFO("Done");
    }

public:
    explicit Compiler(const std::shared_ptr<ov::Model>& model, ::intel_npu::Config& cfg)
        : m_model(model),
          m_snapshot(std::make_shared<Snapshot>(model)),
          m_cfg(cfg) {
        if (currentPipeline(model) == Pipeline::NONE) {
            LOG_INFO("NPUW_ONLINE_PIPELINE is set to NONE. No online partitioning is done.");
            return;
        }

        // Parse OV Model into internal data structures. After this
        // stage each layer = it's own group (excluding Parameters,
        // Results, Constants and Converts).
        LOG_INFO("Online partitioning: building initial graph...");
        m_snapshot->buildGraph();

        PassContext ctx;
        ctx.min_graph_size = detail::getMinGraphSize(model, m_cfg);
        ctx.avoids = detail::getAvoids(model, m_cfg);

        m_snapshot->setCtx(ctx);

        switch (currentPipeline(model)) {
        case Pipeline::NONE:
            break;  // unreachable
        case Pipeline::INIT:
            init();
            break;
        case Pipeline::JUST:
            just();
            break;
        case Pipeline::REP:
            rep();
            break;
        }

        LOG_DEBUG("Online partitioning: group sizes after compilation:");
        auto graph = m_snapshot->getGraph();
        for (const auto& nh : graph->sorted()) {
            LOG_BLOCK();
            Group::GPtr group = graph->meta(nh).get<Group::GPtr>();
            LOG_DEBUG("Group " << group->getId() << ", size " << group->size());
        }

        LOG_INFO("Done");
    }

    ov::npuw::Ensemble getPartitioning() {
        LOG_INFO("Online partitioning: converting to plugin-compatible format...");
        LOG_BLOCK();

        ov::npuw::Ensemble ens;
        ens.gflops = 1.;  // FIXME: calculate proper flops

        auto graph = m_snapshot->getGraph();
        // Iterate in topological order
        for (const auto& nh : graph->sorted()) {
            auto gptr = graph->meta(nh).get<ov::npuw::online::Group::GPtr>();
            ens.groups.emplace_back(gptr->toGroup());
        }

        std::map<std::string, ov::npuw::RepeatedBlock> repeated;
        for (const auto& reptag_and_matches : m_snapshot->getMatches()) {
            LOG_BLOCK();
            ov::npuw::RepeatedBlock block;
            block.matches = reptag_and_matches.second;  // Other fields are filled inside the plugin
            repeated.insert({reptag_and_matches.first, block});
            LOG_INFO("Got " << block.matches.at(0).size() << " repeated blocks of size " << block.matches.size());
        }
        ens.repeated = std::move(repeated);

        std::string dump_plan_path = m_cfg.get<::intel_npu::NPUW_ONLINE_DUMP_PLAN>();
        if (!dump_plan_path.empty()) {
            detail::dump_partitioning(ens, m_model, dump_plan_path);
            LOG_INFO("Dumped online partitioning to " << dump_plan_path << ".");
        }

        LOG_INFO("DONE.");

        return ens;
    }

private:
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<Snapshot> m_snapshot;
    ::intel_npu::Config& m_cfg;
};

}  // namespace online
}  // namespace npuw
}  // namespace ov

// TODO: decouple configuration for partitioning from the plugin's cfg.
ov::npuw::Ensemble ov::npuw::online::buildPartitioning(const std::shared_ptr<ov::Model>& model,
                                                       ::intel_npu::Config& cfg) {
    // Creates compiler and runs partitioning algorithm.
    ov::npuw::online::Compiler partitioner(model, cfg);

    // Convert groups formed after passes into plugin-compatible data structure.
    return partitioner.getPartitioning();
}
