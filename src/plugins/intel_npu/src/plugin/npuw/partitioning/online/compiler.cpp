// Copyright (C) 2024 Intel Corporationov::npuw::
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler.hpp"

#include <iostream>
#include <sstream>

#include "../../logging.hpp"
#include "../../util.hpp"
#include "group.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "pugixml.hpp"
#include "snapshot.hpp"

namespace ov {
namespace npuw {
namespace online {

namespace detail {

namespace {
static const std::map<std::string, std::string> ISOL_PRESETS = {{"COMPUTE",
                                                                 "P:DQMatMulGQu4/compute,P:DQMatMulCWu4/compute,"
                                                                 "P:DQMatMulGQi4/compute,P:DQMatMulCWi4/compute,"
                                                                 "P:DQMatMulConv/compute,"
                                                                 "P:VocabMatMul/compute,"
                                                                 "P:RMSNorm/compute,P:RMSNorm2/compute,"
                                                                 "P:VariadicSplit/compute"},
                                                                {"FAKE", "P:FakeConvert/fake,P:FakeQuantize/fake"}};
}

// For missing declaration warning
// FIXME: Instead, one should use namespace{}
size_t getMinGraphSize(::intel_npu::Config& cfg);
size_t getMinRepBlocks(::intel_npu::Config& cfg);
size_t getMinRepBlockSize(::intel_npu::Config& cfg);
std::vector<Avoid> getAvoids(::intel_npu::Config& cfg);
std::vector<Isolate> getIsolates(::intel_npu::Config& cfg);
std::vector<Isolate> getIsolates(const std::string& isolates_unparsed);
std::vector<std::string> getNoFolds(::intel_npu::Config& cfg);
std::vector<std::string> getNoFolds(const std::string& nofolds_unparsed);
// Set default predefined values for COMPUTE pipeline
void dump_partitioning(const ov::npuw::Ensemble& ens, const std::string& to);

size_t getMinGraphSize(::intel_npu::Config& cfg) {
    std::size_t min_size = cfg.get<::intel_npu::NPUW_ONLINE_MIN_SIZE>();

    // Sanity check
    if (min_size < 10) {
        LOG_WARN("Minimum possible partitioning size is too small: " << min_size << ", using a default value of 10.");
        min_size = 10;
    }

    LOG_INFO("Online partitioning will continue until there are " << min_size << " or less subgraphs.");

    return min_size;
}

size_t getMinRepBlocks(::intel_npu::Config& cfg) {
    std::size_t min_size = cfg.get<::intel_npu::NPUW_ONLINE_KEEP_BLOCKS>();

    return min_size;
}

size_t getMinRepBlockSize(::intel_npu::Config& cfg) {
    std::size_t min_size = cfg.get<::intel_npu::NPUW_ONLINE_KEEP_BLOCK_SIZE>();

    return min_size;
}

std::vector<Avoid> getAvoids(::intel_npu::Config& cfg) {
    std::vector<Avoid> avoids;

    std::string avoids_opt = cfg.getString<::intel_npu::NPUW_ONLINE_AVOID>();
    if (avoids_opt.empty()) {
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

std::vector<Isolate> getIsolates(::intel_npu::Config& cfg) {
    return getIsolates(cfg.getString<::intel_npu::NPUW_ONLINE_ISOLATE>());
}

std::vector<Isolate> getIsolates(const std::string& isolates_unparsed) {
    if (isolates_unparsed.empty()) {
        return {};
    }

    std::vector<Isolate> isolates;
    std::string s = isolates_unparsed;

    auto preset_iter = ISOL_PRESETS.find(s);
    if (preset_iter != ISOL_PRESETS.end()) {
        s = preset_iter->second;
    }

    size_t pos = 0;
    size_t start = 0;
    std::string token;

    while ((pos = s.find(',', start)) != std::string::npos) {
        token = s.substr(start, pos - start);
        auto isolate_opt = util::parseIsolate(token);
        // Check that parsing was a success
        if (isolate_opt) {
            isolates.push_back(*isolate_opt);
        }
        start = pos + 1;
    }

    // Parse the tail
    auto isolate_opt = util::parseIsolate(s.substr(start, s.size() - start));
    // Check that parsing was a success
    if (isolate_opt) {
        isolates.push_back(*isolate_opt);
    }

    if (!isolates.empty()) {
        LOG_INFO("Online partitioning will isolate subgraphs containing specified patterns.");
    } else {
        LOG_WARN("Incorect pattern in NPUW_ONLINE_ISOLATE! No isolate rules will be taken into account during "
                 "partitioning!");
    }

    return isolates;
}

std::vector<std::string> getNoFolds(::intel_npu::Config& cfg) {
    return getNoFolds(cfg.getString<::intel_npu::NPUW_ONLINE_NO_FOLD>());
}

std::vector<std::string> getNoFolds(const std::string& nofolds_unparsed) {
    if (nofolds_unparsed.empty()) {
        return {};
    }

    std::vector<std::string> nofolds;
    std::string s = std::move(nofolds_unparsed);

    size_t pos = 0;
    size_t start = 0;
    std::string token;

    while ((pos = s.find(',', start)) != std::string::npos) {
        token = s.substr(start, pos - start);
        if (!token.empty()) {
            nofolds.push_back(token);
        }
        start = pos + 1;
    }

    // Parse the tail
    std::string tail = s.substr(start, s.size() - start);
    if (!tail.empty()) {
        nofolds.push_back(tail);
    }

    if (!nofolds.empty()) {
        LOG_INFO("Online partitioning will mark specified tags as non-foldable.");
    } else {
        LOG_WARN("Incorect pattern in NPUW_ONLINE_NO_FOLD!"
                 << " Please, follow the example: "
                 << "compute,compute2. "
                 << "No non-fold rules will be taken into account during partitioning!");
    }

    return nofolds;
}

void dump_partitioning(const ov::npuw::Ensemble& ens, const std::string& to) {
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
        if (!group.tag.empty()) {
            gr.append_attribute("tag") = group.tag.data();
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
        NONE,     // Partitioning will consist of a single group with all the Ops
        INIT,     // Initialize only. The hardest mode, every group has just 1 layer inside
        JUST,     // "justParitioning" - combination of LHF + Remnants
        REP,      // Repeated blocks pipeline - combination of repeatedBlocks and Remnants
        REG,      // Regularized repeated blocks pipeline - same as REP, but with some strong hints first
        COMPUTE,  // Separates non-foldable compute subgraphs from the model based on predefined rules + REP
        SPATIAL   // Similar to COMPUTE but allows folding
    };

    template <class C>
    void warn_unused() {
        const auto& val = m_cfg.get<C>();
        if (val != C::defaultValue()) {
            LOG_WARN("User-specified configuration {" << C::key() << " : " << val
                                                      << "} is ignored in the current pipeline "
                                                      << m_cfg.get<::intel_npu::NPUW_ONLINE_PIPELINE>());
        }
    }

    Pipeline currentPipeline() {
        std::string pipeline_opt = m_cfg.getString<::intel_npu::NPUW_ONLINE_PIPELINE>();
        if (pipeline_opt == "NONE") {
            return Pipeline::NONE;
        } else if (pipeline_opt == "INIT") {
            return Pipeline::INIT;
        } else if (pipeline_opt == "JUST") {
            return Pipeline::JUST;
        } else if (pipeline_opt == "REP") {
            return Pipeline::REP;
        } else if (pipeline_opt == "REG") {
            return Pipeline::REG;
        } else if (pipeline_opt == "COMPUTE") {
            return Pipeline::COMPUTE;
        } else if (pipeline_opt == "SPATIAL") {
            return Pipeline::SPATIAL;
        } else {
            LOG_WARN("Unknown partitioning compiler pipeline " << pipeline_opt << ", switching to REP");
            return Pipeline::REP;
        }
    }

    std::vector<Isolate> getAllIsolates() {
        auto isolates = detail::getIsolates(detail::ISOL_PRESETS.at("COMPUTE"));
        for (const auto& isol : detail::getIsolates(detail::ISOL_PRESETS.at("FAKE"))) {
            isolates.push_back(isol);
        }
        return isolates;
    }

    // Compiler pipelines
    void none() {
        LOG_INFO("Online partitioning: compiling single group pipeline...");
        LOG_BLOCK();

        m_snapshot->singleGroup();

        LOG_INFO("Done");
    }

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
        m_snapshot->earlyRegroup();
        m_snapshot->repeatedBlocks();
        m_snapshot->repeat([&] {
            m_snapshot->fuseRemnantsExtended();
        });

        LOG_INFO("Done");
    }

    void reg() {
        LOG_INFO("Online partitioning: compiling regularized repeated blocks pipeline...");
        LOG_BLOCK();

        m_snapshot->earlyAvoids();
        m_snapshot->earlyRegroup();
        m_snapshot->repeatedBlocks([&]() {
            // This callback is called when repeatingBlocks algorithm thinks it is done
            m_snapshot->stripTag("compute");
        });
        m_snapshot->repeat([&] {
            m_snapshot->fuseRemnantsExtended();
        });

        LOG_INFO("Done");
    }

public:
    Compiler(const std::shared_ptr<ov::Model>& model, ::intel_npu::Config& cfg)
        : m_model(model),
          m_snapshot(std::make_shared<Snapshot>(model)),
          m_cfg(cfg) {
        if (currentPipeline() == Pipeline::NONE) {
            none();
            return;
        }

        // Parse OV Model into internal data structures. After this
        // stage each layer = it's own group (excluding Parameters,
        // Results, Constants and Converts).
        LOG_INFO("Online partitioning: building initial graph...");
        m_snapshot->buildGraph();

        PassContext ctx;
        ctx.min_graph_size = detail::getMinGraphSize(m_cfg);
        ctx.keep_blocks = detail::getMinRepBlocks(m_cfg);
        ctx.keep_block_size = detail::getMinRepBlockSize(m_cfg);
        ctx.avoids = detail::getAvoids(m_cfg);
        ctx.isolates = detail::getIsolates(m_cfg);
        ctx.nofolds = detail::getNoFolds(m_cfg);

        m_snapshot->setCtx(ctx);

        switch (currentPipeline()) {
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
        case Pipeline::REG:
            warn_unused<::intel_npu::NPUW_ONLINE_ISOLATE>();

            // Only get isolates here.
            // NB: We ignore NO_FOLD everywhere except pipeline COMPUTE - this needs
            // to be aligned in the future
            ctx.isolates = getAllIsolates();
            m_snapshot->setCtx(ctx);
            reg();
            break;
        case Pipeline::COMPUTE:
            warn_unused<::intel_npu::NPUW_ONLINE_ISOLATE>();
            warn_unused<::intel_npu::NPUW_ONLINE_NO_FOLD>();

            // Manually set predefined isolates and nofolds then do rep() pipeline
            // FIXME: initialize via a dedicated function instead of parsing
            ctx.isolates = getAllIsolates();
            ctx.nofolds = detail::getNoFolds("compute");
            m_snapshot->setCtx(ctx);
            rep();
            break;
        case Pipeline::SPATIAL:
            warn_unused<::intel_npu::NPUW_ONLINE_ISOLATE>();
            m_cfg.update(::intel_npu::Config::ConfigMap{{std::string(::intel_npu::NPUW_SPATIAL::key()), "YES"}});

            // Manually set predefined isolates and nofolds then do rep() pipeline
            // FIXME: initialize via a dedicated function instead of parsing
            ctx.isolates = getAllIsolates();
            m_snapshot->setCtx(ctx);
            rep();
            break;
        }

        LOG_DEBUG("Online partitioning: group sizes after compilation:");
        auto graph = m_snapshot->getGraph();
        for (const auto& nh : graph->sorted()) {
            LOG_BLOCK();
            Group::GPtr group = graph->meta(nh).get<Group::GPtr>();
            LOG_DEBUG("Group " << group->getId() << ", size " << group->size() << ", tag " << group->specialTags());
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
            detail::dump_partitioning(ens, dump_plan_path);
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
