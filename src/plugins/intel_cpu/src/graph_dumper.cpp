// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_dumper.h"

#include "dnnl_debug.h"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "utils/debug_capabilities.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {

void serializeToCout(const Graph &graph);
void serializeToXML(const Graph &graph, const std::string& path);

namespace {

std::map<std::string, std::string> extract_node_metadata(const NodePtr &node) {
    std::map<std::string, std::string> serialization_info;

    if (node->getType() == Type::Input && node->isConstant()) {
        // We need to separate Input and Const layers
        serialization_info[ov::exec_model_info::LAYER_TYPE] = "Const";
    } else {
        serialization_info[ov::exec_model_info::LAYER_TYPE] = NameFromType(node->getType());
    }

    // Original layers
    serialization_info[ov::exec_model_info::ORIGINAL_NAMES] = node->getOriginalLayers();

    // Implementation type name
    serialization_info[ov::exec_model_info::IMPL_TYPE] = node->getPrimitiveDescriptorType();

    std::string outputPrecisionsStr;
    if (!node->getChildEdges().empty()) {
        outputPrecisionsStr = node->getChildEdgeAt(0)->getMemory().getDesc().getPrecision().get_type_name();

        bool isAllEqual = true;
        for (size_t i = 1; i < node->getChildEdges().size(); i++) {
            if (node->getChildEdgeAt(i - 1)->getMemory().getDesc().getPrecision() != node->getChildEdgeAt(i)->getMemory().getDesc().getPrecision()) {
                isAllEqual = false;
                break;
            }
        }

        // If all output precisions are the same, we store the name only once
        if (!isAllEqual) {
            for (size_t i = 1; i < node->getChildEdges().size(); i++)
                outputPrecisionsStr += "," + std::string(node->getChildEdgeAt(i)->getMemory().getDesc().getPrecision().get_type_name());
        }
    } else {
        // Branch to correctly handle output nodes
        if (!node->getParentEdges().empty()) {
            outputPrecisionsStr = node->getParentEdgeAt(0)->getMemory().getDesc().getPrecision().get_type_name();
        }
    }
    serialization_info[ov::exec_model_info::OUTPUT_PRECISIONS] = outputPrecisionsStr;

    std::string outputLayoutsStr;
    auto outDescs = node->getSelectedPrimitiveDescriptor()->getConfig().outConfs;

    if (!outDescs.empty()) {
        outputLayoutsStr = outDescs[0].getMemDesc()->serializeFormat();

        bool isAllEqual = true;
        for (size_t i = 1; i < outDescs.size(); i++) {
            if (outDescs[i - 1].getMemDesc()->serializeFormat() != outDescs[i].getMemDesc()->serializeFormat()) {
                isAllEqual = false;
                break;
            }
        }

        // If all output layouts are the same, we store the name only once
        if (!isAllEqual) {
            for (size_t i = 1; i < outDescs.size(); i++) {
                outputLayoutsStr += "," + outDescs[i].getMemDesc()->serializeFormat();
            }
        }
    } else {
        outputLayoutsStr = dnnl::utils::fmt2str(dnnl::memory::format_tag::undef);
    }
    serialization_info[ov::exec_model_info::OUTPUT_LAYOUTS] = outputLayoutsStr;

    // Performance
    if (node->PerfCounter().avg() != 0) {
        serialization_info[ov::exec_model_info::PERF_COUNTER] = std::to_string(node->PerfCounter().avg());
    } else {
        serialization_info[ov::exec_model_info::PERF_COUNTER] = "not_executed";  // it means it was not calculated yet
    }

    serialization_info[ov::exec_model_info::EXECUTION_ORDER] = std::to_string(node->getExecIndex());

    serialization_info[ov::exec_model_info::RUNTIME_PRECISION] = node->getRuntimePrecision().get_type_name();

    return serialization_info;
}

}  // namespace

std::shared_ptr<ov::Model> dump_graph_as_ie_ngraph_net(const Graph &graph) {
    std::map<NodePtr, std::shared_ptr<ov::Node> > node2layer;

    ov::ResultVector results;
    ov::ParameterVector params;
    ov::NodeVector to_hold;

    std::map<std::size_t, std::shared_ptr<op::v0::Parameter>> paramsMap;
    std::map<std::size_t, std::shared_ptr<ov::op::v0::Result>> resultsMap;

    auto get_inputs = [&] (const NodePtr & node) {
        auto pr_edges = node->getParentEdges();
        ov::OutputVector inputs(pr_edges.size());

        for (size_t i = 0; i < pr_edges.size(); i++) {
            auto edge = node->getParentEdgeAt(i);
            int pr_port = edge->getInputNum();
            int ch_port = edge->getOutputNum();
            auto pr_node = edge->getParent();

            OPENVINO_ASSERT(node2layer.count(pr_node) == 1);
            auto pr = node2layer[pr_node];

            inputs[ch_port] = pr->output(pr_port);
        }

        return inputs;
    };

    auto create_ngraph_node = [&](const NodePtr &node) {
        bool is_input = false, is_output = false, should_be_hold = false;
        size_t input_index = -1, output_index = -1;
        for (auto && kvp : graph.inputNodesMap) {
            if (kvp.second == node) {
                is_input = true;
                input_index = kvp.first;
                break;
            }
        }

        for (auto && kvp : graph.outputNodesMap) {
            if (kvp.second == node) {
                is_output = true;
                output_index = kvp.first;
                break;
            }
        }

        if (!is_output && node->getChildEdges().empty()) {
            // The node has no consumer and is not an output.
            // Should be hold in other irregular way.
            should_be_hold = true;
        }

        auto meta_data = extract_node_metadata(node);
        std::shared_ptr<ov::Node> return_node;
        if (is_input) {
            auto& desc = node->getChildEdgeAt(0)->getMemory().getDesc();
            auto param = std::make_shared<ov::op::v0::Parameter>(desc.getPrecision(), desc.getShape().toPartialShape());
            return_node = param;
            paramsMap[input_index] = param;
        } else if (is_output) {
            auto result = std::make_shared<ov::op::v0::Result>(get_inputs(node).back());
            resultsMap[output_index] = result;
            return_node = result;
        } else {
            return_node = std::make_shared<ov::exec_model_info::ExecutionNode>(
                get_inputs(node), node->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size());

            for (size_t port = 0; port < return_node->get_output_size(); ++port) {
                auto& desc = node->getChildEdgeAt(port)->getMemory().getDesc();
                return_node->set_output_type(port, desc.getPrecision(), desc.getShape().toPartialShape());
            }
        }

        if (should_be_hold) {
            to_hold.push_back(return_node);
        }

        for (auto && kvp : meta_data)
            return_node->get_rt_info()[kvp.first] = kvp.second;
        return_node->set_friendly_name(node->getName());

        return return_node;
    };

    ov::NodeVector nodes;
    nodes.reserve(graph.graphNodes.size());
    for (auto &node : graph.graphNodes) {  // important: graph.graphNodes are in topological order
        nodes.emplace_back(create_ngraph_node(node));
        node2layer[node] = nodes.back();
    }

    for (auto && kvp : paramsMap)
        params.push_back(kvp.second);
    for (auto && kvp : resultsMap)
        results.push_back(kvp.second);

    auto holder = !results.empty() ? results[0] : std::make_shared<ov::op::v0::Result>();
    for (auto &node : to_hold) {
        holder->add_control_dependency(node);
    }

    return std::make_shared<ov::Model>(results, params, graph._name);
}

#ifdef CPU_DEBUG_CAPS
void serialize(const Graph &graph) {
    const std::string& path = graph.getConfig().debugCaps.execGraphPath;

    if (path.empty())
        return;

    if (path == "cout") {
        serializeToCout(graph);
    } else if (!path.compare(path.size() - 4, 4, ".xml")) {
        static int g_idx = 0;
        std::string xmlPath = std::string(path, 0, path.size() - 4) + "_" + std::to_string(g_idx++) + ".xml";
        serializeToXML(graph, xmlPath);
    } else {
        OPENVINO_THROW("Unknown serialize format. Should be either 'cout' or '*.xml'. Got ", path);
    }
}

void serializeToXML(const Graph &graph, const std::string& path) {
    if (path.empty())
        return;

    std::string binPath;
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(path,
                                               binPath,
                                               ov::pass::Serialize::Version::IR_V10);
    manager.run_passes(graph.dump());
}

void serializeToCout(const Graph &graph) {
    for (const auto& node : graph.GetNodes()) {
        std::cout << "name: " << node->getName() << " [ ";
        auto nodeDesc = node->getSelectedPrimitiveDescriptor();
        if (nodeDesc) {
            auto& inConfs = nodeDesc->getConfig().inConfs;
            if (!inConfs.empty()) {
                std::cout << "in: " << inConfs.front().getMemDesc()->getPrecision().get_type_name()
                          << "/l=" << inConfs.front().getMemDesc()->serializeFormat()
                          << "; ";
            }
            auto& outConfs = nodeDesc->getConfig().outConfs;
            if (!outConfs.empty()) {
                std::cout << "out: " << outConfs.front().getMemDesc()->getPrecision().get_type_name()
                          << "/l=" << outConfs.front().getMemDesc()->serializeFormat();
            }
        }
        std::cout << " ]"  << std::endl;
    }
}

void summary_perf(const Graph &graph) {
    if (!graph.getGraphContext()) {
        return;
    }
    const std::string& summaryPerf = graph.getConfig().debugCaps.summaryPerf;

    if (summaryPerf.empty() || !std::stoi(summaryPerf))
        return;

    std::map<std::string, double> perf_by_type;
    std::map<NodePtr, double> perf_by_node;
    double total_avg = 0;
    uint64_t total = 0;
    for (auto &node : graph.GetNodes()) {  // important: graph.graphNodes are in topological order
        double avg = node->PerfCounter().avg();
        auto type = node->getTypeStr() + "_" + node->getPrimitiveDescriptorType();
        auto name = node->getName();

        total += node->PerfCounter().count() * avg;
        total_avg += avg;

        if (perf_by_type.count(type))
            perf_by_type[type] += avg;
        else
            perf_by_type[type] = avg;

        if (perf_by_node.count(node))
            perf_by_node[node] += avg;
        else
            perf_by_node[node] = avg;
    }

    if (total_avg < 1) return;

    std::cout << "======= ENABLE_DEBUG_CAPS:OV_CPU_SUMMARY_PERF ======" << std::endl;
    std::cout << "Summary of " << graph.GetName() << " @" << std::hash<uint64_t>{}(reinterpret_cast<uint64_t>(&graph)) << std::endl;
    std::cout << "     Total(us): " << (uint64_t)(total) << std::endl;
    std::cout << " Total_avg(us): " << (uint64_t)(total_avg) << std::endl;
    {
        std::cout << " perf_by_type:" << std::endl;
        std::vector<std::pair<std::string, double> > A;
        for (auto& it : perf_by_type)
            A.push_back(it);
        sort(A.begin(), A.end(),
             [](std::pair<std::string, double>& a,
                std::pair<std::string, double>& b){
                 return a.second > b.second;
             });

        for (auto& it : A) {
            std::stringstream ss;
            int percentage = static_cast<int>(it.second*100/total_avg);
            if (percentage == 0) break;
            ss << std::setw(10) << std::right << percentage << " % :  " << std::setw(8) << std::right << it.second << "(us)  " << it.first << std::endl;
            std::cout << ss.str();
        }
    }
    {
        std::cout << " perf_by_node:" << std::endl;
        std::vector<std::pair<NodePtr, double> > A;
        for (auto& it : perf_by_node)
            A.push_back(it);
        sort(A.begin(), A.end(),
            [](std::pair<NodePtr, double>& a,
                std::pair<NodePtr, double>& b){
            return a.second > b.second;
        });

        for (auto& it : A) {
            std::stringstream ss;
            auto percentage = it.second*100/total_avg;
            auto node = it.first;
            if (node->PerfCounter().count() == 0) continue;
            if (node->PerfCounter().avg() < 1) continue;
            ss << std::setw(10) << std::right << std::fixed << std::setprecision(2) << percentage << " %  "
               << std::setw(8) << std::right  << node->PerfCounter().avg() << "(us)x" << node->PerfCounter().count()
               << " #" << node->getExecIndex()
               << " " << node->getName()
               << " " << node->getTypeStr() + "_" + node->getPrimitiveDescriptorType() << std::endl;
            std::cout << ss.str();
        }
    }
}

#endif
}   // namespace intel_cpu
}   // namespace ov
