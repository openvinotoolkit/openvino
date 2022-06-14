// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_dumper.h"

#include "utils/debug_capabilities.h"
#include <ie_ngraph_utils.hpp>
#include "exec_graph_info.hpp"
#include "ie_common.h"
#include <dnnl_debug.h>
#include <ngraph/variant.hpp>
#include "ngraph/ngraph.hpp"
#include <ngraph/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include <vector>
#include <string>
#include <memory>
#include <map>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

void serializeToCout(const Graph &graph);
void serializeToXML(const Graph &graph, const std::string& path);

namespace {

std::map<std::string, std::string> extract_node_metadata(const NodePtr &node) {
    std::map<std::string, std::string> serialization_info;

    if (node->getType() == Type::Input && node->isConstant()) {
        // We need to separate Input and Const layers
        serialization_info[ExecGraphInfoSerialization::LAYER_TYPE] = "Const";
    } else if (node->getType() == Type::Generic) {
        // Path to print actual name for extension layers
        serialization_info[ExecGraphInfoSerialization::LAYER_TYPE] = node->getTypeStr();
    } else {
        serialization_info[ExecGraphInfoSerialization::LAYER_TYPE] = NameFromType(node->getType());
    }

    serialization_info["isConstant"] = node->isConstant() ? "True" : "False";

    // Original layers
    serialization_info[ExecGraphInfoSerialization::ORIGINAL_NAMES] = node->getOriginalLayers();

    // Implementation type name
    serialization_info[ExecGraphInfoSerialization::IMPL_TYPE] = node->getPrimitiveDescriptorType();

    std::string outputPrecisionsStr;
    if (!node->getChildEdges().empty()) {
        outputPrecisionsStr = node->getChildEdgeAt(0)->getMemory().getDesc().getPrecision().name();

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
                outputPrecisionsStr += "," + std::string(node->getChildEdgeAt(i)->getMemory().getDesc().getPrecision().name());
        }
    } else {
        // Branch to correctly handle output nodes
        if (!node->getParentEdges().empty()) {
            outputPrecisionsStr = node->getParentEdgeAt(0)->getMemory().getDesc().getPrecision().name();
        }
    }
    serialization_info[ExecGraphInfoSerialization::OUTPUT_PRECISIONS] = outputPrecisionsStr;

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
    serialization_info[ExecGraphInfoSerialization::OUTPUT_LAYOUTS] = outputLayoutsStr;

    // Performance
    if (node->PerfCounter().avg() != 0) {
        serialization_info[ExecGraphInfoSerialization::PERF_COUNTER] = std::to_string(node->PerfCounter().avg());
    } else {
        serialization_info[ExecGraphInfoSerialization::PERF_COUNTER] = "not_executed";  // it means it was not calculated yet
    }

    serialization_info[ExecGraphInfoSerialization::EXECUTION_ORDER] = std::to_string(node->getExecIndex());

    serialization_info[ExecGraphInfoSerialization::RUNTIME_PRECISION] = node->getRuntimePrecision().name();

    return serialization_info;
}

std::string toString(const MemoryDesc* p) {
    std::stringstream ss;
    ss << p->getShape().toString() << " ";
    ss << p->getPrecision().name() << " ";

    auto pdesc = dynamic_cast<const BlockedMemoryDesc*>(p);
    if (!pdesc)
        return ss.str();
    auto& xShape = pdesc->getShape();
    auto& xBlockDims = pdesc->getBlockDims();
    auto& xOffsetPaddingToData = pdesc->getOffsetPaddingToData();
    auto& xStrides = pdesc->getStrides();

    ss << pdesc->serializeFormat() << " ";

    if (Shape(xBlockDims) != xShape)
        ss << "BlockDims=" << MemoryDescUtils::dims2str(xBlockDims) << " ";

    if (xStrides.size() != xBlockDims.size()) {
        ss << "Strides=" << MemoryDescUtils::dims2str(pdesc->getStrides()) << " ";
    } else {
        size_t expected = 1;
        for (int i = xStrides.size() - 1; i >= 0; i--) {
            if (xStrides[i] != expected) {
                ss << "Strides=" << MemoryDescUtils::dims2str(pdesc->getStrides()) << " ";
                break;
            }
            expected *= xBlockDims[i];
        }
    }

    if (!std::all_of(xOffsetPaddingToData.begin(), xOffsetPaddingToData.end(), [](size_t s) {
            return s == 0;
        }))
        ss << "OffsetPaddingToData=" << MemoryDescUtils::dims2str(xOffsetPaddingToData) << " ";

    if (pdesc->getOffsetPadding())
        ss << "OffsetPadding=" << MemoryDescUtils::dim2str(pdesc->getOffsetPadding()) << " ";

    return ss.str();
}

std::string toString(const PortConfig* p) {
    std::stringstream ss;
    ss << toString(p->getMemDesc().get()) << " constant=" << p->constant() << " inPlace=" << p->inPlace();
    return ss.str();
}

std::string toString(const NodeConfig* p) {
    std::stringstream ss;
    ss << "  inConfs:" << std::endl;
    for (auto c : p->inConfs)
        ss << "    " << toString(&c) << std::endl;
    ss << "  outConfs:" << std::endl;
    for (auto c : p->outConfs)
        ss << "    " << toString(&c) << std::endl;
    return ss.str();
}

}  // namespace

std::shared_ptr<ngraph::Function> dump_graph_as_ie_ngraph_net(const Graph &graph) {
    std::map<NodePtr, std::shared_ptr<ngraph::Node> > node2layer;

    ngraph::ResultVector results;
    ngraph::ParameterVector params;
    ngraph::NodeVector to_hold;

    auto get_inputs = [&] (const NodePtr & node) {
        auto pr_edges = node->getParentEdges();
        ngraph::OutputVector inputs(pr_edges.size());

        for (int i = 0; i < pr_edges.size(); i++) {
            auto edge = node->getParentEdgeAt(i);
            int pr_port = edge->getInputNum();
            int ch_port = edge->getOutputNum();
            auto pr_node = edge->getParent();

            IE_ASSERT(node2layer.count(pr_node) == 1);
            auto pr = node2layer[pr_node];

            inputs[ch_port] = pr->output(pr_port);
        }

        return inputs;
    };

    auto create_ngraph_node = [&](const NodePtr &node) {
        bool is_input = false, is_output = false, should_be_hold = false;
        for (auto && kvp : graph.inputNodesMap) {
            if (kvp.second == node) {
                is_input = true;
                break;
            }
        }

        for (auto && kvp : graph.outputNodesMap) {
            if (kvp.second == node) {
                is_output = true;
                break;
            }
        }

        if (!is_output && node->getChildEdges().empty()) {
            // The node has no consumer and is not an output.
            // Should be hold in other irregular way.
            should_be_hold = true;
        }

        auto meta_data = extract_node_metadata(node);
        std::shared_ptr<ngraph::Node> return_node;
        if (is_input) {
            auto& desc = node->getChildEdgeAt(0)->getMemory().getDesc();
            auto param = std::make_shared<ngraph::op::Parameter>(details::convertPrecision(desc.getPrecision()), desc.getShape().toPartialShape());
            return_node = param;
            params.push_back(param);
        } else if (is_output) {
            results.emplace_back(std::make_shared<ngraph::op::Result>(get_inputs(node).back()));
            return_node = results.back();
        } else {
            return_node = std::make_shared<ExecGraphInfoSerialization::ExecutionNode>(
                get_inputs(node), node->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size());

            for (size_t port = 0; port < return_node->get_output_size(); ++port) {
                auto& desc = node->getChildEdgeAt(port)->getMemory().getDesc();
                return_node->set_output_type(port, details::convertPrecision(desc.getPrecision()), desc.getShape().toPartialShape());
            }
        }

        auto& node_rt_info = return_node->get_rt_info();
        node_rt_info["SPD"] = toString(&node->getSelectedPrimitiveDescriptor()->getConfig());
        {
            std::stringstream ss;
            ss << node->getTypeStr();
            for (auto & n : node->getFusedWith()) {
                ss << "," << n->getTypeStr();
            }
            node_rt_info["fusedTypes"] = ss.str();
        }

        for (size_t port = 0; port < return_node->get_output_size(); ++port) {
            if (node->getChildEdges().size() == 0) continue;
            auto& mem = node->getChildEdgeAt(port)->getMemory();
            auto& desc = mem.getDesc();
            // rt_info in each output descriptor encodes edditional
            // edge/memory informations besides precision and shape
            auto& rt_info = return_node->output(port).get_rt_info();
            rt_info["Format"] = desc.serializeFormat();
            rt_info["Precision"] = desc.getPrecision().name();
            rt_info["Data"] = (int64_t)((uintptr_t) mem.GetData());
            rt_info["Ptr"] = (int64_t)((uintptr_t) mem.GetPtr());
            rt_info["MaxMemSize"] = (int64_t)desc.getMaxMemSize();

            auto vecint = [](const VectorDims & vd) {
                std::vector<int> ret;
                for (auto& v : vd)
                    ret.push_back(v);
                return ret;
            };

            auto * pblkdesc = dynamic_cast<const BlockedMemoryDesc *>(&desc);
            if (pblkdesc) {
                rt_info["Strides"] = vecint(pblkdesc->getStrides());
                rt_info["BlockDims"] = vecint(pblkdesc->getBlockDims());
                rt_info["Order"] = vecint(pblkdesc->getOrder());
                rt_info["OffsetPadding"] = (int64_t)pblkdesc->getOffsetPadding();
                rt_info["OffsetPaddingToData"] = vecint(pblkdesc->getOffsetPaddingToData());
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

    ngraph::NodeVector nodes;
    nodes.reserve(graph.graphNodes.size());
    for (auto &node : graph.graphNodes) {  // important: graph.graphNodes are in topological order
        nodes.emplace_back(create_ngraph_node(node));
        node2layer[node] = nodes.back();
    }

    auto holder = results[0];
    for (auto &node : to_hold) {
        holder->add_control_dependency(node);
    }

    return std::make_shared<ngraph::Function>(results, params, graph._name);
}

#ifdef CPU_DEBUG_CAPS
void serialize(const Graph &graph) {
    const std::string& path = graph.getConfig().execGraphPath;

    if (path.empty())
        return;

    if (path == "cout")
        serializeToCout(graph);
    else if (!path.compare(path.size() - 4, 4, ".xml"))
        serializeToXML(graph, path);
    else
        IE_THROW() << "Unknown serialize format. Should be either 'cout' or '*.xml'. Got " << path;
}

void serializeToXML(const Graph &graph, const std::string& path) {
    if (path.empty())
        return;

    std::string binPath;
    ngraph::pass::Manager manager;
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
                std::cout << "in: " << inConfs.front().getMemDesc()->getPrecision().name()
                          << "/l=" << inConfs.front().getMemDesc()->serializeFormat()
                          << "; ";
            }
            auto& outConfs = nodeDesc->getConfig().outConfs;
            if (!outConfs.empty()) {
                std::cout << "out: " << outConfs.front().getMemDesc()->getPrecision().name()
                          << "/l=" << outConfs.front().getMemDesc()->serializeFormat();
            }
        }
        std::cout << " ]"  << std::endl;
    }
}

void summary_perf(const Graph &graph) {
    const std::string& summaryPerf = graph.getConfig().summaryPerf;

    if (summaryPerf.empty())
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
            ss << std::setw(10) << std::right << percentage << " % :" << it.first << std::endl;
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
