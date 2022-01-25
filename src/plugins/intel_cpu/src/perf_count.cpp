// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef CPU_DEBUG_CAPS

#include "perf_count.h"
#include "mkldnn_exec_network.h"
#include <fstream>
#include <iomanip>

namespace MKLDNNPlugin {

double perfAvg(const uint64_t sum, const uint64_t num);
double perfPercent(const uint64_t val, const uint64_t total);
double perfDeviationPercent(const std::vector<double>& values, const double sum);
void perfDumpNodes(const std::string& path, const std::vector<MKLDNNNodePtr>& nodes);

void PerfCount::finish_itr(const std::string& itrKey, const std::string& itrNodeShape) {
    finish_itr(Total);

    auto &perfData = _perfDataMap[itrKey];
    for (uint8_t i = 0; i < NumberOfCounters; i++) {
        if (_itrDuration[i] != std::chrono::high_resolution_clock::duration::zero()) {
            perfData.counters[i].duration += std::chrono::duration_cast<std::chrono::microseconds>(_itrDuration[i]).count();
            perfData.counters[i].num++;
        }
    }
    perfData.nodeShapesSet.insert(itrNodeShape);

    _isItrStarted = false;
}

PerfHelperTotal::PerfHelperTotal(const std::shared_ptr<MKLDNNNode>& node, const std::string& itrKey) :
        _node(node), _count(node->PerfCounter()), _itrKey(itrKey) {
    _count.start_itr();
}

PerfHelperTotal::~PerfHelperTotal() {
    std::string shape;
    typedef const MKLDNNEdgePtr (MKLDNNNode::*GetEdgeAtMp)(size_t) const;
    auto edgesToShapeStr = [&] (const size_t edgeNum, GetEdgeAtMp getEdgeAt) {
        for (auto i = 0; i < edgeNum; i++) {
            if (i)
                shape.push_back(',');
            shape.push_back('{');
            const auto& dims = ((*_node).*(getEdgeAt))(i)->getMemory().getStaticDims();
            for (auto j = 0; j < dims.size(); j++) {
                if (j)
                    shape.push_back(',');
                shape.append(std::to_string(dims[j]));
            }
            shape.push_back('}');
        }
    };
    edgesToShapeStr(_node->getParentEdges().size(), &MKLDNNNode::getParentEdgeAt);
    shape.append("->");
    edgesToShapeStr(_node->getChildEdges().size(), &MKLDNNNode::getChildEdgeAt);

    _count.finish_itr(_itrKey, shape);
}

double perfAvg(const uint64_t sum, const uint64_t num) {
    return num ? sum / static_cast<double>(num) : 0;
}

double perfPercent(const uint64_t val, const uint64_t total) {
    return val ? static_cast<double>(val) / total * 100 : 0;
}

double perfDeviationPercent(const std::vector<double>& values, const double sum) {
    if (values.size() == 0 || sum == 0)
        return 0;

    const double n = static_cast<double>(values.size());
    const double avg = sum / n;
    double variance = 0;

    for (double val : values) {
        val -= avg;
        variance += val * val;
    }
    variance /= n;

    return std::sqrt(variance) / avg * 100;
}

std::string perfGetModelInputStr(const MKLDNNGraph& graph) {
    std::string str;
    for (const auto& input : graph.GetInputNodesMap()) {
        if (str.size())
            str.push_back(',');
        str.append(input.first + '{');
        const auto& dims = input.second->getChildEdgeAt(0)->getMemory().getStaticDims();
        for (auto i = 0; i < dims.size(); i++) {
            if (i)
                str.push_back(',');
            str.append(std::to_string(dims[i]));
        }
        str.push_back('}');
    }
    return str;
}
void perfDump(const MKLDNNExecNetwork& execNet) {
    const auto& graphs = execNet._graphs;
    const auto graphNum = graphs.size();

    if (graphNum == 0)
        return;

    const auto& graph = graphs[0];
    if (graph.config.perfTablesPath.empty()) {
        return;
    }

    // use 1st graph to accamulate perf data (save memory and time)
    for (auto i = 1; i < graphNum; i++) {
        for (auto nodeIdx = 0; nodeIdx < graph.executableGraphNodes.size(); nodeIdx++) {
            auto& aggPerfMap = graph.executableGraphNodes[nodeIdx]->PerfCounter()._perfDataMap;
            const auto& perfMap = graphs[i].executableGraphNodes[nodeIdx]->PerfCounter()._perfDataMap;
            for (const auto& perf : perfMap) {
                aggPerfMap[perf.first] += perf.second;
            }
        }
    }

    perfDumpNodes(graph.config.perfTablesPath, graph.executableGraphNodes);
}

void perfDumpNodes(const std::string& path, const std::vector<MKLDNNNodePtr>& nodes) {
    const std::string pathPrefix(path + "perf_");
    std::ofstream csv;
    auto openCsv = [&] (std::ofstream& csv, const std::string& name) {
        csv.open(pathPrefix + name);
        csv << std::fixed << std::setprecision(2);
    };

    enum CounterIdx : uint8_t {
        Total = 0,
        Exec,
        ShapeInfer,
        PrepareParams,
        NumberOfCounters
    };

    struct NodeTypePerfCounter {
        std::vector<double> avgValues = {};
        double avgSum = 0;
        uint64_t durationSum = 0;
    };

    struct NodeTypePerfData {
        void cleanup() {
            for (uint8_t idx = 0; idx < NumberOfCounters; idx++) {
                auto& cntr = counters[idx];
                cntr.avgValues.clear();
                cntr.avgSum = cntr.durationSum = 0;
            }
            uniqShapesSet.clear();
            nodeNum = 0;
        }
        void validate(const size_t avgNum) const {
            assert(nodeNum >= 1);
            assert(uniqShapesSet.size() >= 1);
            assert(counters[Total].durationSum ==
                   counters[ShapeInfer].durationSum + counters[PrepareParams].durationSum + counters[Exec].durationSum);
            assert(counters[ShapeInfer].durationSum != 0 ||
                   counters[PrepareParams].durationSum != 0 ||
                   counters[Total].durationSum == counters[Exec].durationSum);
            for (uint8_t idx = Total; idx < NumberOfCounters; idx++) {
                assert(avgNum >= 1);
                assert(counters[idx].avgValues.size() == avgNum);
                assert(counters[idx].avgSum ==
                          std::accumulate(counters[idx].avgValues.begin(),
                                          counters[idx].avgValues.end(), 0.0));
            }
        }

        NodeTypePerfCounter counters[NumberOfCounters];
        std::set<std::string> uniqShapesSet;
        uint32_t nodeNum = 0;
    };

    struct ModelInputPerfData {
        std::ofstream csv;
        std::map<Type, NodeTypePerfData> nodeTypesMap;
    };

    std::map<std::string, ModelInputPerfData> modelInputsMap;
    std::map<Type, NodeTypePerfData> aggregateNodeTypesMap;
    NodeTypePerfData total;

    openCsv(csv, "raw_nodes.csv");
    csv << "Node name,Node type,Model input shape,Node in->out shapes,"
           "\"Total time (avg, us)\",\"Total time (sum, us)\",Total (n),"
           "\"Exec time (avg, us)\",\"Exec time (sum, us)\",Exec (n),"
           "\"ShapeInfer time (avg, us)\",\"ShapeInfer time (sum, us)\",ShapeInfer (n),"
           "\"PrepareParams time (avg, us)\",\"PrepareParams time (sum, us)\",PrepareParams (n),"
           "Comments\n";
    for (const auto& node : nodes) {
        const std::string nodePrefixStr('"' + node->getName() + "\",\"" +
                                        NameFromType(node->getType()) + "\",\"");
        for (const auto& nodePerf : node->PerfCounter()._perfDataMap) {
            const auto& modelInput = nodePerf.first;
            const auto& nodePerfData = nodePerf.second;
            auto& modelInputData = modelInputsMap[modelInput];
            auto& modelInputCsv = modelInputData.csv;

            csv << nodePrefixStr << modelInput << "\",\"";
            if (!modelInputCsv.is_open()) {
                openCsv(modelInputCsv, "modelInput_" + modelInput + "_nodes" + ".csv");
                modelInputCsv << "Node name,Node type,Node in->out shapes,"
                                 "\"Total time (avg, us)\",\"Exec time (avg, us)\","
                                 "\"ShapeInfer time (avg, us)\",\"PrepareParams time (avg, us)\","
                                 "Comments\n";
            }
            modelInputCsv << nodePrefixStr;

            auto& nodeTypePerfData = modelInputData.nodeTypesMap[node->getType()];
            nodeTypePerfData.nodeNum++;

            for (auto const &shape : nodePerfData.nodeShapesSet) {
                assert(!shape.empty());
                csv << shape;
                modelInputCsv << shape;
                nodeTypePerfData.uniqShapesSet.insert(shape);
            }
            csv << "\",";
            modelInputCsv << "\",";

            for (uint8_t nodeTypeCntrIdx = Total; nodeTypeCntrIdx < NumberOfCounters; nodeTypeCntrIdx++) {
                auto& nodeTypeCntr = nodeTypePerfData.counters[nodeTypeCntrIdx];
                const PerfCount::CounterIdx nodeCntrIdxMap[NumberOfCounters] =
                    {PerfCount::Total, PerfCount::NumberOfCounters, PerfCount::ShapeInfer, PerfCount::PrepareParams};
                const auto nodeCntrIdx = nodeCntrIdxMap[nodeTypeCntrIdx];

                uint64_t nodeCntrDuration;
                uint32_t nodeCntrNum;
                if (nodeCntrIdx != PerfCount::NumberOfCounters) {
                    const auto& nodeCntr = nodePerfData.counters[nodeCntrIdx];
                    nodeCntrDuration = nodeCntr.duration;
                    nodeCntrNum = nodeCntr.num;
                } else {
                    assert(nodeTypeCntrIdx == Exec);
                    nodeCntrDuration = nodePerfData.calcExecDuration();
                    nodeCntrNum = nodePerfData.counters[PerfCount::Total].num;
                }

                nodeTypeCntr.durationSum += nodeCntrDuration;
                const auto avg = perfAvg(nodeCntrDuration, nodeCntrNum);
                nodeTypeCntr.avgValues.push_back(avg);
                nodeTypeCntr.avgSum += avg;

                csv << avg  << ',' << nodeCntrDuration << ',' << nodeCntrNum << ',';
                modelInputCsv << avg << ',';
            }

            const std::string ending(nodePerfData.nodeShapesSet.size() > 1 ? "internal dynamism\n" : ",\n");
            csv << ending;
            modelInputCsv << ending;
        }
    }
    csv.close();
    for (auto& modelInput : modelInputsMap) {
        modelInput.second.csv.close();
    } // raw_nodes.csv and modelInput_*_nodes.csv

    const bool isDynamicModelInput = (modelInputsMap.size() > 1);

    const std::string nodeTypeHeader(
        "Node type,Node number,Uniq shape number,"
        "\"Total time (avg, us)\",\"Total time (dev, %)\",\"Total time (sum, us)\",Total time (%),"
        "\"Exec time (avg, us)\",\"Exec time (dev, %)\",\"Exec time (sum, us)\",Exec time (%),"
        "\"ShapeInfer time (avg, us)\",\"ShapeInfer time (dev, %)\",\"ShapeInfer time (sum, us)\",ShapeInfer time (%),"
        "\"PrepareParams time (avg, us)\",\"PrepareParams time (dev, %)\",\"PrepareParams time (sum, us)\",PrepareParams time (%)\n");
    auto printNodeTypeCntr = [] (std::ofstream& csv, const NodeTypePerfCounter& cntr,
                                 const double avg, const uint64_t totalDuration) {
        csv << ','
            << avg << ','
            << perfDeviationPercent(cntr.avgValues, cntr.avgSum) << ','
            << cntr.durationSum << ','
            << perfPercent(cntr.durationSum, totalDuration);
    };
    auto finalizeNodeTypeCsv = [&] (std::ofstream& csv,
                                   const NodeTypePerfData& aggregateData, const size_t num) {
        aggregateData.validate(num);
        csv << "Total,"
            << aggregateData.nodeNum << ','
            << aggregateData.uniqShapesSet.size();
        for (uint8_t idx = Total; idx < NumberOfCounters; idx++) {
            const auto& cntr = aggregateData.counters[idx];
            printNodeTypeCntr(csv, aggregateData.counters[idx],
                              perfAvg(cntr.avgSum, num), aggregateData.counters[Total].durationSum);
        }
        csv << std::endl;;
        csv.close();
    };

    for (const auto& modelInput : modelInputsMap) {
        openCsv(csv, "modelInput_" + modelInput.first + "_nodeTypes" + ".csv");
        csv << nodeTypeHeader;

        NodeTypePerfData modelInputTotal;
        const auto& nodeTypesMap = modelInput.second.nodeTypesMap;
        for (auto& nodeType : nodeTypesMap) {
            for (uint8_t idx = Total; idx < NumberOfCounters; idx++) {
                modelInputTotal.counters[idx].durationSum += nodeType.second.counters[idx].durationSum;
            }
        }
        for (auto& nodeType : nodeTypesMap) {
            auto& nodeTypePerfData = nodeType.second;
            nodeTypePerfData.validate(nodeTypePerfData.nodeNum);

            csv << '"' << NameFromType(nodeType.first) << "\","
                << nodeTypePerfData.nodeNum << ',' << nodeTypePerfData.uniqShapesSet.size();

            modelInputTotal.nodeNum += nodeTypePerfData.nodeNum;
            modelInputTotal.uniqShapesSet.insert(nodeTypePerfData.uniqShapesSet.begin(),
                                                 nodeTypePerfData.uniqShapesSet.end());
            if (isDynamicModelInput) {
                auto& aggregateNodeType = aggregateNodeTypesMap[nodeType.first];
                assert(aggregateNodeType.nodeNum == nodeTypePerfData.nodeNum ||
                       aggregateNodeType.nodeNum == 0);
                aggregateNodeType.nodeNum = nodeTypePerfData.nodeNum;

                aggregateNodeType.uniqShapesSet.insert(nodeTypePerfData.uniqShapesSet.begin(),
                                                    nodeTypePerfData.uniqShapesSet.end());
            }

            for (uint8_t idx = Total; idx < NumberOfCounters; idx++) {
                auto& cntr = nodeTypePerfData.counters[idx];
                auto& modelInputAggregateCntr = modelInputTotal.counters[idx];
                const auto avg = perfAvg(cntr.avgSum, nodeTypePerfData.nodeNum);

                printNodeTypeCntr(csv, cntr, avg, modelInputAggregateCntr.durationSum);

                modelInputAggregateCntr.avgValues.push_back(avg);
                modelInputAggregateCntr.avgSum += avg;
                if (isDynamicModelInput) {
                    auto& nodeTypeAggregateCntr = aggregateNodeTypesMap[nodeType.first].counters[idx];
                    nodeTypeAggregateCntr.durationSum += cntr.durationSum;
                    nodeTypeAggregateCntr.avgValues.push_back(avg);
                    nodeTypeAggregateCntr.avgSum += avg;
                }
            }
            csv << std::endl;
        }
        finalizeNodeTypeCsv(csv, modelInputTotal, nodeTypesMap.size());

        if (isDynamicModelInput) {
            for (uint8_t idx = Total; idx < NumberOfCounters; idx++) {
                total.counters[idx].durationSum += modelInputTotal.counters[idx].durationSum;
            }
        }
    } // modelInput_*_nodeTypes.csv

    if (isDynamicModelInput) {
        openCsv(csv, "modelInput_all_nodeTypes.csv");
        csv << nodeTypeHeader;
        for (auto& nodeType : aggregateNodeTypesMap) {
            auto& nodeTypePerfData = nodeType.second;
            nodeTypePerfData.validate(modelInputsMap.size());

            csv << '"' << NameFromType(nodeType.first) << "\","
                << nodeTypePerfData.nodeNum << ',' << nodeTypePerfData.uniqShapesSet.size();
            total.nodeNum += nodeTypePerfData.nodeNum;
            total.uniqShapesSet.insert(nodeTypePerfData.uniqShapesSet.begin(),
                                       nodeTypePerfData.uniqShapesSet.end());

            for (uint8_t idx = Total; idx < NumberOfCounters; idx++) {
                const auto& cntr = nodeTypePerfData.counters[idx];
                auto& totalCntr = total.counters[idx];
                const auto avg = perfAvg(cntr.avgSum, cntr.avgValues.size());

                printNodeTypeCntr(csv, cntr, avg, total.counters[Total].durationSum);

                totalCntr.avgValues.push_back(avg);
                totalCntr.avgSum += avg;
            }
            csv << std::endl;
        }
        finalizeNodeTypeCsv(csv, total, aggregateNodeTypesMap.size());
    }
}
} // namespace MKLDNNPlugin
#endif // CPU_DEBUG_CAPS
