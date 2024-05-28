// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef CPU_DEBUG_CAPS

#include "perf_count_debug_caps.h"
#include "compiled_model.h"
#include <fstream>
#include <iomanip>

namespace ov {
namespace intel_cpu {

static double perfAvg(const uint64_t sum, const uint64_t num);
static double perfAvg(const double sum, const uint64_t num);
static double perfPercent(const uint64_t val, const uint64_t total);
static double perfDeviationPercent(const std::vector<double>& values, const double sum);
static void perfShapeToStr(const VectorDims& shape, std::string& str);
static void perfDumpNodes(const std::string& path, const std::vector<NodePtr>& nodes, const std::vector<std::string>& modelInputShapes);

void PerfCount::finish_itr(const PerfKey itrKey, const std::shared_ptr<Node>& node) {
    finish_itr();

    assert(itrKey <= _perfData.size());
    if (itrKey == _perfData.size())
        _perfData.emplace_back();

    auto& perfData = _perfData[itrKey];
    for (uint8_t i = 0; i < NumberOfCounters; i++) {
        if (_itrDuration[i] != std::chrono::high_resolution_clock::duration::zero()) {
            perfData.counters[i].duration += std::chrono::duration_cast<std::chrono::microseconds>(_itrDuration[i]).count();
            perfData.counters[i].num++;
        }
    }

    std::vector<VectorDims> in, out;
    in.reserve(node->getParentEdges().size());
    for (auto i = 0; i < node->getParentEdges().size(); i++) {
        in.push_back(node->getParentEdgeAt(i)->getMemory().getStaticDims());
    }
    out.reserve(node->getChildEdges().size());
    for (auto i = 0; i < node->getChildEdges().size(); i++) {
        out.push_back(node->getChildEdgeAt(i)->getMemory().getStaticDims());
    }
    perfData.nodeShapesSet.insert(std::make_pair(std::move(in), std::move(out)));
}

PerfHelper::PerfHelper(const std::shared_ptr<Node>& node, const PerfKey itrKey) :
        _node(node), _itrKey(itrKey) {
    _node->PerfCounter().start_itr();
}

PerfHelper::~PerfHelper() {
    if (_itrKey != std::numeric_limits<PerfKey>::max()) {
        _node->PerfCounter().finish_itr(_itrKey, _node);
    } else {
        _node->PerfCounter().finish_itr();
    }
}

PerfKey perfGetKey(Graph& graph) {
    if (graph.getConfig().debugCaps.perfTablesPath.empty() || !graph.getConfig().collectPerfCounters)
        return std::numeric_limits<PerfKey>::max();

    std::vector<VectorDims> modelInputDims;
    modelInputDims.reserve(graph.inputNodesMap.size());
    for (const auto& input : graph.inputNodesMap) {
        modelInputDims.push_back(input.second->getChildEdgeAt(0)->getMemory().getStaticDims());
    }
    return graph.perfKeysMap.emplace(std::move(modelInputDims), graph.perfKeysMap.size()).first->second;
}

void perfDump(const CompiledModel& execNet) {
    auto& graphs = execNet.m_graphs;
    const auto graphsNum = graphs.size();
    auto& graph = graphs[0];

    if (graphsNum == 0 ||
        graph.getConfig().debugCaps.perfTablesPath.empty() || graph.perfKeysMap.size() == 0)
        return;

    // use 1st graph to accamulate perf data (save memory and time)
    for (auto graphIdx = 1; graphIdx < graphsNum; graphIdx++) {
        if (graphs[graphIdx].perfKeysMap.size() == 0)
            continue;

        PerfKey keyToKey[graphs[graphIdx].perfKeysMap.size()];
        for (const auto& pair : graphs[graphIdx].perfKeysMap) {
            keyToKey[pair.second] =
                graph.perfKeysMap.emplace(pair.first, graph.perfKeysMap.size()).first->second;;
        }
        assert(graph.m_executableGraphNodes.size() == graphs[graphIdx].executableGraphNodes.size());
        for (auto nodeIdx = 0; nodeIdx < graph.m_executableGraphNodes.size(); nodeIdx++) {
            auto& aggPerfData = graph.m_executableGraphNodes[nodeIdx]->PerfCounter()._perfData;
            const auto& perfData = graphs[graphIdx].m_executableGraphNodes[nodeIdx]->PerfCounter()._perfData;
            aggPerfData.resize(graph.perfKeysMap.size());
            for (auto key = 0; key < perfData.size(); key++) {
                aggPerfData[keyToKey[key]] += perfData[key];
            }
        }
    }

    std::vector<std::string> inputNames;
    inputNames.reserve(graph.inputNodesMap.size());
    for (const auto& input : graph.inputNodesMap) {
        inputNames.emplace_back(inputNames.size() ? ',' + std::to_string(input.first) : std::to_string(input.first));
    }
    std::vector<std::string> modelInputs(graph.perfKeysMap.size());
    for (auto perf : graph.perfKeysMap) {
        assert(perf.second < modelInputs.size());
        auto& modelInput = modelInputs[perf.second];
        assert(modelInput.empty());
        assert(perf.first.size() == inputNames.size());
        for (auto idx = 0; idx < inputNames.size(); idx++) {
            modelInput.append(inputNames[idx]);
            perfShapeToStr(perf.first[idx], modelInput);
        }
    }

    perfDumpNodes(graph.getConfig().debugCaps.perfTablesPath, graph.m_executableGraphNodes, modelInputs);
}

static double perfAvg(const uint64_t sum, const uint64_t num) {
    return num ? sum / static_cast<double>(num) : 0;
}
static double perfAvg(const double sum, const uint64_t num) {
    return num ? sum / num : 0;
}

static double perfPercent(const uint64_t val, const uint64_t total) {
    return val ? static_cast<double>(val) / total * 100 : 0;
}

static double perfDeviationPercent(const std::vector<double>& values, const double sum) {
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

static void perfShapeToStr(const VectorDims& shape, std::string& str) {
    str.push_back('{');
    for (auto i = 0; i < shape.size(); i++) {
        if (i)
            str.push_back(',');
        str.append(std::to_string(shape[i]));
    }
    str.push_back('}');
}

static void perfDumpNodes(const std::string& path, const std::vector<NodePtr>& nodes,
                   const std::vector<std::string>& modelInputShapes) {
    assert(modelInputShapes.size());
    const std::string pathPrefix(path + "/perf_");
    std::ofstream csv;
    auto openCsv = [&] (std::ofstream& csv, const std::string& name) {
        std::string fullPath(pathPrefix + name);
        csv.open(fullPath);
        if (!csv.is_open())
            OPENVINO_THROW("Dumping perf counters. Cannot open file ", fullPath);

        csv << std::fixed << std::setprecision(2);
    };

    openCsv(csv, "modelInputs.csv");
    const auto modelInputsNum = modelInputShapes.size();
    csv << "Model input index,Model input shape\n";
    for (auto i = 0; i < modelInputsNum; i++) {
        csv << i << ",\"" << modelInputShapes[i] << "\"\n";
    }
    csv.close();

    enum CounterIdx : uint8_t {
        Total = 0,
        Exec,
        ShapeInfer,
        RedefineOutputMemory,
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
                   counters[ShapeInfer].durationSum + counters[RedefineOutputMemory].durationSum +
                   counters[PrepareParams].durationSum + counters[Exec].durationSum);
            assert(counters[ShapeInfer].durationSum != 0 ||
                   counters[RedefineOutputMemory].durationSum != 0 ||
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
        std::set<PerfCount::PerfData::PerfNodeShape> uniqShapesSet;
        uint32_t nodeNum = 0;
    };

    auto nodeShapeToStr = [] (const PerfCount::PerfData::PerfNodeShape& shape) {
        auto shapesToStr = [] (const std::vector<VectorDims>& shape, std::string& str) {
            for (auto i = 0; i < shape.size(); i++) {
                if (i)
                    str.push_back(',');
                perfShapeToStr(shape[i], str);
            }
        };
        std::string str;
        shapesToStr(shape.first, str);
        str.append("->");
        shapesToStr(shape.second, str);
        return str;
    };

    auto modelInputToFileName = [&modelInputShapes] (const size_t idx) {
        return std::to_string(idx) + '_' + modelInputShapes[idx].substr(0, 100);
    };
    auto closeWithModelInput = [&modelInputShapes] (std::ofstream& csv, const size_t idx) {
        csv << "Model Input Shape:,\"" << modelInputShapes[idx] << "\"\n";
        csv.close();
    };

    std::vector<std::map<Type, NodeTypePerfData>> modelInputs(modelInputsNum);

    openCsv(csv, "raw_nodes.csv");
    csv << "Node name,Node type,Model input index,Model input shape,Node in->out shapes,"
           "\"Total time (avg, us)\",\"Total time (sum, us)\",Total (n),"
           "\"Exec time (avg, us)\",\"Exec time (sum, us)\",Exec (n),"
           "\"ShapeInfer time (avg, us)\",\"ShapeInfer time (sum, us)\",ShapeInfer (n),"
           "\"RedefineOutputMemory time (avg, us)\",\"RedefineOutputMemory time (sum, us)\",RedefineOutputMemory (n),"
           "\"PrepareParams time (avg, us)\",\"PrepareParams time (sum, us)\",PrepareParams (n),"
           "Comments\n";
    std::vector<std::ofstream> nodeCsvs(modelInputsNum);
    for (const auto& node : nodes) {
        const std::string nodePrefixStr('"' + node->getName() + "\",\"" +
                                        NameFromType(node->getType()) + "\",");
        for (auto idx = 0; idx < node->PerfCounter()._perfData.size(); idx++) {
            const auto& nodePerfData = node->PerfCounter()._perfData[idx];
            auto& modelInputNodeTypesMap = modelInputs[idx];
            auto& modelInputCsv = nodeCsvs[idx];


            csv << nodePrefixStr << idx << ",\"" << modelInputShapes[idx] << "\",\"";
            if (!modelInputCsv.is_open()) {
                openCsv(modelInputCsv, modelInputToFileName(idx) + "_nodes.csv");
                modelInputCsv << "Node name,Node type,Node in->out shapes,"
                                 "\"Total time (avg, us)\",\"Exec time (avg, us)\","
                                 "\"ShapeInfer time (avg, us)\",\"RedefineOutputMemory time (avg, us)\","
                                 "\"PrepareParams time (avg, us)\","
                                 "Comments\n";
            }
            modelInputCsv << nodePrefixStr << '"';

            auto& nodeTypePerfData = modelInputNodeTypesMap[node->getType()];
            nodeTypePerfData.nodeNum++;

            for (const auto& shape : nodePerfData.nodeShapesSet) {
                std::string shapeStr(nodeShapeToStr(shape));
                csv << shapeStr;
                modelInputCsv << shapeStr;
                nodeTypePerfData.uniqShapesSet.insert(shape);
            }
            csv << "\",";
            modelInputCsv << "\",";

            for (uint8_t nodeTypeCntrIdx = 0; nodeTypeCntrIdx < CounterIdx::NumberOfCounters; nodeTypeCntrIdx++) {
                auto& nodeTypeCntr = nodeTypePerfData.counters[nodeTypeCntrIdx];
                const PerfCount::CounterIdx nodeCntrIdxMap[NumberOfCounters] =
                    {PerfCount::NumberOfCounters, PerfCount::Exec, PerfCount::ShapeInfer, PerfCount::RedefineOutputMemory, PerfCount::PrepareParams};
                const auto nodeCntrIdx = nodeCntrIdxMap[nodeTypeCntrIdx];

                uint64_t nodeCntrDuration;
                uint32_t nodeCntrNum;
                if (nodeCntrIdx == PerfCount::NumberOfCounters) {
                    assert(nodeTypeCntrIdx == CounterIdx::Total);
                    nodeCntrDuration = nodePerfData.calcTotalDuration();
                    nodeCntrNum = nodePerfData.counters[PerfCount::Exec].num;
                } else {
                    const auto& nodeCntr = nodePerfData.counters[nodeCntrIdx];
                    nodeCntrDuration = nodeCntr.duration;
                    nodeCntrNum = nodeCntr.num;
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
    for (size_t i = 0; i < nodeCsvs.size(); i++) {
        closeWithModelInput(nodeCsvs[i], i);
    } // raw_nodes.csv and *_nodes.csv
    assert(modelInputs.size() == modelInputsNum);

    const std::string nodeTypeHeader(
        "Node type,Node number,Uniq shape number,"
        "\"Total time (avg, us)\",\"Total time (dev, %)\",\"Total time (sum, us)\",Total time (%),"
        "\"Exec time (avg, us)\",\"Exec time (dev, %)\",\"Exec time (sum, us)\",Exec time (%),"
        "\"ShapeInfer time (avg, us)\",\"ShapeInfer time (dev, %)\",\"ShapeInfer time (sum, us)\",ShapeInfer time (%),"
        "\"RedefineOutputMemory time (avg, us)\",\"RedefineOutputMemory time (dev, %)\",\"RedefineOutputMemory time (sum, us)\",RedefineOutputMemory time (%),"
        "\"PrepareParams time (avg, us)\",\"PrepareParams time (dev, %)\",\"PrepareParams time (sum, us)\",PrepareParams time (%)\n");
    auto printNodeTypeCntr = [] (std::ofstream& csv, const NodeTypePerfCounter& cntr,
                                 const double avg, const uint64_t totalDuration) {
        csv << ','
            << avg << ','
            << perfDeviationPercent(cntr.avgValues, cntr.avgSum) << ','
            << cntr.durationSum << ','
            << perfPercent(cntr.durationSum, totalDuration);
    };
    auto finalizeNodeTypeCsv = [&printNodeTypeCntr] (std::ofstream& csv,
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
        csv << std::endl;
    };

    const bool isDynamicModelInput = (modelInputsNum > 1);
    std::map<Type, NodeTypePerfData> aggregateNodeTypesMap;

    NodeTypePerfData modelInputTotal, total;
    const auto nodeTypesNum = modelInputs[0].size();
    for (uint8_t idx = Total; idx < NumberOfCounters; idx++) {
        modelInputTotal.counters[idx].avgValues.reserve(nodeTypesNum);
        if (isDynamicModelInput)
            total.counters[idx].avgValues.reserve(nodeTypesNum);
    }
    for (auto& nodeType : modelInputs[0]) {
        total.nodeNum += nodeType.second.nodeNum;
        if (isDynamicModelInput) {
            auto& aggregateNodeType = aggregateNodeTypesMap[nodeType.first];
            aggregateNodeType.nodeNum = nodeType.second.nodeNum;
            for (uint8_t idx = Total; idx < NumberOfCounters; idx++) {
                aggregateNodeType.counters[idx].avgValues.reserve(modelInputsNum);
            }
        }
    }

    for (auto i = 0; i < modelInputsNum; i++) {
        const auto& nodeTypesMap = modelInputs[i];

        openCsv(csv, modelInputToFileName(i) + "_nodeTypes.csv");
        csv << nodeTypeHeader;

        modelInputTotal.nodeNum = total.nodeNum;
        for (auto& nodeType : nodeTypesMap) {
            for (uint8_t idx = Total; idx < NumberOfCounters; idx++) {
                modelInputTotal.counters[idx].durationSum += nodeType.second.counters[idx].durationSum;
            }
        }
        if (isDynamicModelInput) {
            for (uint8_t idx = Total; idx < NumberOfCounters; idx++) {
                total.counters[idx].durationSum += modelInputTotal.counters[idx].durationSum;
            }
        }

        for (auto& nodeType : nodeTypesMap) {
            auto& nodeTypePerfData = nodeType.second;
            nodeTypePerfData.validate(nodeTypePerfData.nodeNum);

            csv << '"' << NameFromType(nodeType.first) << "\","
                << nodeTypePerfData.nodeNum << ',' << nodeTypePerfData.uniqShapesSet.size();

            modelInputTotal.uniqShapesSet.insert(nodeTypePerfData.uniqShapesSet.begin(),
                                                 nodeTypePerfData.uniqShapesSet.end());
            if (isDynamicModelInput) {
                auto& aggregateNodeType = aggregateNodeTypesMap[nodeType.first];
                assert(aggregateNodeType.nodeNum == nodeTypePerfData.nodeNum);
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
        closeWithModelInput(csv, i);
        modelInputTotal.cleanup();
    } // *_nodeTypes.csv

    if (isDynamicModelInput) {
        openCsv(csv, "all_nodeTypes.csv");
        csv << nodeTypeHeader;
        for (auto& nodeType : aggregateNodeTypesMap) {
            auto& nodeTypePerfData = nodeType.second;
            nodeTypePerfData.validate(modelInputsNum);

            csv << '"' << NameFromType(nodeType.first) << "\","
                << nodeTypePerfData.nodeNum << ',' << nodeTypePerfData.uniqShapesSet.size();
            total.uniqShapesSet.insert(nodeTypePerfData.uniqShapesSet.begin(),
                                       nodeTypePerfData.uniqShapesSet.end());

            for (uint8_t idx = Total; idx < NumberOfCounters; idx++) {
                const auto& cntr = nodeTypePerfData.counters[idx];
                auto& totalCntr = total.counters[idx];
                const auto avg = perfAvg(cntr.avgSum, cntr.avgValues.size());

                printNodeTypeCntr(csv, cntr, avg, totalCntr.durationSum);

                totalCntr.avgValues.push_back(avg);
                totalCntr.avgSum += avg;
            }
            csv << std::endl;
        }
        finalizeNodeTypeCsv(csv, total, aggregateNodeTypesMap.size());
        csv.close();
    }
}

}   // namespace intel_cpu
}   // namespace ov
#endif // CPU_DEBUG_CAPS
