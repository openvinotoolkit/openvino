// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef CPU_DEBUG_CAPS

#include "perf_count.h"
#include "cpu_types.h"

#include <string>
#include <vector>
#include <set>
#include <memory>
#include <cassert>

namespace ov {
namespace intel_cpu {

class Node;
typedef size_t PerfKey;

class PerfCount {
public:
    enum CounterIdx : uint8_t {
        Exec = 0,
        ShapeInfer,
        PrepareParams,
        RedefineOutputMemory,
        NumberOfCounters
    };
    struct CounterSum {
        uint64_t duration = 0;
        uint32_t num = 0;
    };
    struct PerfData {
        uint64_t calcTotalDuration() const {
            return counters[Exec].duration + counters[ShapeInfer].duration + counters[RedefineOutputMemory].duration + counters[PrepareParams].duration;
        }

        PerfData& operator+=(const PerfData& rhs) {
            for (uint8_t i = 0; i < NumberOfCounters; i++) {
                counters[i].duration += rhs.counters[i].duration;
                counters[i].num += rhs.counters[i].num;
            }
            nodeShapesSet.insert(rhs.nodeShapesSet.begin(),
                                 rhs.nodeShapesSet.end());
            return *this;
        }

        CounterSum counters[NumberOfCounters];
        typedef std::pair<std::vector<VectorDims>, std::vector<VectorDims>> PerfNodeShape;
        std::set<PerfNodeShape> nodeShapesSet;
    };

    std::vector<PerfData> _perfData;

    PerfCount() {
        _itrDuration[ShapeInfer] = _itrDuration[RedefineOutputMemory] = _itrDuration[PrepareParams] =
            std::chrono::high_resolution_clock::duration::zero();
    }

    std::vector<std::pair<std::string, double>> getItrDurationReport() const {
        const auto& exec = _itrDuration[Exec];
        const auto& shapeInfer = _itrDuration[ShapeInfer];
        const auto& redefineOutputMemory = _itrDuration[RedefineOutputMemory];
        const auto& prepareParams = _itrDuration[PrepareParams];

        std::vector<std::pair<std::string, double>> report{{"exec", exec.count()}};
        if (shapeInfer != std::chrono::high_resolution_clock::duration::zero() ||
            redefineOutputMemory != std::chrono::high_resolution_clock::duration::zero() ||
            prepareParams != std::chrono::high_resolution_clock::duration::zero()) {
            report.emplace_back(std::make_pair("shapeInfer", shapeInfer.count()));
            report.emplace_back(std::make_pair("redefineOutputMemory", redefineOutputMemory.count()));
            report.emplace_back(std::make_pair("prepareParams", prepareParams.count()));
            report.emplace_back(std::make_pair("total", (exec + shapeInfer + redefineOutputMemory + prepareParams).count()));
        }
        return report;
    }

    uint64_t avg() const { return _exec.num ? _exec.duration / _exec.num : 0; }
    uint32_t count() const { return _exec.num; }

private:
    std::chrono::high_resolution_clock::time_point _itrStart[NumberOfCounters] = {};
    std::chrono::duration<double, std::milli> _itrDuration[NumberOfCounters] = {};
    CounterSum _exec;

    void start_itr() {
        _itrStart[Exec] = std::chrono::high_resolution_clock::now();
    }

    void start_stage(const CounterIdx cntrIdx) {
        assert(cntrIdx == ShapeInfer || cntrIdx == RedefineOutputMemory || cntrIdx == PrepareParams);
        _itrStart[cntrIdx] = std::chrono::high_resolution_clock::now();
    }
    void finish_stage(const CounterIdx cntrIdx) {
        assert(cntrIdx == ShapeInfer || cntrIdx == RedefineOutputMemory || cntrIdx == PrepareParams);
        _itrDuration[cntrIdx] = std::chrono::high_resolution_clock::now() - _itrStart[cntrIdx];
    }

    void finish_itr() {
        _itrDuration[Exec] = std::chrono::high_resolution_clock::now() - _itrStart[Exec];
        _exec.duration += std::chrono::duration_cast<std::chrono::microseconds>(_itrDuration[Exec]).count();
        _exec.num++;
    }
    void finish_itr(const PerfKey itrKey, const std::shared_ptr<Node>& node);

    friend class PerfHelper;
    friend class PerfHelperStage;
};

class PerfHelper {
    const std::shared_ptr<Node>& _node;
    const PerfKey _itrKey;

public:
    explicit PerfHelper(const std::shared_ptr<Node>& node, const PerfKey itrKey);
    ~PerfHelper();
};

class PerfHelperStage {
    PerfCount& _count;
    const PerfCount::CounterIdx _cntrIdx;

public:
    explicit PerfHelperStage(PerfCount& count, const PerfCount::CounterIdx cntrIdx) :
        _count(count), _cntrIdx(cntrIdx) { _count.start_stage(_cntrIdx); }
    ~PerfHelperStage() { _count.finish_stage(_cntrIdx); }
};

class Graph;
PerfKey perfGetKey(Graph& graph);
class CompiledModel;
void perfDump(const CompiledModel& execNet);

}   // namespace intel_cpu
}   // namespace ov

#  define PERF(_node, _need, _itrKey) PERF_COUNTER(_need, PerfHelper, _node, _itrKey)
#  define PERF_SHAPE_INFER(_node) PERF_COUNTER(true, PerfHelperStage, _node->PerfCounter(), PerfCount::ShapeInfer)
#  define PERF_PREDEFINE_OUTPUT_MEMORY(_node) PERF_COUNTER(true, PerfHelperStage, _node->PerfCounter(), PerfCount::RedefineOutputMemory)
#  define PERF_PREPARE_PARAMS(_node) PERF_COUNTER(true, PerfHelperStage, _node->PerfCounter(), PerfCount::PrepareParams)
#endif // CPU_DEBUG_CAPS
