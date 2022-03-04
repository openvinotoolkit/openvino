// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <ratio>
#ifdef CPU_DEBUG_CAPS
#include "cpu_types.h"
#include <string>
#include <vector>
#include <set>
#include <memory>
#include <cassert>
#endif // CPU_DEBUG_CAPS

namespace ov {
namespace intel_cpu {

#ifndef CPU_DEBUG_CAPS

class PerfCount {
    uint64_t total_duration;
    uint32_t num;

    std::chrono::high_resolution_clock::time_point __start = {};
    std::chrono::high_resolution_clock::time_point __finish = {};

public:
    PerfCount(): total_duration(0), num(0) {}

    uint64_t avg() const { return (num == 0) ? 0 : total_duration / num; }

private:
    void start_itr() {
        __start = std::chrono::high_resolution_clock::now();
    }

    void finish_itr() {
        __finish = std::chrono::high_resolution_clock::now();
        total_duration += std::chrono::duration_cast<std::chrono::microseconds>(__finish - __start).count();
        num++;
    }

    friend class PerfHelper;
};

class PerfHelper {
    PerfCount &counter;

public:
    explicit PerfHelper(PerfCount &count): counter(count) { counter.start_itr(); }

    ~PerfHelper() { counter.finish_itr(); }
};

#else // CPU_DEBUG_CAPS

typedef size_t PerfKey;

class PerfCount {
public:
    enum CounterIdx : uint8_t {
        Total = 0,
        ShapeInfer,
        PrepareParams,
        NumberOfCounters
    };
    struct CounterSum {
        uint64_t duration = 0;
        uint32_t num = 0;
    };
    struct PerfData {
        uint64_t calcExecDuration() const {
            return counters[Total].duration - counters[ShapeInfer].duration - counters[PrepareParams].duration;
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

    std::vector<std::pair<std::string, double>> getItrDurationReport() const {
        const auto& total = _itrDuration[Total];
        const auto& shapeInfer = _itrDuration[ShapeInfer];
        const auto& prepareParams = _itrDuration[PrepareParams];

        std::vector<std::pair<std::string, double>> report{{"total", total.count()}};
        if (shapeInfer != std::chrono::high_resolution_clock::duration::zero() ||
            prepareParams != std::chrono::high_resolution_clock::duration::zero()) {
            report.emplace_back(std::make_pair("shapeInfer", shapeInfer.count()));
            report.emplace_back(std::make_pair("prepareParams", prepareParams.count()));
            report.emplace_back(std::make_pair("exec", (total - shapeInfer - prepareParams).count()));
        }
        return report;
    }

    uint64_t avg() const { return _total.num ? _total.duration / _total.num : 0; }
    bool isItrStarted() const { return _isItrStarted; }

private:
    bool _isItrStarted = false;
    std::chrono::high_resolution_clock::time_point _itrStart[NumberOfCounters] = {};
    std::chrono::duration<double, std::milli> _itrDuration[NumberOfCounters] = {};
    CounterSum _total;

    void start_itr() {
        _isItrStarted = true;
        _itrDuration[ShapeInfer] = _itrDuration[PrepareParams] =
            std::chrono::high_resolution_clock::duration::zero();
        _itrStart[Total] = std::chrono::high_resolution_clock::now();
    }

    void start_stage(const CounterIdx cntrIdx) {
        assert(cntrIdx == ShapeInfer || cntrIdx == PrepareParams);
        _itrStart[cntrIdx] = std::chrono::high_resolution_clock::now();
    }
    void finish_stage(const CounterIdx cntrIdx) {
        assert(cntrIdx == ShapeInfer || cntrIdx == PrepareParams);
        _itrDuration[cntrIdx] = std::chrono::high_resolution_clock::now() - _itrStart[cntrIdx];
    }

    void finish_itr() {
        _itrDuration[Total] = std::chrono::high_resolution_clock::now() - _itrStart[Total];
        _total.duration += std::chrono::duration_cast<std::chrono::microseconds>(_itrDuration[Total]).count();
        _total.num++;
        _isItrStarted = false;
    }
    void finish_itr(const PerfKey itrKey, PerfData::PerfNodeShape&& itrNodeShape);

    friend class PerfHelper;
    friend class PerfHelperStage;
};

class MKLDNNNode;
class PerfHelper {
    const std::shared_ptr<MKLDNNNode>& _node;
    const PerfKey _itrKey;

public:
    explicit PerfHelper(const std::shared_ptr<MKLDNNNode>& node, const PerfKey itrKey);
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

class MKLDNNGraph;
PerfKey perfGetKey(MKLDNNGraph& graph);
class MKLDNNExecNetwork;
void perfDump(const MKLDNNExecNetwork& execNet);

#endif // CPU_DEBUG_CAPS

#define GET_PERF(_helper, ...) std::unique_ptr<_helper>(new _helper(__VA_ARGS__))
#define PERF_COUNTER(_need, _helper, ...) auto pc = _need ? GET_PERF(_helper, __VA_ARGS__) : nullptr;

#ifdef CPU_DEBUG_CAPS
#define PERF(_node, _need, _itrKey) PERF_COUNTER(_need, PerfHelper, _node, _itrKey)
#define PERF_SHAPE_INFER(_node) PERF_COUNTER(_node->PerfCounter().isItrStarted(), PerfHelperStage, _node->PerfCounter(), PerfCount::ShapeInfer)
#define PERF_PREPARE_PARAMS(_node) PERF_COUNTER(_node->PerfCounter().isItrStarted(), PerfHelperStage, _node->PerfCounter(), PerfCount::PrepareParams)
#else
#define PERF(_node, _need, _itrKey) PERF_COUNTER(_need, PerfHelper, _node->PerfCounter())
#define PERF_SHAPE_INFER(_node)
#define PERF_PREPARE_PARAMS(_node)
#endif // CPU_DEBUG_CAPS

}   // namespace intel_cpu
}   // namespace ov
