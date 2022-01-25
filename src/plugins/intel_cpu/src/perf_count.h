// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <ratio>
#ifdef CPU_DEBUG_CAPS
#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#endif // CPU_DEBUG_CAPS

namespace MKLDNNPlugin {
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
        std::set<std::string> nodeShapesSet;
    };

    std::map<std::string, PerfData> _perfDataMap;
    bool _isItrStarted = false;

    std::string getItrDurationReport() const {
        const auto& total = _itrDuration[Total];
        const auto& shapeInfer = _itrDuration[ShapeInfer];
        const auto& prepareParams = _itrDuration[PrepareParams];

        std::string reportStr("time(ms):total:" + std::to_string(total.count()));
        if (shapeInfer != std::chrono::high_resolution_clock::duration::zero() ||
            prepareParams != std::chrono::high_resolution_clock::duration::zero()) {
            reportStr.append(" time(ms):shapeInfer:" + std::to_string(shapeInfer.count()) +
                             " time(ms):prepareParams:" + std::to_string(prepareParams.count()) +
                             " time(ms):exec:" + std::to_string((total - shapeInfer - prepareParams).count()));
        }
        return reportStr;
    }

    uint64_t avg() const {
        uint64_t totalDur = 0;
        uint64_t totalNum = 0;
        for (const auto& input : _perfDataMap) {
            totalDur += input.second.counters[Total].duration;
            totalNum += input.second.counters[Total].num;
        }
        return totalNum ? totalDur / totalNum : 0;
    }

private:
    std::chrono::high_resolution_clock::time_point _itrStart[NumberOfCounters] = {};
    std::chrono::duration<double, std::milli> _itrDuration[NumberOfCounters] = {};

    void start_itr() {
        _itrDuration[ShapeInfer] = _itrDuration[PrepareParams] =
            std::chrono::high_resolution_clock::duration::zero();
        start_itr(Total);
    }
    void start_itr(const CounterIdx cntrIdx) {
        _isItrStarted = true;
        _itrStart[cntrIdx] = std::chrono::high_resolution_clock::now();
    }

    void finish_itr(const CounterIdx cntrIdx) {
        _itrDuration[cntrIdx] = std::chrono::high_resolution_clock::now() - _itrStart[cntrIdx];
    }
    void finish_itr(const std::string& itrKey, const std::string& itrNodeShape);

    friend class PerfHelper;
    friend class PerfHelperTotal;
};

class MKLDNNGraph;
std::string perfGetModelInputStr(const MKLDNNGraph& graph);
class MKLDNNExecNetwork;
void perfDump(const MKLDNNExecNetwork& execNet);

class PerfHelper {
    PerfCount& _count;
    const PerfCount::CounterIdx _cntrIdx;

public:
    explicit PerfHelper(PerfCount& count, PerfCount::CounterIdx cntrIdx) :
        _count(count), _cntrIdx(cntrIdx) { _count.start_itr(_cntrIdx); }
    ~PerfHelper() { _count.finish_itr(_cntrIdx); }
};

class MKLDNNNode;
class PerfHelperTotal {
    const std::shared_ptr<MKLDNNNode>& _node;
    PerfCount& _count;
    const std::string& _itrKey;

public:
    explicit PerfHelperTotal(const std::shared_ptr<MKLDNNNode>& node, const std::string& itrKey);
    ~PerfHelperTotal();
};

#endif // CPU_DEBUG_CAPS

#define GET_PERF(_helper, ...) std::unique_ptr<_helper>(new _helper(__VA_ARGS__))
#define PERF_COUNTER(_need, _helper, ...) auto pc = _need ? GET_PERF(_helper, __VA_ARGS__) : nullptr;

#ifdef CPU_DEBUG_CAPS
#define PERF(_node, _need, _itrKey) PERF_COUNTER(_need, PerfHelperTotal, _node, _itrKey)
#define PERF_SHAPE_INFER(_node) PERF_COUNTER(_node->PerfCounter()._isItrStarted, PerfHelper, _node->PerfCounter(), PerfCount::ShapeInfer)
#define PERF_PREPARE_PARAMS(_node) PERF_COUNTER(_node->PerfCounter()._isItrStarted, PerfHelper, _node->PerfCounter(), PerfCount::PrepareParams)
#else
#define PERF(_node, _need, _itrKey) PERF_COUNTER(_need, PerfHelper, _node->PerfCounter())
#define PERF_SHAPE_INFER(_node)
#define PERF_PREPARE_PARAMS(_node)
#endif // CPU_DEBUG_CAPS
}  // namespace MKLDNNPlugin


