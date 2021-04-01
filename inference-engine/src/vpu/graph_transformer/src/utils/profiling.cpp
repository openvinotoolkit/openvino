// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/profiling.hpp>

#include <cmath>
#include <iomanip>
#include <limits>
#include <string>
#include <algorithm>

#include <vpu/compile_env.hpp>

namespace vpu {

#ifdef ENABLE_PROFILING_RAW

Profiler::Section::Section(const std::string& name) {
    _env = CompileEnv::getOrNull();
    if (_env == nullptr) {
        return;
    }

    if (!_env->log->isActive(LogLevel::Info)) {
        _env = nullptr;
        return;
    }

    _start = Time::now();
    _env->profile.startSection(name);
}

Profiler::Section::~Section() {
    if (_env == nullptr) {
        return;
    }

    auto end = Time::now();
    auto dur = std::chrono::duration_cast<MilliSecondsFP64>(end - _start).count();

    _env->profile.endSection(dur);
}

Profiler::Profiler() {
    _nodes.reserve(128);
    _nodes.emplace_back(Node());
}

size_t Profiler::getMaxNameLength(const Node& node) const {
    size_t maxNameLength = node.name.length();
    for (auto ind : node.childInds) {
        maxNameLength = std::max(maxNameLength, getMaxNameLength(_nodes[ind]) + Logger::IDENT_SIZE);
    }
    return maxNameLength;
}

void Profiler::print(const Node& node, int nameWidth) const {
    assert(!node.timings.empty());

    const int numWidth = 5;
    const int timeWidth = 10;

    const auto valNum = node.timings.size();

    double valSum = 0;
    double valMin = std::numeric_limits<double>::max();
    double valMax = std::numeric_limits<double>::lowest();

    for (auto val : node.timings) {
        valSum += val;
        valMin = std::min(val, valMin);
        valMax = std::max(val, valMax);
    }

    _log->info("%f%f%s : total %f%f%v ms, %f%f%v samples, mean %f%f%v ms, min %f%f%v ms, max %f%f%v ms",
        std::setw(nameWidth), std::left,  node.name,
        std::setw(timeWidth), std::right, valSum,
        std::setw(numWidth),  std::right, valNum,
        std::setw(timeWidth), std::right, valSum / valNum,
        std::setw(timeWidth), std::right, valMin,
        std::setw(timeWidth), std::right, valMax);
    VPU_LOGGER_SECTION(_log);

    for (auto ind : node.childInds) {
        print(_nodes[ind], nameWidth - Logger::IDENT_SIZE);
    }
}

Profiler::~Profiler() {
    if (!_log->isActive(LogLevel::Info)) {
        return;
    }

    assert(_curNodeInd == 0);
    const auto& rootNode = _nodes[0];
    assert(rootNode.name.empty());
    assert(rootNode.timings.empty());

    _log->info("Profiling");
    VPU_LOGGER_SECTION(_log);

    const int nameWidth = static_cast<int>(getMaxNameLength(rootNode));
    for (auto ind : rootNode.childInds) {
        print(_nodes[ind], nameWidth);
    }
}

void Profiler::startSection(const std::string& name) {
    IE_ASSERT(_curNodeInd < _nodes.size());
    auto& curNode = _nodes[_curNodeInd];

    auto it = curNode.childNames.find(name);
    if (it != curNode.childNames.end()) {
        _curNodeInd = it->second;
    } else {
        _nodes.emplace_back(name);

        auto& newNode = _nodes.back();
        auto newNodeInd = _nodes.size() - 1;

        newNode.parentInd = _curNodeInd;
        curNode.childInds.emplace(newNodeInd);
        curNode.childNames.emplace(name, newNodeInd);

        _curNodeInd = newNodeInd;
    }
}

void Profiler::endSection(double dur) {
    IE_ASSERT(_curNodeInd < _nodes.size());
    auto& curNode = _nodes[_curNodeInd];

    curNode.timings.push_back(dur);

    _curNodeInd = curNode.parentInd;
}

#endif  // ENABLE_PROFILING_RAW

}  // namespace vpu
