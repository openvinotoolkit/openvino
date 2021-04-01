// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <set>


#include <vpu/utils/extra.hpp>
#include <vpu/utils/logger.hpp>
#include <vpu/utils/ie_itt.hpp>

namespace vpu {

#ifdef ENABLE_PROFILING_RAW

struct CompileEnv;

class Profiler final {
public:
    class Section final {
    public:
        explicit Section(const std::string& name);
        ~Section();

    private:
        using Time = std::chrono::high_resolution_clock;
        using TimePoint = Time::time_point;
        using MilliSecondsFP64 = std::chrono::duration<double, std::milli>;

    private:
        const CompileEnv* _env = nullptr;
        TimePoint _start;
    };

public:
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    Profiler(Profiler&&) = delete;
    Profiler& operator=(Profiler&&) = delete;

private:
    Profiler();
    ~Profiler();

    inline void setLogger(const Logger::Ptr& log) { _log = log; }

    void startSection(const std::string& name);
    void endSection(double dur);

private:
    struct Node final {
        std::string name;
        std::vector<double> timings;

        size_t parentInd = 0;
        std::set<size_t> childInds;
        std::unordered_map<std::string, size_t> childNames;

        inline Node() = default;
        inline explicit Node(const std::string& name) : name(name) {}

        inline ~Node() = default;

        inline Node(const Node&) = default;
        inline Node& operator=(const Node&) = default;

        inline Node(Node&&) = default;
        inline Node& operator=(Node&&) = default;
    };

private:
    size_t getMaxNameLength(const Node& node) const;
    void print(const Node& node, int nameWidth) const;

private:
    Logger::Ptr _log;

    std::vector<Node> _nodes;
    size_t _curNodeInd = 0;

    friend class Section;
    friend struct CompileEnv;
};

#define VPU_PROFILER_SECTION(name) vpu::Profiler::Section VPU_COMBINE(profileSec, __LINE__) (# name)
#define VPU_PROFILE(name) OV_ITT_SCOPED_TASK(vpu::itt::domains::VPU, "VPU_" #name); VPU_PROFILER_SECTION(name)

#else

#define VPU_PROFILE(name) OV_ITT_SCOPED_TASK(vpu::itt::domains::VPU, "VPU_" #name);

#endif  // ENABLE_PROFILING_RAW

}  // namespace vpu
