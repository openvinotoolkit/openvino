// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <chrono>
#include <utility>
#include <unordered_map>
#include <deque>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <limits>
#include <mutex>
#include <cfloat>

#ifdef ENABLE_PROFILING_ITT
#include <ittnotify.h>
#endif

namespace InferenceEngine {

template< typename Static, typename Block>
void annotateBegin(Static&& static_, Block&& block_);

template< typename Static, typename Block>
void annotateEnd(Static&& static_, Block&& block_);

template< typename Static, typename Block, typename Local>
struct Annotate {
    struct Static_ {
        template<std::size_t...> struct idx{};

        template<std::size_t N, std::size_t... S> struct idx<N, S...> : idx<N-1, N-1, S...> {};

        template<std::size_t... S> struct idx<0, S...> {
            using type = idx<S...>;
        };

        template<typename ArgTuple, std::size_t ...I>
        Static_(ArgTuple&& arg_tuple, idx<I...>)
        : static_{std::get<I>(std::forward<ArgTuple>(arg_tuple))...} {}

        template<typename ArgTuple>
        explicit Static_(ArgTuple&& arg_tuple)
        : Static_{std::forward<ArgTuple>(arg_tuple), typename idx< std::tuple_size<ArgTuple>::value >::type{}} {}

        Static static_;
    };

    static Static_ static_;

    Block   block_;

    Annotate(const Annotate&)               = delete;
    Annotate& operator=(const Annotate&)    = delete;
    Annotate(Annotate&&)                    = default;
    Annotate& operator=(Annotate&&)         = default;

    template<typename ...Ts>
    inline explicit Annotate(Ts&& ...xs)
    : block_{std::forward<Ts>(xs)...} {
        annotateBegin(static_.static_, block_);
    }

    inline ~Annotate() {
        annotateEnd(static_.static_, block_);
    }
};

template< typename Static, typename Block, typename Local>
typename Annotate< Static, Block, Local >::Static_ Annotate< Static, Block, Local >::static_(Local::static_args());

#define IE_ANNOTATE_CONCAT(x, y) IE_ANNOTATE_CONCAT_EVAL(x, y)
#define IE_ANNOTATE_CONCAT_EVAL(x, y)  x ## y

#define IE_ANNOTATE_UNPACK(tuple) IE_ANNOTATE_UNPACK_EVAL tuple
#define IE_ANNOTATE_UNPACK_EVAL(...)  __VA_ARGS__

#define IE_ANNOTATE_MAKE_NAME(lib_name, postfix)        \
    IE_ANNOTATE_CONCAT(                                 \
        IE_ANNOTATE_CONCAT(                             \
            IE_ANNOTATE_CONCAT(__intel_util_annotate_,  \
                                    lib_name),          \
            postfix),                                   \
    __LINE__)

#define IE_ANNOTATE_LOCAL_TYPE_NAME(lib_name)   IE_ANNOTATE_MAKE_NAME(lib_name, _ctx)
#define IE_ANNOTATE_VARIABLE_NAME(lib_name)     IE_ANNOTATE_MAKE_NAME(lib_name, _variable)
#define IE_ANNOTATE_FUNC_NAME(lib_name)         IE_ANNOTATE_MAKE_NAME(lib_name, _func)

#define IE_ANNOTATE_MAKE_SCOPE_TYPE(lib_name, static_type, block_type, make_static_args_tuple)  \
    struct IE_ANNOTATE_LOCAL_TYPE_NAME(lib_name) :                                              \
        ::InferenceEngine::Annotate<                                                            \
            static_type,                                                                        \
            block_type,                                                                         \
            IE_ANNOTATE_LOCAL_TYPE_NAME(lib_name) > {                                           \
            using ::InferenceEngine::Annotate<                                                  \
                static_type,                                                                    \
                block_type,                                                                     \
                IE_ANNOTATE_LOCAL_TYPE_NAME(lib_name) >::Annotate;                              \
            static auto static_args()                                                           \
            ->decltype(std::make_tuple(IE_ANNOTATE_UNPACK(make_static_args_tuple))) {           \
                return std::make_tuple(IE_ANNOTATE_UNPACK(make_static_args_tuple));}            \
        }

#define IE_ANNOTATE_MAKE_SCOPE(lib_name, static_type, block_type, make_static_args_tuple, make_block_args_tuple)    \
    IE_ANNOTATE_MAKE_SCOPE_TYPE(lib_name, static_type, block_type, make_static_args_tuple)                          \
    IE_ANNOTATE_VARIABLE_NAME(lib_name){IE_ANNOTATE_UNPACK(make_block_args_tuple)};

#ifdef ENABLE_PROFILING_ITT
struct IttTaskHandles {
    __itt_domain*        const domain;
    __itt_string_handle* const handle;

    explicit IttTaskHandles(const char* task_name)
    : domain{ __itt_domain_create("InferenceEngine") }
    , handle{ __itt_string_handle_create(task_name) } {}
};

struct IttBlock{};

inline static void annotateBegin(IttTaskHandles& h, IttBlock&) {
    __itt_task_begin(h.domain, __itt_null, __itt_null, h.handle);
}

inline static void annotateEnd(IttTaskHandles& h, IttBlock&) {
    __itt_task_end(h.domain);
}

#define IE_ITT_SCOPE(task_name)                                 \
    IE_ANNOTATE_MAKE_SCOPE(InferenceEngineItt,                  \
                           ::InferenceEngine::IttTaskHandles,   \
                           ::InferenceEngine::IttBlock,         \
                           (task_name), ())
#else
    #define IE_ITT_SCOPE(task_name)
#endif

class TimeResultsMap {
protected:
    std::unordered_map<std::string, std::deque<double> > m_map;
    std::mutex mutex;

public:
    inline void add(const  std::string& name, double val) {
        std::unique_lock<std::mutex> lock(mutex);
        m_map[name].push_back(val);
    }

    inline ~TimeResultsMap() {
        for (auto && iter : m_map) {
            const size_t num = iter.second.size();
            double valSum = 0, valMin = (std::numeric_limits<double>::max)(), valMax = std::numeric_limits<double>::lowest(), logSum = 0;
            int logCount = 0;
            for (auto val : iter.second) {
                if (val > 0) {
                    logCount++;
                    logSum += std::log(val);
                }
                valSum += val;
                valMin = std::fmin(val, valMin);
                valMax = std::fmax(val, valMax);
            }

            std::cout << std::setw(20) << iter.first << " collected by " << std::setw(8) << num << " samples, ";
            std::cout << "mean " << std::setw(12) << (valSum / num)/1000000 << " ms, ";
            std::cout << "geomean " << std::setw(12) << (logCount ? std::exp(logSum / logCount) : 0)/1000000 << " ms, ";
            std::cout << "min " << std::setw(12) << valMin/1000000 << " ms, ";
            std::cout << "max " << std::setw(12) << valMax/1000000 << " ms" << std::endl;
        }
    }
};

struct TimeSampler {
    using Clock = std::chrono::high_resolution_clock;

    std::string name;

    Clock::time_point t;
};

inline static void annotateBegin(TimeResultsMap&, TimeSampler& t) {
    t.t = TimeSampler::Clock::now();
}

inline static void annotateEnd(TimeResultsMap& m, TimeSampler& t) {
    m.add(t.name, std::chrono::duration_cast<std::chrono::nanoseconds>(TimeSampler::Clock::now() - t.t).count());
}

#if ENABLE_PROFILING_RAW
    #define IE_TIMER_SCOPE(timerName)           \
        IE_ANNOTATE_MAKE_SCOPE(                 \
            InferenceEngineTimer,               \
            ::InferenceEngine::TimeResultsMap,  \
            ::InferenceEngine::TimeSampler,     \
            (),                                 \
            (timerName))
#else
    #define IE_TIMER_SCOPE(timerName)
#endif

#define IE_STR(x) IE_STR_(x)
#define IE_STR_(x) #x

#define IE_PROFILING_AUTO_SCOPE(NAME) IE_ITT_SCOPE(IE_STR(NAME)); IE_TIMER_SCOPE(IE_STR(NAME))

struct ProfilingTask {
    std::string name;

#ifdef ENABLE_PROFILING_ITT
    __itt_domain*        domain;
    __itt_string_handle* handle;
#endif

    ProfilingTask() = default;
    ProfilingTask(const ProfilingTask&) = default;

    inline explicit ProfilingTask(const std::string& task_name)
    : name(task_name)
#ifdef ENABLE_PROFILING_ITT
    , domain(__itt_domain_create("InferenceEngine"))
    , handle(__itt_string_handle_create(task_name.c_str()))
#endif
    {}
};

struct IttStatic{};

struct IttProfilingTask {
    ProfilingTask& t;
};

inline static void annotateBegin(IttStatic&, IttProfilingTask& t) {
#ifdef ENABLE_PROFILING_ITT
    __itt_task_begin(t.t.domain, __itt_null, __itt_null, t.t.handle);
#endif
}

inline static void annotateEnd(IttStatic&, IttProfilingTask& t) {
#ifdef ENABLE_PROFILING_ITT
    __itt_task_end(t.t.domain);
#endif
}

#ifdef ENABLE_PROFILING_ITT
    #define IE_ITT_TASK_SCOPE(profilingTask)            \
        IE_ANNOTATE_MAKE_SCOPE(                         \
            InferenceEngineIttScopeTask,                \
            ::InferenceEngine::IttStatic,               \
            ::InferenceEngine::IttProfilingTask,        \
            (),                                         \
            (profilingTask))
#else
    #define IE_ITT_TASK_SCOPE(profiling_task)
#endif

#define IE_PROFILING_AUTO_SCOPE_TASK(PROFILING_TASK) IE_ITT_TASK_SCOPE(PROFILING_TASK); IE_TIMER_SCOPE(PROFILING_TASK.name)

inline static void annotateSetThreadName(const char* name) {
    #ifdef ENABLE_PROFILING_ITT
    __itt_thread_set_name(name);
    #endif
    // to suppress "unused" warning
    (void)(name);
}
}  // namespace InferenceEngine
