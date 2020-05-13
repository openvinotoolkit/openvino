// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines API to profile your plugin using Intel VTune.
 * @file ie_profiling.hpp
 */

#pragma once

#include <cfloat>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#ifdef ENABLE_PROFILING_ITT
#include <ittnotify.h>
#endif

/**
 * @cond
 */

namespace InferenceEngine {

template <typename Static, typename Block>
void annotateBegin(Static&& static_, Block&& block_);

template <typename Static, typename Block>
void annotateEnd(Static&& static_, Block&& block_);

template <typename Static, typename Block, typename Local>
struct Annotate {
    struct Static_ {
        template <std::size_t...>
        struct idx {};

        template <std::size_t N, std::size_t... S>
        struct idx<N, S...> : idx<N - 1, N - 1, S...> {};

        template <std::size_t... S>
        struct idx<0, S...> {
            using type = idx<S...>;
        };

        template <typename ArgTuple, std::size_t... I>
        Static_(ArgTuple&& arg_tuple, idx<I...>): static_ {std::get<I>(std::forward<ArgTuple>(arg_tuple))...} {}

        template <typename ArgTuple>
        explicit Static_(ArgTuple&& arg_tuple)
            : Static_ {std::forward<ArgTuple>(arg_tuple), typename idx<std::tuple_size<ArgTuple>::value>::type {}} {}

        Static static_;
    };

    static Static_ static_;

    Block block_;

    Annotate(const Annotate&) = delete;
    Annotate& operator=(const Annotate&) = delete;
    Annotate(Annotate&&) = default;
    Annotate& operator=(Annotate&&) = default;

    template <typename... Ts>
    inline explicit Annotate(Ts&&... xs): block_ {std::forward<Ts>(xs)...} {
        annotateBegin(static_.static_, block_);
    }

    inline ~Annotate() {
        annotateEnd(static_.static_, block_);
    }
};

template <typename Static, typename Block, typename Local>
typename Annotate<Static, Block, Local>::Static_ Annotate<Static, Block, Local>::static_(Local::staticArgs());

#define IE_ANNOTATE_CONCAT(x, y) IE_ANNOTATE_CONCAT_EVAL(x, y)
#define IE_ANNOTATE_CONCAT_EVAL(x, y) x##y

#define IE_ANNOTATE_UNPACK(tuple) IE_ANNOTATE_UNPACK_EVAL tuple
#define IE_ANNOTATE_UNPACK_EVAL(...) __VA_ARGS__

#define IE_ANNOTATE_MAKE_NAME(lib_name, postfix) \
    IE_ANNOTATE_CONCAT(IE_ANNOTATE_CONCAT(IE_ANNOTATE_CONCAT(__intel_util_annotate_, lib_name), postfix), __LINE__)

#define IE_ANNOTATE_LOCAL_TYPE_NAME(lib_name) IE_ANNOTATE_MAKE_NAME(lib_name, _ctx)
#define IE_ANNOTATE_VARIABLE_NAME(lib_name) IE_ANNOTATE_MAKE_NAME(lib_name, _variable)
#define IE_ANNOTATE_FUNC_NAME(lib_name) IE_ANNOTATE_MAKE_NAME(lib_name, _func)

#define IE_ANNOTATE_MAKE_SCOPE_TYPE(lib_name, static_type, block_type, make_static_args_tuple)                       \
    struct IE_ANNOTATE_LOCAL_TYPE_NAME(lib_name)                                                                     \
        : ::InferenceEngine::Annotate<static_type, block_type, IE_ANNOTATE_LOCAL_TYPE_NAME(lib_name)> {              \
        using ::InferenceEngine::Annotate<static_type, block_type, IE_ANNOTATE_LOCAL_TYPE_NAME(lib_name)>::Annotate; \
        static auto staticArgs() -> decltype(std::make_tuple(IE_ANNOTATE_UNPACK(make_static_args_tuple))) {         \
            return std::make_tuple(IE_ANNOTATE_UNPACK(make_static_args_tuple));                                      \
        }                                                                                                            \
    }

#define IE_ANNOTATE_MAKE_SCOPE(lib_name, static_type, block_type, make_static_args_tuple, make_block_args_tuple) \
    IE_ANNOTATE_MAKE_SCOPE_TYPE(lib_name, static_type, block_type, make_static_args_tuple)                       \
    IE_ANNOTATE_VARIABLE_NAME(lib_name) {IE_ANNOTATE_UNPACK(make_block_args_tuple)};

#ifdef ENABLE_PROFILING_ITT
struct IttTaskHandles {
    __itt_domain* const domain;
    __itt_string_handle* const handle;

    explicit IttTaskHandles(const char* taskName)
        : domain {__itt_domain_create("InferenceEngine")}, handle {__itt_string_handle_create(taskName)} {}
};

struct IttBlock {};

inline static void annotateBegin(IttTaskHandles& h, IttBlock&) {
    __itt_task_begin(h.domain, __itt_null, __itt_null, h.handle);
}

inline static void annotateEnd(IttTaskHandles& h, IttBlock&) {
    __itt_task_end(h.domain);
}

#define IE_ITT_SCOPE(task_name)                                                                                \
    IE_ANNOTATE_MAKE_SCOPE(InferenceEngineItt, ::InferenceEngine::IttTaskHandles, ::InferenceEngine::IttBlock, \
                           (task_name), ())
#else
#define IE_ITT_SCOPE(task_name)
#endif

#define IE_STR(x) IE_STR_(x)
#define IE_STR_(x) #x

struct ProfilingTask;

struct IttStatic {};

struct IttProfilingTask {
    ProfilingTask* t;
};

static void annotateBegin(IttStatic&, IttProfilingTask& t);
static void annotateEnd(IttStatic&, IttProfilingTask& t);

/**
 * @endcond
 */

/**
 * @class ProfilingTask
 * @ingroup ie_dev_profiling
 * @brief Used to annotate section of code which would be named at runtime
 */
struct ProfilingTask {
    ProfilingTask() = default;
    //! @private
    ProfilingTask(const ProfilingTask&) = default;

    ProfilingTask& operator =(const ProfilingTask&) = default;

    /**
    * @brief Construct ProfilingTask with runtime defined name
    */
    inline explicit ProfilingTask(const std::string& taskName)
        : name(taskName)
#ifdef ENABLE_PROFILING_ITT
          ,
          domain(__itt_domain_create("InferenceEngine")),
          handle(__itt_string_handle_create(taskName.c_str()))
#endif
    {
    }

private:
    friend void annotateBegin(IttStatic&, IttProfilingTask& t);
    friend void annotateEnd(IttStatic&, IttProfilingTask& t);

    std::string name;
#ifdef ENABLE_PROFILING_ITT
    __itt_domain* domain;
    __itt_string_handle* handle;
#endif
};

/**
 * @cond
 */

inline static void annotateBegin(IttStatic&, IttProfilingTask& t) {
#ifdef ENABLE_PROFILING_ITT
    __itt_task_begin(t.t->domain, __itt_null, __itt_null, t.t->handle);
#else
    (void)t;
#endif
}

inline static void annotateEnd(IttStatic&, IttProfilingTask& t) {
#ifdef ENABLE_PROFILING_ITT
    __itt_task_end(t.t->domain);
#else
    (void)t;
#endif
}

#ifdef ENABLE_PROFILING_ITT
#define IE_ITT_TASK_SCOPE(profilingTask)                                              \
    IE_ANNOTATE_MAKE_SCOPE(InferenceEngineIttScopeTask, ::InferenceEngine::IttStatic, \
                           ::InferenceEngine::IttProfilingTask, (), (&(profilingTask)))
#else
#define IE_ITT_TASK_SCOPE(profiling_task)
#endif

inline static void annotateSetThreadName(const char* name) {
#ifdef ENABLE_PROFILING_ITT
    __itt_thread_set_name(name);
#else
    (void)(name);
#endif
}

/**
 * @endcond
 */


/**
 * @def IE_PROFILING_AUTO_SCOPE(NAME)
 * @ingroup ie_dev_profiling
 * @brief Annotate section of code till scope exit to be profiled using known at compile time @p NAME as section id
 * @param NAME Known at compile time name of section of code that is passed to profiling back end
 */
#define IE_PROFILING_AUTO_SCOPE(NAME) IE_ITT_SCOPE(IE_STR(NAME));


/**
 * @def IE_PROFILING_AUTO_SCOPE_TASK(PROFILING_TASK)
 * @ingroup ie_dev_profiling
 * @brief Annotate section of code till scope exit to be profiled runtime configured variable of ProfilingTask type.
 *        ProfilingTask::name will be used as section id.
 * @param PROFILING_TASK variable of ProfilingTask type
 */
#define IE_PROFILING_AUTO_SCOPE_TASK(PROFILING_TASK) IE_ITT_TASK_SCOPE(PROFILING_TASK);

}  // namespace InferenceEngine
