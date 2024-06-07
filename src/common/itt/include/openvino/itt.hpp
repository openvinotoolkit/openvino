// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines API to trace using Intel ITT.
 * @file itt.hpp
 */

#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "openvino/function_name.hpp"
#include "openvino/util/pp.hpp"

/** @ingroup ov_dev_profiling
 * @brief openvino namespace
 */
namespace openvino {
namespace itt {
/**
 * @typedef domain_t
 * @ingroup ov_dev_profiling
 * @brief A domain type which enables tagging trace data for different modules or libraries in a program.
 */
typedef struct domain_ {
}* domain_t;

/**
 * @typedef handle_t
 * @ingroup ov_dev_profiling
 * @brief Annotation handle for section of code which would be named at runtime.
 */
typedef struct handle_ {
}* handle_t;

/**
 * @cond
 */
namespace internal {
domain_t domain(char const* name);
handle_t handle(char const* name);
void taskBegin(domain_t d, handle_t t);
void taskEnd(domain_t d);
void threadName(const char* name);
}  // namespace internal
/**
 * @endcond
 */

/**
 * @fn void threadName(const char* name)
 * @ingroup ov_dev_profiling
 * @brief Set thread name using a char string.
 * @param name [in] The thread name
 */
inline void threadName(const char* name) {
    internal::threadName(name);
}

inline void threadName(const std::string& name) {
    internal::threadName(name.c_str());
}

inline handle_t handle(char const* name) {
    return internal::handle(name);
}

inline handle_t handle(const std::string& name) {
    return internal::handle(name.c_str());
}

/**
 * @fn handle_t handle(char const *name)
 * @ingroup ov_dev_profiling
 * @brief Create annotation handle with a given name.
 * @details If template function is instantiated with a tag, the handle is created as a singleton.
 * @param name [in] The annotation name
 */
template <typename Tag>
handle_t handle(char const* name) {
    static auto h = internal::handle(name);
    return h;
}

template <typename Tag>
handle_t handle(const std::string& name) {
    return handle<Tag>(name.c_str());
}

template <typename Tag>
handle_t handle(handle_t h) {
    return h;
}

/**
 * @class ScopedTask
 * @ingroup ov_dev_profiling
 * @brief Used to annotate section of code which would be named at runtime
 * @tparam The @p domain parameter is domain type which shoud be defined with OV_ITT_DOMAIN() macro.
 */
template <domain_t (*domain)()>
struct ScopedTask {
    /**
     * @brief Construct ScopedTask with defined annotation handle
     */
    ScopedTask(handle_t taskHandle) noexcept {
        internal::taskBegin(domain(), taskHandle);
    }

    /**
     * @brief The ScopedTask destructor closes or ends the task scope
     */
    ~ScopedTask() noexcept {
        internal::taskEnd(domain());
    }

    ScopedTask(const ScopedTask&) = delete;
    ScopedTask& operator=(const ScopedTask&) = delete;
};

/**
 * @class TaskChain
 * @ingroup ov_dev_profiling
 * @brief Used to annotate a sequence of sections of code which would be named at runtime
 * @tparam The @p domain parameter is domain type which shoud be defined with OV_ITT_DOMAIN() macro.
 */
template <domain_t (*domain)()>
class TaskChain {
    uint32_t _id = 1;
    std::string _prefix;
    bool _skipped{};

    TaskChain(const TaskChain&) = delete;
    TaskChain& operator=(const TaskChain&) = delete;

public:
    /**
     * @brief Construct TaskChain with defined annotation handle
     */
    TaskChain(handle_t taskHandle, std::string&& prefix) noexcept : _prefix(std::forward<std::string>(prefix)) {
        internal::taskBegin(domain(), taskHandle);
    }

    /**
     * @brief The TaskChain destructor closes or ends the task scope
     */
    ~TaskChain() noexcept {
        skip();
    }

    /**
     * @brief Ends the previous task from the chain and starts a new one with the given annotation handle
     */
    void next(handle_t taskHandle) {
        if (_skipped)
            _skipped = false;
        else
            internal::taskEnd(domain());
        internal::taskBegin(domain(), taskHandle);
        ++_id;
    }

    /*
     * @brief Generating a task name using a sequence number.
     */
    std::string taskName() const {
        return _prefix + "_" + std::to_string(_id);
    }

    /*
     * @brief Generating a task name using a scope name.
     */
    std::string taskNameOrHandle(const std::string& name) const {
        return _prefix + "_" + name;
    }

    /*
     * @brief Returns a handle provided as argument.
     */
    handle_t taskNameOrHandle(handle_t handle) const {
        return handle;
    }

    /*
     * @brief Skips the remaining task scope.
     */
    void skip() {
        if (!_skipped) {
            _skipped = true;
            internal::taskEnd(domain());
        }
    }
};

/**
 * @def OV_ITT_DOMAIN(domainName)
 * @ingroup ov_dev_profiling
 * @brief Declare domain with a given name.
 * @param domainName [in] Known at compile time name of module or library (the domain name).
 * @param domainDisplayName [in] Domain name used as the ITT counter name and displayed in Intel VTune. Parameter is
 * optional.
 */
#define OV_ITT_DOMAIN(...) OV_PP_OVERLOAD(OV_ITT_DOMAIN, __VA_ARGS__)

#define OV_ITT_GROUP(group) OV_PP_CAT(ENABLE_PROFILING_, group)

/**
 * @cond
 */

#define OV_ITT_DOMAIN_1(domainName)                                   \
    inline openvino::itt::domain_t domainName() noexcept {            \
        static auto d = openvino::itt::internal::domain(#domainName); \
        return d;                                                     \
    }

#define OV_ITT_DOMAIN_2(domainName, domainDisplayName)                      \
    inline openvino::itt::domain_t domainName() noexcept {                  \
        static auto d = openvino::itt::internal::domain(domainDisplayName); \
        return d;                                                           \
    }

/**
 * @endcond
 */

/**
 * @def OV_ITT_SCOPE(domain, handleOrTaskName)
 * @ingroup ov_dev_profiling
 * @brief Annotate section of code till scope exit to be profiled using known @p handle or @p taskName as section id.
 * @details In case if handle or taskName absent, the current function name is used.
 * @param group [in] ITT counter group name used for enabling/disabling at compile time.
 * @param domainName [in] Known at compile time name of module or library (the domain name).
 * @param handleOrTaskName [in] The annotation name or handle for section of code. Parameter is optional.
 */
#define OV_ITT_SCOPE(group, ...) \
    OV_PP_EXPAND(OV_PP_CAT(OV_ITT_SCOPE_IMPL_, OV_PP_IS_ENABLED(OV_ITT_GROUP(group)))(__VA_ARGS__))

/**
 * @cond
 */

#define OV_ITT_SCOPE_IMPL_0(...)
#define OV_ITT_SCOPE_IMPL_1(...) OV_PP_OVERLOAD(OV_ITT_SCOPE, __VA_ARGS__)

#define OV_ITT_SCOPE_1(domain)                                            \
    openvino::itt::ScopedTask<domain> OV_PP_CAT(ittScopedTask, __LINE__)( \
        openvino::itt::handle<struct OV_PP_CAT(Task, __LINE__)>(ITT_FUNCTION_NAME));

#define OV_ITT_SCOPE_2(domain, taskOrTaskName)                            \
    openvino::itt::ScopedTask<domain> OV_PP_CAT(ittScopedTask, __LINE__)( \
        openvino::itt::handle<struct OV_PP_CAT(Task, __LINE__)>(taskOrTaskName));

/**
 * @endcond
 */

/**
 * @def OV_ITT_SCOPED_TASK(domain, handleOrTaskName)
 * @ingroup ov_dev_profiling
 * @brief Annotate section of code till scope exit to be profiled using known @p handle or @p taskName as section id.
 * @details In case if handle or taskName absent, the current function name is used.
 * @param domainName [in] Known at compile time name of module or library (the domain name).
 * @param handleOrTaskName [in] The annotation name or handle for section of code. Parameter is optional.
 */
#define OV_ITT_SCOPED_TASK(...) OV_ITT_SCOPE(ALL, __VA_ARGS__)

/**
 * @def OV_ITT_TASK_CHAIN(chainId, domain, prefix, taskName)
 * @ingroup ov_dev_profiling
 * @brief Begins the sequrence of an annotated sections of code using @p prefix and @p taskName as section id.
 * @details In case if prefix absent, the current function name is used,
 *          if taskName absent, the first chain index is used, i.e 1.
 * @param group [in] ITT counter group name used for enabling/disabling at compile time.
 * @param chainId [in] The tasks chain identifier.
 * @param domainName [in] Known at compile time name of module or library (the domain name).
 * @param prefix [in] The task chain name prefix. The task name starts with this prefix. Parameter is optional.
 * @param taskName [in] The annotation name for section of code. Parameter is optional.
 */
#define OV_ITT_SCOPE_CHAIN(group, ...) \
    OV_PP_EXPAND(OV_PP_CAT(OV_ITT_SCOPE_CHAIN_IMPL_, OV_PP_IS_ENABLED(OV_ITT_GROUP(group)))(__VA_ARGS__))

/**
 * @cond
 */

#define OV_ITT_SCOPE_CHAIN_IMPL_0(...)
#define OV_ITT_SCOPE_CHAIN_IMPL_1(...) OV_PP_OVERLOAD(OV_ITT_SCOPE_CHAIN, __VA_ARGS__)

#define OV_ITT_SCOPE_CHAIN_2(chainId, domain)                                                           \
    openvino::itt::TaskChain<domain> chainId(                                                           \
        openvino::itt::handle<struct OV_PP_CAT(Task, __LINE__)>(std::string(ITT_FUNCTION_NAME) + "_1"), \
        ITT_FUNCTION_NAME);

#define OV_ITT_SCOPE_CHAIN_3(chainId, domain, prefix)                                        \
    openvino::itt::TaskChain<domain> chainId(                                                \
        openvino::itt::handle<struct OV_PP_CAT(Task, __LINE__)>(std::string(prefix) + "_1"), \
        prefix);

#define OV_ITT_SCOPE_CHAIN_4(chainId, domain, prefix, taskName)                                        \
    openvino::itt::TaskChain<domain> chainId(                                                          \
        openvino::itt::handle<struct OV_PP_CAT(Task, __LINE__)>(std::string(prefix) + "_" + taskName), \
        prefix);

/**
 * @endcond
 */

/**
 * @def OV_ITT_SCOPE_NEXT(group, chainId, taskName)
 * @ingroup ov_dev_profiling
 * @brief Inserts new annotated section of code to tasks chain using @p taskName as section id.
 * @details If taskName is missing, the current chain index is used.
 * @param group [in] ITT counter group name used for enabling/disabling at compile time.
 * @param chainId [in] The tasks chain identifier.
 * @param taskOrTaskName [in] The annotation name or handle for section of code. Parameter is optional.
 */
#define OV_ITT_SCOPE_NEXT(group, ...) \
    OV_PP_EXPAND(OV_PP_CAT(OV_ITT_SCOPE_NEXT_IMPL_, OV_PP_IS_ENABLED(OV_ITT_GROUP(group)))(__VA_ARGS__))

/**
 * @cond
 */

#define OV_ITT_SCOPE_NEXT_IMPL_0(...)
#define OV_ITT_SCOPE_NEXT_IMPL_1(...) OV_PP_OVERLOAD(OV_ITT_SCOPE_NEXT, __VA_ARGS__)

#define OV_ITT_SCOPE_NEXT_1(chainId) \
    chainId.next(openvino::itt::handle<struct OV_PP_CAT(Task, __LINE__)>(chainId.taskName()));

#define OV_ITT_SCOPE_NEXT_2(chainId, taskOrTaskName) \
    chainId.next(openvino::itt::handle<struct OV_PP_CAT(Task, __LINE__)>(chainId.taskNameOrHandle(taskOrTaskName)));

/**
 * @endcond
 */

/**
 * @def OV_ITT_SCOPE_SKIP(group, chainId)
 * @ingroup ov_dev_profiling
 * @brief Skips the remaining task scope.
 * @param group [in] ITT counter group name used for enabling/disabling at compile time.
 * @param chainId [in] The tasks chain identifier.
 */
#define OV_ITT_SCOPE_SKIP(group, chainId) \
    OV_PP_EXPAND(OV_PP_CAT(OV_ITT_SCOPE_SKIP_, OV_PP_IS_ENABLED(OV_ITT_GROUP(group)))(chainId))

/**
 * @cond
 */

#define OV_ITT_SCOPE_SKIP_0(chainId)
#define OV_ITT_SCOPE_SKIP_1(chainId) chainId.skip();

/**
 * @endcond
 */

/**
 * @def OV_ITT_TASK_CHAIN(chainId, domain, prefix, taskName)
 * @ingroup ov_dev_profiling
 * @brief Begins the sequrence of an annotated sections of code using @p prefix and @p taskName as section id.
 * @details In case if prefix absent, the current function name is used,
 *          if taskName absent, the first chain index is used, i.e 1.
 * @param chainId [in] The tasks chain identifier.
 * @param domainName [in] Known at compile time name of module or library (the domain name).
 * @param prefix [in] The task chain name prefix. The task name starts with this prefix. Parameter is optional.
 * @param taskName [in] The annotation name for section of code. Parameter is optional.
 */
#define OV_ITT_TASK_CHAIN(...) OV_ITT_SCOPE_CHAIN(ALL, __VA_ARGS__)

/**
 * @def OV_ITT_TASK_NEXT(chainId, taskName)
 * @ingroup ov_dev_profiling
 * @brief Inserts new annotated section of code to tasks chain using @p taskName as section id.
 * @details If taskName is missing, the current chain index is used.
 * @param chainId [in] The tasks chain identifier.
 * @param taskOrTaskName [in] The annotation name or handle for section of code. Parameter is optional.
 */
#define OV_ITT_TASK_NEXT(...) OV_ITT_SCOPE_NEXT(ALL, __VA_ARGS__)

/**
 * @def OV_ITT_TASK_SKIP(chainId)
 * @ingroup ov_dev_profiling
 * @brief Skips the remaining task scope.
 * @param chainId [in] The tasks chain identifier.
 */
#define OV_ITT_TASK_SKIP(chainId) OV_ITT_SCOPE_SKIP(ALL, chainId);

}  // namespace itt
}  // namespace openvino
