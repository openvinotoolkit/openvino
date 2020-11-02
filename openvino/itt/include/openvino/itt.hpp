//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

/**
 * @brief Defines API to trace using Intel ITT.
 * @file itt.hpp
 */

#pragma once
#include <openvino/function_name.hpp>
#include <openvino/macro_overload.hpp>
#include <string>

namespace openvino
{
    namespace itt
    {
        /**
         * @typedef domain_t
         * @ingroup ie_dev_profiling
         * @brief A domain type which enables tagging trace data for different modules or libraries in a program.
         */
        typedef struct domain_ {} *domain_t;

        /**
         * @typedef handle_t
         * @ingroup ie_dev_profiling
         * @brief Annotation handle for section of code which would be named at runtime.
         */
        typedef struct handle_ {} *handle_t;

/**
 * @cond
 */
        namespace internal
        {
            domain_t domain(char const* name);
            handle_t handle(char const* name);
            void taskBegin(domain_t d, handle_t t);
            void taskEnd(domain_t d);
            void threadName(const char* name);
        }
/**
 * @endcond
 */

        /**
         * @fn void threadName(const char* name)
         * @ingroup ie_dev_profiling
         * @brief Set thread name using a char string.
         * @param name [in] The thread name
         */
        inline void threadName(const char* name)
        {
            internal::threadName(name);
        }

        inline void threadName(const std::string &name)
        {
            internal::threadName(name.c_str());
        }

        inline handle_t handle(char const *name)
        {
            return internal::handle(name);
        }

        inline handle_t handle(const std::string &name)
        {
            return internal::handle(name.c_str());
        }

        /**
         * @fn handle_t handle(char const *name)
         * @ingroup ie_dev_profiling
         * @brief Create annotation handle with a given name.
         * @details If template function is instantiated with a tag, the handle is created as a singleton.
         * @param name [in] The annotation name
         */
        template <typename Tag>
        handle_t handle(char const *name)
        {
            static auto h = internal::handle(name);
            return h;
        }

        template <typename Tag>
        handle_t handle(const std::string &name)
        {
            return handle<Tag>(name.c_str());
        }

        template <typename Tag>
        handle_t handle(handle_t h)
        {
            return h;
        }

        /**
         * @class ScopedTask
         * @ingroup ie_dev_profiling
         * @brief Used to annotate section of code which would be named at runtime
         * @tparam The @p domain parameter is domain type which shoud be defined with OV_ITT_DOMAIN() macro.
         */
        template <domain_t(*domain)()>
        struct ScopedTask
        {
            /**
             * @brief Construct ScopedTask with defined annotation handle
             */
            ScopedTask(handle_t taskHandle) noexcept
            {
                internal::taskBegin(domain(), taskHandle);
            }
            ~ScopedTask() noexcept { internal::taskEnd(domain()); }
            ScopedTask(const ScopedTask&) = delete;
            ScopedTask& operator=(const ScopedTask&) = delete;
        };

/**
 * @def OV_ITT_DOMAIN(domainName)
 * @ingroup ie_dev_profiling
 * @brief Declare domain with a given name.
 * @param domainName [in] Known at compile time name of module or library (the domain name).
 * @param domainDisplayName [in] Domain name used as the ITT counter name and displayed in Intel VTune. Parameter is optional.
 */
#define OV_ITT_DOMAIN(...) OV_ITT_MACRO_OVERLOAD(OV_ITT_DOMAIN, __VA_ARGS__)

/**
 * @cond
 */

#define OV_ITT_DOMAIN_1(domainName)                                                                 \
inline openvino::itt::domain_t domainName() noexcept                                                \
{                                                                                                   \
    static auto d = openvino::itt::internal::domain(#domainName);                                   \
    return d;                                                                                       \
}

#define OV_ITT_DOMAIN_2(domainName, domainDisplayName)                                              \
inline openvino::itt::domain_t domainName() noexcept                                                \
{                                                                                                   \
    static auto d = openvino::itt::internal::domain(domainDisplayName);                             \
    return d;                                                                                       \
}

/**
 * @endcond
 */

/**
 * @def OV_ITT_SCOPED_TASK(domain, handleOrTaskName)
 * @ingroup ie_dev_profiling
 * @brief Annotate section of code till scope exit to be profiled using known @p handle or @p taskName as section id.
 * @details In case if handle or taskName absent, the current function name is used.
 * @param domainName [in] Known at compile time name of module or library (the domain name).
 * @param handleOrTaskName [in] The annotation name or handle for section of code. Parameter is optional.
 */
#define OV_ITT_SCOPED_TASK(...) OV_ITT_MACRO_OVERLOAD(OV_ITT_SCOPED_TASK, __VA_ARGS__)

/**
 * @cond
 */

#define OV_ITT_SCOPED_TASK_1(domain)                                                                \
        struct Task ## __LINE__ {};                                                                 \
        openvino::itt::ScopedTask<domain> ittScopedTask ## __LINE__                                 \
                    (openvino::itt::handle<Task ## __LINE__>(ITT_FUNCTION_NAME));

#define OV_ITT_SCOPED_TASK_2(domain, taskOrTaskName)                                                \
        struct Task ## __LINE__ {};                                                                 \
        openvino::itt::ScopedTask<domain> ittScopedTask ## __LINE__                                 \
                    (openvino::itt::handle<Task ## __LINE__>(taskOrTaskName));

/**
 * @endcond
 */
    } // namespace itt
} // namespace openvino
