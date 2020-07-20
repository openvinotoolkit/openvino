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
#include "itt_domains.hpp"
#include <string>

namespace openvino
{
    namespace itt
    {
        typedef struct domain_ {} *domain_t;
        typedef struct handle_ {} *handle_t;

        namespace internal
        {
            domain_t domain(char const* name);
            handle_t handle(char const* name);
            void taskBegin(domain_t d, handle_t t);
            void taskEnd(domain_t d);
            void threadName(const char* name);
        }

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

        template <typename Tag>
        domain_t domain()
        {
            static auto d = internal::domain(Tag::name());
            return d;
        }

        template <typename Tag>
        handle_t handle()
        {
            static auto h = internal::handle(Tag::name());
            return h;
        }

        template <typename DomainTag>
        struct ScopedTask
        {
            ScopedTask(handle_t taskHandle) noexcept
            {
                internal::taskBegin(domain<DomainTag>(), taskHandle);
            }
            ~ScopedTask() noexcept { internal::taskEnd(domain<DomainTag>()); }
            ScopedTask(const ScopedTask&) = delete;
            ScopedTask& operator=(const ScopedTask&) = delete;
        };

#define OV_ITT_DOMAIN(domainName)                                                                   \
        struct domainName                                                                           \
        {                                                                                           \
            static const char * name() noexcept                                                     \
            {                                                                                       \
                return #domainName;                                                                 \
            }                                                                                       \
        };

#define OV_ITT_SCOPED_TASK(domain, taskName)                                                        \
    struct Task ## __LINE__                                                                         \
    {                                                                                               \
        static const char * name() noexcept                                                         \
        {                                                                                           \
            return taskName;                                                                        \
        }                                                                                           \
    };                                                                                              \
    openvino::itt::ScopedTask<domain> ittScopedTask ## __LINE__                                     \
                                            (openvino::itt::handle<Task ## __LINE__>());
    } // namespace itt
} // namespace openvino
