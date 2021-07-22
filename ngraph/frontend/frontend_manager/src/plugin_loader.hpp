// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>

#ifdef _WIN32
static const char FileSeparator[] = "\\";
static const char PathSeparator[] = ";";
#else
static const char FileSeparator[] = "/";
static const char PathSeparator[] = ":";
#endif // _WIN32

namespace ngraph
{
    namespace frontend
    {
        /// Plugin library handle wrapper. On destruction calls internal function which frees
        /// library handle
        class PluginHandle
        {
        public:
            PluginHandle(std::function<void()> call_on_destruct)
                : m_call_on_destruct(call_on_destruct)
            {
            }

            PluginHandle(const PluginHandle&) = delete;

            PluginHandle& operator=(const PluginHandle&) = delete;

            PluginHandle(PluginHandle&&) = default;

            PluginHandle& operator=(PluginHandle&&) = default;

            ~PluginHandle()
            {
                if (m_call_on_destruct)
                {
                    m_call_on_destruct();
                }
            }

        private:
            std::function<void()> m_call_on_destruct;
        };

        struct PluginData
        {
            PluginData(PluginHandle&& h, FrontEndPluginInfo&& info)
                : m_lib_handle(std::move(h))
                , m_plugin_info(info)
            {
            }

            PluginHandle
                m_lib_handle; // Shall be destroyed when plugin is not needed anymore to free memory
            FrontEndPluginInfo m_plugin_info;
        };

        // Searches for available plugins in a specified directory
        std::vector<PluginData> load_plugins(const std::string& dir_name);

    } // namespace frontend
} // namespace ngraph
