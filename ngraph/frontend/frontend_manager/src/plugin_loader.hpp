// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>

#ifdef _WIN32
const char FileSeparator[] = "\\";
const char PathSeparator[] = ";";
#else
const char FileSeparator[] = "/";
const char PathSeparator[] = ":";
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
            PluginHandle(std::function<void()> callOnDestruct)
                : m_callOnDestruct(callOnDestruct)
            {
            }

            PluginHandle(const PluginHandle&) = delete;

            PluginHandle& operator=(const PluginHandle&) = delete;

            PluginHandle(PluginHandle&&) = default;

            PluginHandle& operator=(PluginHandle&&) = default;

            ~PluginHandle()
            {
                if (m_callOnDestruct)
                {
                    m_callOnDestruct();
                }
            }

        private:
            std::function<void()> m_callOnDestruct;
        };

        struct PluginData
        {
            PluginData(PluginHandle&& h, FrontEndPluginInfo&& info)
                : m_libHandle(std::move(h))
                , m_pluginInfo(info)
            {
            }

            PluginHandle
                m_libHandle; // Shall be destroyed when plugin is not needed anymore to free memory
            FrontEndPluginInfo m_pluginInfo;
        };

        // Searches for available plugins in a specified directory
        std::vector<PluginData> loadPlugins(const std::string& dirName);

    } // namespace frontend
} // namespace ngraph
