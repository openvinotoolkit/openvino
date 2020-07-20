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
 * @brief Defines openvino domains for tracing
 * @file itt_domains.hpp
 */

#pragma once

namespace openvino
{
    namespace itt
    {
        namespace domains
        {
            struct Ngraph
            {
                static const char * name() noexcept
                {
                    return "Ngraph";
                }
            };

            struct IE
            {
                static const char * name() noexcept
                {
                    return "IE";
                }
            };

            struct IEPreproc
            {
                static const char * name() noexcept
                {
                    return "IEPreproc";
                }
            };

            struct IETransform
            {
                static const char * name() noexcept
                {
                    return "IETransform";
                }
            };

            struct LPT
            {
                static const char * name() noexcept
                {
                    return "LPT";
                }
            };

            struct Plugin
            {
                static const char * name() noexcept
                {
                    return "Plugin";
                }
            };

            struct MKLDNNPlugin
            {
                static const char * name() noexcept
                {
                    return "MKLDNNPlugin";
                }
            };

            struct HeteroPlugin
            {
                static const char * name() noexcept
                {
                    return "HeteroPlugin";
                }
            };

            struct TemplatePlugin
            {
                static const char * name() noexcept
                {
                    return "TemplatePlugin";
                }
            };

            struct CLDNNPlugin
            {
                static const char * name() noexcept
                {
                    return "CLDNNPlugin";
                }
            };

            struct V7Reader
            {
                static const char * name() noexcept
                {
                    return "V7Reader";
                }
            };

            struct V10Reader
            {
                static const char * name() noexcept
                {
                    return "V10Reader";
                }
            };
        } // namespace domains
    } // namespace itt
} // namespace openvino
