//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#pragma once

#include <onnx/onnx_pb.h>

#include "except.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            class ExternalDataInfo
            {
            public:
                ExternalDataInfo() = delete;
                explicit ExternalDataInfo(const ONNX_NAMESPACE::TensorProto& tensor);

                ExternalDataInfo(const ExternalDataInfo&) = default;
                ExternalDataInfo(ExternalDataInfo&&) = default;

                ExternalDataInfo& operator=(const ExternalDataInfo&) = delete;
                ExternalDataInfo& operator=(ExternalDataInfo&&) = delete;

                std::string load_external_data() const;

                struct invalid_external_data : ngraph_error
                {
                    invalid_external_data(const ExternalDataInfo& external_data_info)
                        : ngraph_error{std::string{"invalid external data - "} + "location: " +
                                       external_data_info.m_data_location + ", offset: " +
                                       std::to_string(external_data_info.m_offset) + ", lenght: " +
                                       std::to_string(external_data_info.m_data_lenght) +
                                       ", sha1_digest: " +
                                       std::to_string(external_data_info.m_sha1_digest)}
                    {
                    }
                };

            private:
                std::string m_data_location;
                int m_offset = 0;
                int m_data_lenght = 0;
                int m_sha1_digest = 0;
            };
        }
    }
}
