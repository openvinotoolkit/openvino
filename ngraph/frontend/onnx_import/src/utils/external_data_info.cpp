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

#include <fstream>
#include <iostream>

#include "external_data_info.hpp"
#include "log.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            ExternalDataInfo::ExternalDataInfo(const ONNX_NAMESPACE::TensorProto& tensor)
            {
                for (const auto& entry : tensor.external_data())
                {
                    if (entry.key() == "location")
                        m_data_location = entry.value();
                    if (entry.key() == "offset")
                        m_offset = std::stoi(entry.value());
                    if (entry.key() == "length")
                        m_data_lenght = std::stoi(entry.value());
                    if (entry.key() == "checksum")
                        m_sha1_digest = std::stoi(entry.value());
                }
            }

            std::string ExternalDataInfo::load_external_data() const
            {
                std::ifstream external_data_stream(m_data_location,
                                                std::ios::binary | std::ios::in | std::ios::ate);
                if (external_data_stream.fail())
                    throw invalid_external_data{*this};

                std::streamsize read_data_lenght;
                if (m_data_lenght == 0) // read entire file
                    read_data_lenght = external_data_stream.tellg();
                else
                    read_data_lenght = m_data_lenght;

                const auto page_size = 4096;
                if (m_offset != 0 && m_offset % page_size != 0)
                {
                    NGRAPH_WARN << "offset should be multiples 4096 (page size) to enable mmap support";
                }
                // default value of m_offset is 0
                external_data_stream.seekg(m_offset, std::ios::beg);

                if (m_sha1_digest != 0)
                {
                    NGRAPH_WARN << "SHA1 checksum is not supported";
                }

                std::string read_data;
                read_data.resize(read_data_lenght);
                // TODO CHECK
                external_data_stream.read(&read_data[0], read_data_lenght);
                external_data_stream.close();
                return read_data;
            }
        }
    }
}
