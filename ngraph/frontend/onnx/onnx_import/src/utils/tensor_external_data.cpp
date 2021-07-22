// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <sstream>

#include "exceptions.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/log.hpp"
#include "utils/tensor_external_data.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            TensorExternalData::TensorExternalData(const ONNX_NAMESPACE::TensorProto& tensor)
            {
                for (const auto& entry : tensor.external_data())
                {
                    if (entry.key() == "location")
                        m_data_location = entry.value();
                    if (entry.key() == "offset")
                        m_offset = std::stoi(entry.value());
                    if (entry.key() == "length")
                        m_data_length = std::stoi(entry.value());
                    if (entry.key() == "checksum")
                        m_sha1_digest = std::stoi(entry.value());
                }
            }

            std::string TensorExternalData::load_external_data() const
            {
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
                std::wstring path = file_util::multi_byte_char_to_wstring(m_data_location.c_str());
#else
                std::string path = m_data_location;
#endif
                std::ifstream external_data_stream(path,
                                                   std::ios::binary | std::ios::in | std::ios::ate);
                if (external_data_stream.fail())
                    throw error::invalid_external_data{*this};

                std::streamsize read_data_length;
                if (m_data_length == 0) // read entire file
                    read_data_length = external_data_stream.tellg();
                else
                    read_data_length = m_data_length;

                const auto page_size = 4096;
                if (m_offset != 0 && m_offset % page_size != 0)
                {
                    NGRAPH_WARN << "offset should be multiples 4096 (page size) to enable mmap "
                                   "support, current value is "
                                << m_offset;
                }
                // default value of m_offset is 0
                external_data_stream.seekg(m_offset, std::ios::beg);

                if (m_sha1_digest != 0)
                {
                    NGRAPH_WARN << "SHA1 checksum is not supported";
                }

                std::string read_data;
                read_data.resize(read_data_length);
                external_data_stream.read(&read_data[0], read_data_length);
                external_data_stream.close();

                return read_data;
            }

            std::string TensorExternalData::to_string() const
            {
                std::stringstream s;
                s << "ExternalDataInfo(";
                s << "data_full_path: " << m_data_location;
                s << ", offset: " << m_offset;
                s << ", data_length: " << m_data_length;
                s << ", sha1_digest: " << m_sha1_digest << ")";
                return s.str();
            }
        } // namespace detail
    }     // namespace onnx_import
} // namespace ngraph
