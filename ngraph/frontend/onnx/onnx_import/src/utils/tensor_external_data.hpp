// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            /// \brief  Helper class used to load tensor data from external files
            class TensorExternalData
            {
            public:
                TensorExternalData(const ONNX_NAMESPACE::TensorProto& tensor);

                /// \brief      Load external data from tensor passed to constructor
                ///
                /// \note       If read data from external file fails,
                /// \note       If reading data from external files fails,
                ///             the invalid_external_data exception is thrown.
                ///
                /// \return     External binary data loaded into a std::string
                std::string load_external_data() const;

                /// \brief      Represets parameter of external data as string
                ///
                /// \return     State of TensorExternalData as string representation
                std::string to_string() const;

            private:
                std::string m_data_location{};
                int m_offset = 0;
                int m_data_length = 0;
                int m_sha1_digest = 0;
            };
        } // namespace detail
    }     // namespace onnx_import
} // namespace ngraph
