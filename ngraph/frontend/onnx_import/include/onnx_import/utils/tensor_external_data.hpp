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
                std::string m_data_location;
                int m_offset = 0;
                int m_data_lenght = 0;
                int m_sha1_digest = 0;
            };
        }
    }
}
