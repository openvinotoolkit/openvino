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

#pragma once

#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    /// \brief Serialize a Function to a json string
    /// \param func The Function to serialize
    /// \param indent If 0 then there is no formatting applied and the resulting string is the
    ///    most compact representation. If non-zero then the json string is formatted with the
    ///    indent level specified.
    NGRAPH_API
    std::string serialize(std::shared_ptr<ngraph::Function> func, size_t indent = 0);

    /// \brief Serialize a Function to a json file
    /// \param path The path to the output file
    /// \param func The Function to serialize
    /// \param indent If 0 then there is no formatting applied and the resulting string is the
    ///    most compact representation. If non-zero then the json string is formatted with the
    ///    indent level specified.
    NGRAPH_API
    void serialize(const std::string& path,
                   std::shared_ptr<ngraph::Function> func,
                   size_t indent = 0);

    /// \brief Serialize a Function to a json stream
    /// \param out The output stream to which the data is serialized.
    /// \param func The Function to serialize
    /// \param indent If 0 then there is no formatting applied and the json is the
    ///    most compact representation. If non-zero then the json is formatted with the
    ///    indent level specified.
    NGRAPH_API
    void serialize(std::ostream& out, std::shared_ptr<ngraph::Function> func, size_t indent = 0);

    /// \brief Deserialize a Function
    /// \param in An isteam to the input data
    NGRAPH_API
    std::shared_ptr<ngraph::Function> deserialize(std::istream& in);

    /// \brief Deserialize a Function
    /// \param str The json formatted string to deseriailze.
    NGRAPH_API
    std::shared_ptr<ngraph::Function> deserialize(const std::string& str);

    /// \brief If enabled adds output shapes to the serialized graph
    /// \param enable Set to true to enable or false otherwise
    ///
    /// Option may be enabled by setting the environment variable NGRAPH_SERIALIZER_OUTPUT_SHAPES
    NGRAPH_API
    void set_serialize_output_shapes(bool enable);
    NGRAPH_API
    bool get_serialize_output_shapes();

    class WithSerializeOutputShapesEnabled
    {
    public:
        WithSerializeOutputShapesEnabled(bool enabled = true)
        {
            m_serialize_output_shapes_enabled = get_serialize_output_shapes();
            set_serialize_output_shapes(enabled);
        }
        ~WithSerializeOutputShapesEnabled()
        {
            set_serialize_output_shapes(m_serialize_output_shapes_enabled);
        }

    private:
        bool m_serialize_output_shapes_enabled;
    };
}
