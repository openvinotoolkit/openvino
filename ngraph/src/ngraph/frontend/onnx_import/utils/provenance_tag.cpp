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

#include <algorithm>
#include <functional>
#include <sstream>

#include "utils/provenance_tag.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            std::string concat_strings(
                const std::vector<std::reference_wrapper<const std::string>>& strings)
            {
                const auto concat_with_comma =
                    [](const std::string& accumulator,
                       std::reference_wrapper<const std::string> next_string) {
                        return accumulator + ", " + next_string.get();
                    };

                return std::accumulate(
                    strings.begin() + 1, strings.end(), strings.begin()->get(), concat_with_comma);
            }

            std::string build_input_provenance_tag(const std::string& input_name,
                                                   const PartialShape& shape)
            {
                std::stringstream tag_builder;
                tag_builder << "<ONNX Input (" << input_name << ") Shape:" << shape << ">";
                return tag_builder.str();
            }

            std::string build_op_provenance_tag(const Node& onnx_node)
            {
                const auto output_names = concat_strings(onnx_node.get_output_names());
                const auto node_name =
                    onnx_node.get_name().empty() ? "" : onnx_node.get_name() + " ";

                return std::string{"<ONNX " + onnx_node.op_type() + " (" + node_name + "-> " +
                                   output_names + ")>"};
            }

        } // namespace detail
    }     // namespace onnx_import
} // namespace ngraph
