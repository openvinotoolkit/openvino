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

#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <memory>

#include "ngraph/except.hpp"
#include "ngraph/file_util.hpp"
#include "onnx_import/core/graph.hpp"
#include "onnx_import/core/model.hpp"
#include "onnx_import/onnx.hpp"
#include "onnx_import/ops_bridge.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            namespace error
            {
                struct file_open : ngraph_error
                {
                    explicit file_open(const std::string& path)
                        : ngraph_error{
                              "Error during import of ONNX model expected to be in file: " + path +
                              ". Could not open the file."}
                    {
                    }
                };

                struct stream_parse_binary : ngraph_error
                {
                    explicit stream_parse_binary()
                        : ngraph_error{
                              "Error during import of ONNX model provided as input stream "
                              " with binary protobuf message."}
                    {
                    }
                };

                struct stream_parse_text : ngraph_error
                {
                    explicit stream_parse_text()
                        : ngraph_error{
                              "Error during import of ONNX model provided as input stream "
                              " with prototxt protobuf message."}
                    {
                    }
                };

                struct stream_corrupted : ngraph_error
                {
                    explicit stream_corrupted()
                        : ngraph_error{"Provided input stream has incorrect state."}
                    {
                    }
                };

            } // namespace error

            static const std::vector<std::string> legacy_ops_to_fixup = {
                "DetectionOutput", "FakeQuantize", "GroupNorm", "Normalize", "PriorBox"};

            // There are some models with custom OPs (list above) that has the default domain set.
            // So in order to load the models, we need overwrite the OPs' domain to the one they're
            // registered
            void fixup_legacy_operators(ONNX_NAMESPACE::GraphProto* graph_proto)
            {
                for (auto& node : *graph_proto->mutable_node())
                {
                    auto it = std::find(
                        legacy_ops_to_fixup.begin(), legacy_ops_to_fixup.end(), node.op_type());
                    if (it != legacy_ops_to_fixup.end())
                    {
                        if (!node.has_domain() || node.domain().empty() ||
                            node.domain() == "ai.onnx")
                        {
                            node.set_domain(OPENVINO_ONNX_DOMAIN);
                        }
                    }
                }
            }

            // The paths to external data files are stored as relative to model path.
            // The helper function below combines them and replaces the original relative path.
            // As a result in futher processing data from external files can be read directly.
            void update_external_data_paths(ONNX_NAMESPACE::ModelProto& model_proto,
                                            const std::string& model_path)
            {
                if (model_path.empty())
                {
                    return;
                }
                const auto model_dir_path = file_util::get_directory(model_path);
                auto graph_proto = model_proto.mutable_graph();
                for (auto& initializer_tensor : *graph_proto->mutable_initializer())
                {
                    const auto location_key_value_index = 0;
                    if (initializer_tensor.has_data_location() &&
                        initializer_tensor.data_location() ==
                            ONNX_NAMESPACE::TensorProto_DataLocation::
                                TensorProto_DataLocation_EXTERNAL)
                    {
                        const auto external_data_relative_path =
                            initializer_tensor.external_data(location_key_value_index).value();
                        const auto external_data_full_path =
                            file_util::path_join(model_dir_path, external_data_relative_path);

                        // Set full paths to the external file
                        initializer_tensor.mutable_external_data(location_key_value_index)
                            ->set_value(external_data_full_path);
                    }
                }
            }

            std::shared_ptr<Function>
                convert_to_ng_function(const ONNX_NAMESPACE::ModelProto& model_proto)
            {
                Model model{model_proto};
                Graph graph{model_proto.graph(), model};
                auto function = std::make_shared<Function>(
                    graph.get_ng_outputs(), graph.get_ng_parameters(), graph.get_name());
                for (std::size_t i{0}; i < function->get_output_size(); ++i)
                {
                    function->get_output_op(i)->set_friendly_name(
                        graph.get_outputs().at(i).get_name());
                }
                return function;
            }
        } // namespace detail

        std::shared_ptr<Function> import_onnx_model(std::istream& stream,
                                                    const std::string& model_path)
        {
            if (!stream.good())
            {
                stream.clear();
                stream.seekg(0);
                if (!stream.good())
                {
                    throw detail::error::stream_corrupted();
                }
            }

            ONNX_NAMESPACE::ModelProto model_proto;
            // Try parsing input as a binary protobuf message
            if (!model_proto.ParseFromIstream(&stream))
            {
#ifdef NGRAPH_USE_PROTOBUF_LITE
                throw detail::error::stream_parse_binary();
#else
                // Rewind to the beginning and clear stream state.
                stream.clear();
                stream.seekg(0);
                google::protobuf::io::IstreamInputStream iistream(&stream);
                // Try parsing input as a prototxt message
                if (!google::protobuf::TextFormat::Parse(&iistream, &model_proto))
                {
                    throw detail::error::stream_parse_text();
                }
#endif
            }

            detail::fixup_legacy_operators(model_proto.mutable_graph());
            detail::update_external_data_paths(model_proto, model_path);

            return detail::convert_to_ng_function(model_proto);
        }

        std::shared_ptr<Function> import_onnx_model(const std::string& file_path)
        {
            std::ifstream ifs{file_path, std::ios::in | std::ios::binary};
            if (!ifs.is_open())
            {
                throw detail::error::file_open{file_path};
            }
            return import_onnx_model(ifs, file_path);
        }

        std::set<std::string> get_supported_operators(std::int64_t version,
                                                      const std::string& domain)
        {
            OperatorSet op_set{
                OperatorsBridge::get_operator_set(domain == "ai.onnx" ? "" : domain, version)};
            std::set<std::string> op_list{};
            for (const auto& op : op_set)
            {
                op_list.emplace(op.first);
            }
            return op_list;
        }

        bool is_operator_supported(const std::string& op_name,
                                   std::int64_t version,
                                   const std::string& domain)
        {
            return OperatorsBridge::is_operator_registered(
                op_name, version, domain == "ai.onnx" ? "" : domain);
        }

    } // namespace onnx_import

} // namespace ngraph
