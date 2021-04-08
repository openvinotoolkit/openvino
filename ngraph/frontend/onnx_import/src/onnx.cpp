// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <memory>
#include <core/null_node.hpp>
#include <onnx/onnx_pb.h>

#include "core/graph.hpp"
#include "core/model.hpp"
#include "core/transform.hpp"

#include "ngraph/except.hpp"
#include "onnx_common/parser.hpp"
#include "onnx_import/onnx.hpp"
#include "onnx_import/utils/onnx_internal.hpp"
#include "ops_bridge.hpp"
#include <onnx_import/onnx_node.hpp>

namespace ngraph
{
    namespace onnx_import
    {
        std::shared_ptr<Function> import_onnx_model(std::istream& stream,
                                                    const std::string& model_path, bool decode_only)
        {
            auto model_proto = std::make_shared<ONNX_NAMESPACE::ModelProto>(onnx_common::parse_from_istream(stream));
            return detail::import_onnx_model(model_proto, model_path, decode_only);
        }

        std::shared_ptr<Function> import_onnx_model(const std::string& file_path, bool decode_only)
        {
            std::ifstream model_stream{file_path, std::ios::in | std::ios::binary};

            if (!model_stream.is_open())
            {
                throw ngraph_error("Error during import of ONNX model expected to be in file: " +
                                   file_path + ". Could not open the file.");
            };

            return import_onnx_model(model_stream, file_path, decode_only);
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


        /// Convert nGraph function with ONNXNode inclusions finally to regular opset
        void convert_onnx_nodes (std::shared_ptr<Function> f)
        {
            auto ops = f->get_ordered_ops();
            for(auto node: ops)
            {
                if(auto raw_node = std::dynamic_pointer_cast<frontend::ONNXNode>(node))
                {
                    // Update cache to make sure that all inputs are properly registered
                    // based on proto names

                    const auto& onnx_node = raw_node->get_onnx_node();
                    auto input_names = onnx_node.get_input_names();
                    assert(raw_node->get_input_size() == input_names.size());
                    for(size_t i = 0; i < input_names.size(); ++i)
                    {
                        raw_node->get_onnx_graph()->update_node_input_cache(input_names[i], raw_node->get_input_source_output(i));
                    }
                    // TODO: Configure it in a proper way -- don't modify graph properties (don't even keep decode_only as a graph property)
                    raw_node->get_onnx_graph()->set_decode_only(false);
                    OutputVector ng_nodes{onnx_node.get_ng_nodes()};
                    // Filter out null outputs
                    while(!ng_nodes.empty())
                    {
                        if(dynamic_cast<onnx_import::NullNode*>(ng_nodes.back().get_node()))
                        {
                            ng_nodes.pop_back();
                        }
                        else break;
                    }
                    std::cerr << "[ INFO ] Translating " << raw_node->get_onnx_node().op_type() << "\n";
                    for(const auto& output: ng_nodes)
                    {
                        std::cerr << "    output: " << output.get_partial_shape() << "\n";
                    }
                    replace_node(raw_node, ng_nodes);
                }
                else
                {
                    // Have to revalidate node because new intpus can affect shape/type propagation for already translated nodes
                    node->revalidate_and_infer_types();
                }
            }
        }
    } // namespace onnx_import

} // namespace ngraph
