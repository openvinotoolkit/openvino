// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <onnx/onnx_pb.h>

#include "core/graph.hpp"
#include "core/model.hpp"
#include "core/transform.hpp"
#include "onnx_import/utils/onnx_internal.hpp"
#include <onnx_import/onnx_node.hpp>

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            std::shared_ptr<Function>
            convert_to_ng_function(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto, bool decode_only)
            {
                auto model = std::make_shared<Model>(*model_proto);
                auto graph = std::make_shared<Graph>(model_proto->graph(), *model, decode_only);
                auto function = std::make_shared<Function>(
                        graph->get_ng_outputs(), graph->get_ng_parameters(), graph->get_name());

                for(auto node: function->get_ops())
                {
                    if(auto raw_node = std::dynamic_pointer_cast<frontend::ONNXNode>(node))
                    {
                        raw_node->set_onnx_graph(graph);
                        raw_node->set_onnx_model(model);
                        raw_node->set_onnx_model_proto(model_proto);
                    }
                }

                for (std::size_t i{0}; i < function->get_output_size(); ++i)
                {
                    function->get_output_op(i)->set_friendly_name(
                            graph->get_outputs().at(i).get_name());
                }
                return function;
            }

            std::shared_ptr<Function> import_onnx_model(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto,
                                                        const std::string& model_path, bool decode_only)
            {
                transform::expand_onnx_functions(*model_proto);
                transform::fixup_legacy_operators(*model_proto);
                transform::update_external_data_paths(*model_proto, model_path);

                return convert_to_ng_function(model_proto, decode_only);
            }
        } // namespace detail
    }     // namespace onnx_import
} // namespace ngraph
