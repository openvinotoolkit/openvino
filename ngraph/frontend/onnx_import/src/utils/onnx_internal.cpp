// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <onnx/onnx_pb.h>

#include "core/graph.hpp"
#include "core/model.hpp"
#include "core/transform.hpp"
#include "onnx_import/utils/onnx_internal.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            std::shared_ptr<Function>
                convert_to_ng_function(const ONNX_NAMESPACE::ModelProto& model_proto)
            {
                auto p_model_proto = common::make_unique<ONNX_NAMESPACE::ModelProto>(model_proto);
                auto model = common::make_unique<Model>(std::move(p_model_proto));

                Graph graph{std::move(model)};
                auto function = std::make_shared<Function>(
                    graph.get_ng_outputs(), graph.get_ng_parameters(), graph.get_name());
                for (std::size_t i{0}; i < function->get_output_size(); ++i)
                {
                    function->get_output_op(i)->set_friendly_name(
                        graph.get_outputs().at(i).get_name());
                }
                return function;
            }

            std::shared_ptr<Function> import_onnx_model(ONNX_NAMESPACE::ModelProto& model_proto,
                                                        const std::string& model_path)
            {
                transform::expand_onnx_functions(model_proto);
                transform::fixup_legacy_operators(model_proto);
                transform::update_external_data_paths(model_proto, model_path);

                return detail::convert_to_ng_function(model_proto);
            }
        } // namespace detail
    }     // namespace onnx_import
} // namespace ngraph
