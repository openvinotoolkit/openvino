// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <memory>
#include <onnx/onnx_pb.h>

#include "ngraph/except.hpp"
#include "onnx_common/parser.hpp"
#include "onnx_import/onnx.hpp"
#include "onnx_import/utils/onnx_internal.hpp"
#include "ops_bridge.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        std::shared_ptr<Function> import_onnx_model(std::istream& stream,
                                                    const std::string& model_path)
        {
            ONNX_NAMESPACE::ModelProto model_proto{onnx_common::parse_from_istream(stream)};

            return detail::import_onnx_model(model_proto, model_path);
        }

        std::shared_ptr<Function> import_onnx_model(const std::string& file_path)
        {
            std::ifstream model_stream{file_path, std::ios::in | std::ios::binary};

            if (!model_stream.is_open())
            {
                throw ngraph_error("Error during import of ONNX model expected to be in file: " +
                                   file_path + ". Could not open the file.");
            };

            return import_onnx_model(model_stream, file_path);
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
