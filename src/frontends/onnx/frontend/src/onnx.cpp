// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_import/onnx.hpp"

#include <onnx/onnx_pb.h>

#include <fstream>
#include <memory>

#include "ngraph/except.hpp"
#include "onnx_common/parser.hpp"
#include "ops_bridge.hpp"
#include "utils/onnx_internal.hpp"

namespace ngraph {
namespace onnx_import {
std::shared_ptr<Function> import_onnx_model(std::istream& stream, const std::string& model_path) {
    auto model_proto = std::make_shared<ONNX_NAMESPACE::ModelProto>(onnx_common::parse_from_istream(stream));
    return detail::import_onnx_model(model_proto, model_path);
}

std::shared_ptr<Function> import_onnx_model(const std::string& file_path) {
    std::ifstream model_stream{file_path, std::ios::in | std::ios::binary};

    if (!model_stream.is_open()) {
        throw ngraph_error("Error during import of ONNX model expected to be in file: " + file_path +
                           ". Could not open the file.");
    };

    return import_onnx_model(model_stream, file_path);
}

std::set<std::string> get_supported_operators(std::int64_t version, const std::string& domain) {
    OperatorSet op_set{OperatorsBridge::get_operator_set(domain == "ai.onnx" ? "" : domain, version)};
    std::set<std::string> op_list{};
    for (const auto& op : op_set) {
        op_list.emplace(op.first);
    }
    return op_list;
}

bool is_operator_supported(const std::string& op_name, std::int64_t version, const std::string& domain) {
    return OperatorsBridge::is_operator_registered(op_name, version, domain == "ai.onnx" ? "" : domain);
}

}  // namespace onnx_import

}  // namespace ngraph
