// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_import/onnx.hpp"

#include <onnx/onnx_pb.h>

#include <fstream>
#include <memory>

#include "ngraph/except.hpp"
#include "onnx_common/parser.hpp"
#include "onnx_import/onnx_utils.hpp"
#include "ops_bridge.hpp"
#include "utils/onnx_internal.hpp"

namespace {
// it's the same type of object as the one used in the ONNX FE
// it's called legacy here because it's supposed to work with legacy public API only
// the new API including ONNX FE uses its own, non static, per-frontend object instance of OperatorsBridge
ngraph::onnx_import::OperatorsBridge legacy_ops_bridge{};
}  // namespace

namespace ngraph {
namespace onnx_import {
std::shared_ptr<Function> import_onnx_model(std::istream& stream, const std::string& model_path) {
    const auto model_proto = std::make_shared<ONNX_NAMESPACE::ModelProto>(onnx_common::parse_from_istream(stream));
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
    const auto op_set = OperatorsBridge{}.get_operator_set(domain == "ai.onnx" ? "" : domain, version);
    // TODO - move this functionality to the OperatorsBridge to avoid obsolete creation of an OperatorSet here
    std::set<std::string> op_list{};
    for (const auto& op : op_set) {
        op_list.emplace(op.first);
    }
    return op_list;
}

bool is_operator_supported(const std::string& op_name, std::int64_t version, const std::string& domain) {
    return legacy_ops_bridge.is_operator_registered(op_name, version, domain == "ai.onnx" ? "" : domain);
}

void register_operator(const std::string& name, std::int64_t version, const std::string& domain, Operator fn) {
    legacy_ops_bridge.register_operator(name, version, domain, std::move(fn));
}

void unregister_operator(const std::string& name, std::int64_t version, const std::string& domain) {
    OperatorsBridge::unregister_operator(name, version, domain);
}

}  // namespace onnx_import

}  // namespace ngraph
