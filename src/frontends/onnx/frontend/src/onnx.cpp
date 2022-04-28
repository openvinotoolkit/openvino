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
#include "utils/legacy_conversion_extension.hpp"
#include "utils/onnx_internal.hpp"

namespace {
const auto legacy_conversion_extension = std::make_shared<ngraph::onnx_import::LegacyConversionExtension>();
}  // namespace

namespace ngraph {
namespace onnx_import {
std::shared_ptr<Function> import_onnx_model(std::istream& stream, const std::string& model_path) {
    const auto model_proto = std::make_shared<ONNX_NAMESPACE::ModelProto>(onnx_common::parse_from_istream(stream));
    ov::frontend::ExtensionHolder extensions;
    extensions.conversions.push_back(legacy_conversion_extension);
    return detail::import_onnx_model(model_proto, model_path, std::move(extensions));
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
    return legacy_conversion_extension->ops_bridge().get_supported_operators(version, domain);
}

bool is_operator_supported(const std::string& op_name, std::int64_t version, const std::string& domain) {
    return legacy_conversion_extension->ops_bridge().is_operator_registered(op_name,
                                                                            version,
                                                                            domain == "ai.onnx" ? "" : domain);
}

void register_operator(const std::string& name, std::int64_t version, const std::string& domain, Operator fn) {
    legacy_conversion_extension->register_operator(name, version, domain, std::move(fn));
}

void unregister_operator(const std::string& name, std::int64_t version, const std::string& domain) {
    legacy_conversion_extension->unregister_operator(name, version, domain);
}

}  // namespace onnx_import

}  // namespace ngraph
