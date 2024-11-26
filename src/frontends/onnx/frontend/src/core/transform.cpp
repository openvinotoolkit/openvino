// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/log.hpp"
#if defined(_MSC_VER)
#    pragma warning(push)
// Protobuf: conversion from 'XXX' to 'YYY', possible loss of data
#    pragma warning(disable : 4244)
#endif

#include <algorithm>

#include "core/model.hpp"
#include "core/transform.hpp"
#include "openvino/util/log.hpp"
#include "ops_bridge.hpp"

void ov::frontend::onnx::transform::fixup_legacy_operators(ModelProto& model_proto) {
    auto graph_proto = model_proto.mutable_graph();
    for (auto& node : *graph_proto->mutable_node()) {
        auto it = std::find(legacy_ops_to_fixup.begin(), legacy_ops_to_fixup.end(), node.op_type());
        if (it != legacy_ops_to_fixup.end()) {
            if (!node.has_domain() || node.domain().empty() || node.domain() == "ai.onnx") {
                node.set_domain(OPENVINO_ONNX_DOMAIN);
            }
        }
    }
}

#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
