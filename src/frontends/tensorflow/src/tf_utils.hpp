// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "attr_value.pb.h"
#include "node_def.pb.h"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/runtime/tensor.hpp"
#include "tensor.pb.h"
#include "tensor_shape.pb.h"
#include "types.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {

ov::element::Type get_ov_type(const ::tensorflow::DataType& type);

ov::Any unpack_tensor_proto(const ::tensorflow::TensorProto& tensor_proto);

ov::Any unpack_tensor_proto(const ::tensorflow::TensorProto& tensor_proto,
                            const ::tensorflow::TensorShapeProto& tensor_shape,
                            const ::tensorflow::DataType& tensor_type);

// structure to save conditional flow marker
struct CfMarkerType {
    // new_markers serves to mark Switch node and saves its own marker id
    // Switch node contains only one marker
    // for other type of nodes, new_markers vector is empty
    std::vector<uint32_t> new_markers;

    // existing_markers contains Switch node markers collected so far
    // and indices of conditional flow branches
    // for example, If operation (represented with Switch-Merge node) has two branches with indices 0 and 1
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> existing_markers;

    // eliminated_markers is not empty only for Merge node where several branches of the same conditional flow
    // are merging
    // for example, if existing_markers contains element with a key = 4 (it means the fourth conditional flow)
    // and value {0, 1}, it means Merge node eliminates this conditional flow
    // and futher data edges are not affected by this particular conditional flow
    std::vector<uint32_t> eliminated_markers;
};

// a map type to save data/control edges from which the node is dependent
using ControlDepsMap = std::unordered_map<std::string, std::set<ov::Output<ov::Node>>>;

// check if the given node contains conditional flow marker in run-time info
bool cf_marker_exists(const std::shared_ptr<const ov::Node>& node);

// retrieves conditional flow marker in run-time info for the given node
CfMarkerType get_cf_marker(const std::shared_ptr<const ov::Node>& node);

// sets conditional flow marker for the given node in run-time info
void set_cf_marker(const CfMarkerType& cf_marker, const std::shared_ptr<ov::Node>& node);

// generates unique conditional flow marker index
uint32_t generate_cf_marker();

// propagate conditional flow markers to nodes generating ov_outputs
// based on conditional flow presence in input nodes
// also, it creates a vector of tensor/edges output_control_deps from which the current node(s) is dependent
bool propogate_conditional_flow(const ov::OutputVector& ov_inputs,
                                const ov::frontend::NamedOutputVector& ov_outputs,
                                const std::set<ov::Output<ov::Node>>& input_control_deps,
                                std::set<ov::Output<ov::Node>>& output_control_deps);
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
