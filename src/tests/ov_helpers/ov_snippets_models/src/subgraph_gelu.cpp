// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "subgraph_gelu.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/init_node_info.hpp"

namespace ov {
namespace test {
namespace snippets {

CodegenGeluFunction::CodegenGeluFunction(const std::vector<ov::PartialShape>& inputShapes,
                                         ov::element::Type_t precision)
    : SnippetsFunctionBase(inputShapes, precision) {
    OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> CodegenGeluFunction::initOriginal() const {
    auto input0 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    auto input1 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[1]);
    auto add = std::make_shared<ov::op::v1::Add>(input0, input1);
    auto gelu = std::make_shared<ov::op::v0::Gelu>(add);
    auto result = std::make_shared<ov::op::v0::Result>(gelu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});

    return model;
}

}  // namespace snippets
}  // namespace test
}  // namespace ov

