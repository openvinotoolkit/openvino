// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lir_test_utils.hpp"

#include "openvino/opsets/opset10.hpp"
#include "snippets/lowered/pass/insert_load_store.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/store.hpp"

namespace ov {
namespace test {
namespace snippets {

using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;

TEST_F(LoweredPassTestsF, InsertLoadStore) {
    const auto vector_size = 16ul;
    const auto input_precision = ov::element::i8;
    const auto convert_precision = ov::element::f32;
    const ov::PartialShape input_shape{1, 3, 16, 16};
    {
        auto param = linear_ir->push_node<ov::opset10::Parameter>(input_precision, input_shape);
        auto convert = linear_ir->push_node<ov::opset10::Convert>(param.second, convert_precision);
        auto result = linear_ir->push_node<ov::opset10::Result>(convert.second);
    }
    pipeline.register_pass<InsertLoadStore>(vector_size);
    {
        auto param = linear_ir_ref->push_node<ov::opset10::Parameter>(input_precision, input_shape);
        auto load = linear_ir_ref->push_node<ov::snippets::op::Load>(param.second, vector_size);
        auto convert = linear_ir_ref->push_node<ov::opset10::Convert>(load.second, convert_precision);
        auto store = linear_ir_ref->push_node<ov::snippets::op::Store>(convert.second, vector_size);
        auto result = linear_ir_ref->push_node<ov::opset10::Result>(store.second);
    }
}
}  // namespace snippets
}  // namespace test
}  // namespace ov
