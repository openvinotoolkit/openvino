// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/variadic_split.hpp"

#include <algorithm>

namespace {

class VariadicSplitConvertGPU : public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        init_input_shapes({{ov::PartialShape{2, 3, 64, 64}, {{2, 3, 64, 64}}}});

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2, 3, 64, 64});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis, lengths);
        auto convert = std::make_shared<ov::op::v0::Convert>(split->output(1), ov::element::f32);
        function = std::make_shared<ov::Model>(ov::OutputVector{convert},
                                               ov::ParameterVector{input},
                                               "VariadicSplitConvertGPU");
        ASSERT_EQ(function->get_output_element_type(0), ov::element::f32);
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        ASSERT_EQ(target_input_static_shapes.size(), 1u);
        ov::Tensor tensor(ov::element::f16, target_input_static_shapes[0]);
        auto* data = tensor.data<ov::float16>();
        const auto batch_size = tensor.get_size() / 2;
        std::fill(data, data + batch_size, ov::float16(0.25f));
        std::fill(data + batch_size, data + tensor.get_size(), ov::float16(0.75f));
        inputs.emplace(function->get_parameters()[0], tensor);
    }
};

TEST_F(VariadicSplitConvertGPU, SecondBatchSlice) {
    run();
}

}  // namespace
