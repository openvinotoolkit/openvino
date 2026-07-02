// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_nd.hpp"

#include <vector>

#include "openvino/op/non_zero.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/transpose.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

class GatherNDLayerTemplateTest : virtual public SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_TEMPLATE;
    }

    template <class TGatherND>
    static std::shared_ptr<Model> make_nonzero_gather_model(PartialShape data_shape, int batch_dims) {
        const auto data = std::make_shared<op::v0::Parameter>(element::i32, data_shape);
        // zero constant to compare against
        const auto zero = op::v0::Constant::create(element::i32, Shape{1, 1}, {0});
        const auto not_equal = std::make_shared<op::v1::NotEqual>(data, zero);
        // NonZero: output shape [2, -1] i64 (data-dependent)
        const auto non_zero = std::make_shared<op::v3::NonZero>(not_equal, element::i64);
        // Transpose order [1, 0]: output shape [-1, 2] i64
        const auto order = op::v0::Constant::create(element::i64, Shape{2}, {1, 0});
        const auto transpose = std::make_shared<op::v1::Transpose>(non_zero, order);
        const auto gather = std::make_shared<TGatherND>(data, transpose, batch_dims);
        return std::make_shared<ov::Model>(gather->outputs(), ov::ParameterVector{data}, "gatherND");
    }
};

TEST_F(GatherNDLayerTemplateTest, smoke_dynamic_v5_nonzero) {
    const auto input_shapes =
        std::vector<InputShape>{InputShape{ov::PartialShape({1, -1}), std::vector<ov::Shape>{ov::Shape{1, 101}}}};
    init_input_shapes(input_shapes);
    function = make_nonzero_gather_model<op::v5::GatherND>(inputDynamicShapes.at(0), 0);

    run();
}

TEST_F(GatherNDLayerTemplateTest, smoke_dynamic_v8_nonzero) {
    const auto input_shapes =
        std::vector<InputShape>{InputShape{ov::PartialShape({1, -1}), std::vector<ov::Shape>{ov::Shape{1, 99}}}};
    init_input_shapes(input_shapes);
    function = make_nonzero_gather_model<op::v8::GatherND>(inputDynamicShapes.at(0), 0);

    run();
}

TEST_F(GatherNDLayerTemplateTest, smoke_static_v5_nonzero) {
    const auto input_shapes =
        std::vector<InputShape>{InputShape{ov::PartialShape({8, 32}), std::vector<ov::Shape>{ov::Shape{8, 32}}}};
    init_input_shapes(input_shapes);
    function = make_nonzero_gather_model<op::v5::GatherND>(inputDynamicShapes.at(0), 0);

    run();
}

TEST_F(GatherNDLayerTemplateTest, smoke_static_v8_nonzero) {
    const auto input_shapes =
        std::vector<InputShape>{InputShape{ov::PartialShape({8, 32}), std::vector<ov::Shape>{ov::Shape{8, 32}}}};
    init_input_shapes(input_shapes);
    function = make_nonzero_gather_model<op::v8::GatherND>(inputDynamicShapes.at(0), 0);

    run();
}
}  // namespace ov::test
