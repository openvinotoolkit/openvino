// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

// replica count, element type, whether the branches contain MaxPool -> Add -> Relu after Convolution
using BatchReplicatedBranchesParams = std::tuple<size_t, ov::element::Type, bool>;

// N branches Gather(i) -> Convolution -> [MaxPool -> Add -> Relu] concatenated along the width,
// expected to be executed as one batched Convolution
class BatchReplicatedBranches : public testing::WithParamInterface<BatchReplicatedBranchesParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BatchReplicatedBranchesParams>& info) {
        const auto& [replicas, type, post_ops] = info.param;
        std::ostringstream result;
        result << "replicas=" << replicas << "_type=" << type << "_post_ops=" << post_ops;
        return result.str();
    }

protected:
    const ov::test::utils::InputGenerateData data_range{-1, 2, 1000, 1};

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [replicas, type, post_ops] = GetParam();
        inType = outType = type;
        if (type == ov::element::f16) {
            // Each output accumulates 3 * 3 * 3 products of values in [-1, 1] in f16
            abs_threshold = 0.02f;
        }

        const ov::Shape input_shape{1, replicas, 3, 16, 16};
        auto input = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto weights = ov::test::utils::make_constant(type, ov::Shape{8, 3, 3, 3}, data_range);
        auto bias = ov::test::utils::make_constant(type, ov::Shape{8, 1, 1}, data_range);

        ov::OutputVector branches;
        for (size_t i = 0; i < replicas; ++i) {
            auto index = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {i});
            auto gather = std::make_shared<ov::op::v8::Gather>(input, index, axis);
            ov::Output<ov::Node> branch = std::make_shared<ov::op::v1::Convolution>(gather,
                                                                                    weights,
                                                                                    ov::Strides{1, 1},
                                                                                    ov::CoordinateDiff{1, 1},
                                                                                    ov::CoordinateDiff{1, 1},
                                                                                    ov::Strides{1, 1});
            if (post_ops) {
                branch = std::make_shared<ov::op::v8::MaxPool>(branch, ov::Strides{2, 2}, ov::Strides{1, 1}, ov::Shape{0, 0}, ov::Shape{0, 0}, ov::Shape{2, 2})
                             ->output(0);
                branch = std::make_shared<ov::op::v1::Add>(branch, bias);
                branch = std::make_shared<ov::op::v0::Relu>(branch);
            }
            branches.push_back(branch);
        }
        auto concat = std::make_shared<ov::op::v0::Concat>(branches, 3);
        function = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input});
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& input = function->get_parameters().front();
        inputs.insert({input, ov::test::utils::create_and_fill_tensor(input->get_element_type(), target_input_static_shapes.front(), data_range)});
    }

    void validate() override {
        ov::test::SubgraphBaseTest::validate();
        ov::test::CheckNumberOfNodesWithType(compiledModel, "Convolution", 1);
    }
};

TEST_P(BatchReplicatedBranches, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_BatchReplicatedBranches,
                         BatchReplicatedBranches,
                         ::testing::Combine(::testing::Values(2, 4), ::testing::Values(ov::element::f32, ov::element::f16), ::testing::Bool()),
                         BatchReplicatedBranches::getTestCaseName);

}  // namespace
