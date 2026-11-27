// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/batch_replicated_branches.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace testing;
using namespace ov;

namespace {

// Each branch maps a [1, 3, 8, 8] slice to [1, 4, 4, 4] ([1, 4, 2, 2] with MaxPool)
struct BranchConfig {
    size_t replicas = 4;
    int64_t replica_axis = 1;
    int64_t concat_axis = 3;
    bool decompress_weights = false;
    bool max_pool = false;
    bool bias = false;
    bool relu = false;
    bool port_rt_info = false;
};

std::ostream& operator<<(std::ostream& os, const BranchConfig& config) {
    return os << "replicas=" << config.replicas << "_replica_axis=" << config.replica_axis << "_concat_axis=" << config.concat_axis
              << (config.decompress_weights ? "_decompress" : "") << (config.max_pool ? "_pool" : "") << (config.bias ? "_bias" : "")
              << (config.relu ? "_relu" : "") << (config.port_rt_info ? "_port_rt_info" : "");
}

// Original: Gather(i) -> branch -> Concat for i = 0..N-1
// Batched:  [Transpose] -> Reshape of data to [N, 3, 8, 8] -> one branch -> Transpose -> Reshape
std::shared_ptr<Model> getModel(const BranchConfig& config, bool batched) {
    Shape data_shape{1, 3, 8, 8};
    data_shape.insert(data_shape.begin() + config.replica_axis, config.replicas);
    auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);

    // Each call creates its own (equal) weights and bias, so the original branches don't share constants
    auto make_branch = [&](const Output<Node>& input) -> Output<Node> {
        Output<Node> weights = op::v0::Constant::create(config.decompress_weights ? element::f16 : element::f32, Shape{4, 3, 2, 2}, {0.5f});
        if (config.decompress_weights) {
            auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
            mark_as_decompression(convert);
            weights = convert;
        }
        auto conv = std::make_shared<op::v1::Convolution>(input, weights, Strides{2, 2}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        if (config.port_rt_info) {
            conv->input(0).get_rt_info()["port_info"] = std::string{"data"};
            conv->output(0).get_rt_info()["port_info"] = std::string{"conv_output"};
        }
        Output<Node> branch = conv;
        if (config.max_pool)
            branch = std::make_shared<op::v8::MaxPool>(branch, Strides{2, 2}, Strides{1, 1}, Shape{0, 0}, Shape{0, 0}, Shape{2, 2})->output(0);
        if (config.bias)
            branch = std::make_shared<op::v1::Add>(branch, op::v0::Constant::create(element::f32, Shape{4, 1, 1}, {0.1f, 0.2f, 0.3f, 0.4f}));
        if (config.relu)
            branch = std::make_shared<op::v0::Relu>(branch);
        return branch;
    };

    if (!batched) {
        OutputVector branches;
        for (size_t i = 0; i < config.replicas; ++i) {
            auto gather = std::make_shared<op::v8::Gather>(data,
                                                           op::v0::Constant::create(element::i64, Shape{}, {i}),
                                                           op::v0::Constant::create(element::i64, Shape{}, {config.replica_axis}));
            branches.push_back(make_branch(gather));
        }
        auto concat = std::make_shared<op::v0::Concat>(branches, config.concat_axis);
        return std::make_shared<Model>(OutputVector{concat}, ParameterVector{data});
    }

    Output<Node> batched_input = data;
    // Dims before the replica axis are 1 for axes 0 and 1; axis 2 ([1, 3, N, 8, 8]) needs a Transpose
    if (config.replica_axis > 1) {
        std::vector<int64_t> order{config.replica_axis};
        for (int64_t axis = 0; axis < 5; ++axis) {
            if (axis != config.replica_axis)
                order.push_back(axis);
        }
        batched_input = std::make_shared<op::v1::Transpose>(data, op::v0::Constant::create(element::i64, Shape{5}, order));
    }
    const Shape batched_input_shape{config.replicas, 3, 8, 8};
    batched_input = std::make_shared<op::v1::Reshape>(batched_input, op::v0::Constant::create(element::i64, Shape{4}, batched_input_shape), false);
    auto batched_branch = make_branch(batched_input);

    const auto branch_shape = batched_branch.get_shape();
    std::vector<int64_t> order;
    std::vector<int64_t> output_shape{1};
    for (int64_t axis = 1; axis < 4; ++axis) {
        if (axis == config.concat_axis)
            order.push_back(0);
        order.push_back(axis);
        output_shape.push_back(static_cast<int64_t>(branch_shape[axis] * (axis == config.concat_axis ? config.replicas : 1)));
    }
    auto transpose = std::make_shared<op::v1::Transpose>(batched_branch, op::v0::Constant::create(element::i64, Shape{4}, order));
    auto reshape = std::make_shared<op::v1::Reshape>(transpose, op::v0::Constant::create(element::i64, Shape{4}, output_shape), false);
    return std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
}

std::shared_ptr<Node> get_concat(const std::shared_ptr<Model>& model) {
    return model->get_results()[0]->get_input_node_shared_ptr(0);
}

class BatchReplicatedBranchesTests : public TransformationTestsF {
public:
    BatchReplicatedBranchesTests() {
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::CONSUMERS_COUNT);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    }
};

// Layout (replicas, replica axis, Concat axis), decompress_weights, max_pool, bias, relu
using BatchReplicatedBranchesParams = std::tuple<BranchConfig, bool, bool, bool, bool>;

class BatchReplicatedBranchesParamTests : public BatchReplicatedBranchesTests, public WithParamInterface<BatchReplicatedBranchesParams> {
public:
    static BranchConfig get_config(const BatchReplicatedBranchesParams& params) {
        auto [config, decompress_weights, max_pool, bias, relu] = params;
        config.decompress_weights = decompress_weights;
        config.max_pool = max_pool;
        config.bias = bias;
        config.relu = relu;
        return config;
    }

    static std::string getTestCaseName(const TestParamInfo<BatchReplicatedBranchesParams>& info) {
        std::ostringstream ss;
        ss << get_config(info.param);
        return ss.str();
    }
};

TEST_P(BatchReplicatedBranchesParamTests, BatchesBranches) {
    const auto config = get_config(GetParam());
    model = getModel(config, false);
    model_ref = getModel(config, true);
}

INSTANTIATE_TEST_SUITE_P(BatchReplicatedBranches,
                         BatchReplicatedBranchesParamTests,
                         Combine(Values(BranchConfig{4, 1, 3}, BranchConfig{2, 0, 1}, BranchConfig{3, 2, 2}), Bool(), Bool(), Bool(), Bool()),
                         BatchReplicatedBranchesParamTests::getTestCaseName);

TEST_F(BatchReplicatedBranchesTests, BatchesBranchesWithPortRtInfo) {
    BranchConfig config{};
    config.port_rt_info = true;
    model = getModel(config, false);
    model_ref = getModel(config, true);
}

TEST_F(BatchReplicatedBranchesTests, IsIdempotent) {
    const BranchConfig config{};
    model = getModel(config, false);
    model_ref = getModel(config, true);
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
}

TEST_F(BatchReplicatedBranchesTests, BatchesBranchesWithGatherV1) {
    const BranchConfig config{};
    model = getModel(config, false);
    for (const auto& branch : get_concat(model)->input_values()) {
        const auto gather = branch.get_node()->get_input_node_shared_ptr(0);
        ASSERT_TRUE(is_type<op::v8::Gather>(gather));
        replace_node(gather, std::make_shared<op::v1::Gather>(gather->input_value(0), gather->input_value(1), gather->input_value(2)));
    }
    model_ref = getModel(config, true);
}

TEST_F(BatchReplicatedBranchesTests, RejectsNonExactReplicaOrder) {
    model = getModel(BranchConfig{}, false);
    const auto concat = get_concat(model);
    const auto input2 = concat->input_value(2);
    concat->input(2).replace_source_output(concat->input_value(3));
    concat->input(3).replace_source_output(input2);
}

TEST_F(BatchReplicatedBranchesTests, RejectsPartialReplicaDimension) {
    model = getModel(BranchConfig{}, false);
    model->get_parameters()[0]->set_partial_shape(PartialShape{1, 5, 3, 8, 8});
    model->validate_nodes_and_infer_types();
}

TEST_F(BatchReplicatedBranchesTests, RejectsConcatOnBatchAxis) {
    BranchConfig config{};
    config.concat_axis = 0;
    model = getModel(config, false);
}

TEST_F(BatchReplicatedBranchesTests, RejectsDifferentBranchTopology) {
    model = getModel(BranchConfig{}, false);
    const auto concat = get_concat(model);
    concat->input(2).replace_source_output(std::make_shared<op::v0::Relu>(concat->input_value(2)));
}

TEST_F(BatchReplicatedBranchesTests, RejectsDifferentWeights) {
    model = getModel(BranchConfig{}, false);
    const auto conv = get_concat(model)->get_input_node_shared_ptr(1);
    ASSERT_TRUE(is_type<op::v1::Convolution>(conv));
    conv->input(1).replace_source_output(op::v0::Constant::create(element::f32, Shape{4, 3, 2, 2}, {1.0f}));
}

TEST_F(BatchReplicatedBranchesTests, RejectsDynamicShape) {
    model = getModel(BranchConfig{}, false);
    model->get_parameters()[0]->set_partial_shape(PartialShape{1, 4, 3, Dimension::dynamic(), 8});
    model->validate_nodes_and_infer_types();
}

TEST_F(BatchReplicatedBranchesTests, RejectsAddWithoutNumpyBroadcast) {
    BranchConfig config{};
    config.bias = true;
    model = getModel(config, false);
    for (const auto& add : get_concat(model)->input_values()) {
        replace_node(add.get_node_shared_ptr(),
                     std::make_shared<op::v1::Add>(add.get_node()->input_value(0),
                                                   op::v0::Constant::create(element::f32, Shape{1, 4, 4, 4}, {0.1f}),
                                                   op::AutoBroadcastType::NONE));
    }
}

TEST_F(BatchReplicatedBranchesTests, RejectsExternalConsumer) {
    model = getModel(BranchConfig{}, false);
    const auto branch = get_concat(model)->input_value(0);
    model->add_results({std::make_shared<op::v0::Result>(std::make_shared<op::v0::Relu>(branch))});
}

TEST_F(BatchReplicatedBranchesTests, RejectsShapeOfConsumer) {
    model = getModel(BranchConfig{}, false);
    const auto branch = get_concat(model)->input_value(0);
    model->add_results({std::make_shared<op::v0::Result>(std::make_shared<op::v3::ShapeOf>(branch))});
}

TEST_F(BatchReplicatedBranchesTests, RejectsUsedMaxPoolIndices) {
    BranchConfig config{};
    config.max_pool = true;
    model = getModel(config, false);
    const auto max_pool = get_concat(model)->get_input_node_shared_ptr(0);
    ASSERT_TRUE(is_type<op::v8::MaxPool>(max_pool));
    model->add_results({std::make_shared<op::v0::Result>(max_pool->output(1))});
}

TEST_F(BatchReplicatedBranchesTests, RejectsControlDependency) {
    model = getModel(BranchConfig{}, false);
    const auto conv = get_concat(model)->get_input_node_shared_ptr(1);
    ASSERT_TRUE(is_type<op::v1::Convolution>(conv));
    conv->add_control_dependency(model->get_parameters()[0]);
}

TEST_F(BatchReplicatedBranchesTests, RejectsTypeRelaxedBranch) {
    // The attributes comparator has no handler for the TypeVector attributes of TypeRelaxed nodes and reports
    // them as an error even for equal nodes, so only the topology and constants are compared
    comparator.disable(FunctionsComparator::ATTRIBUTES);
    model = getModel(BranchConfig{}, false);
    for (const auto& branch : get_concat(model)->input_values()) {
        const auto conv = as_type_ptr<op::v1::Convolution>(branch.get_node_shared_ptr());
        ASSERT_NE(conv, nullptr);
        replace_node(conv, std::make_shared<op::TypeRelaxed<op::v1::Convolution>>(*conv, element::f32));
    }
}

TEST_F(BatchReplicatedBranchesTests, RejectsTypeRelaxedConcat) {
    // The attributes comparator has no handler for the TypeVector attributes of TypeRelaxed nodes and reports
    // them as an error even for equal nodes, so only the topology and constants are compared
    comparator.disable(FunctionsComparator::ATTRIBUTES);
    model = getModel(BranchConfig{}, false);
    const auto concat = as_type_ptr<op::v0::Concat>(get_concat(model));
    ASSERT_NE(concat, nullptr);
    replace_node(concat, std::make_shared<op::TypeRelaxed<op::v0::Concat>>(*concat, element::f32));
}

}  // namespace
