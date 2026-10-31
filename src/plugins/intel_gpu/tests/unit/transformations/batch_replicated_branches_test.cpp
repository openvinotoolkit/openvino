// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/batch_replicated_branches.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
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

const Shape branch_input_shape{1, 3, 8, 8};

std::shared_ptr<op::v0::Parameter> make_data(const BranchConfig& config, element::Type type = element::f32) {
    Shape shape = branch_input_shape;
    shape.insert(shape.begin() + config.replica_axis, config.replicas);
    return std::make_shared<op::v0::Parameter>(type, shape);
}

std::shared_ptr<Node> make_gather(const Output<Node>& data, int64_t index, const BranchConfig& config) {
    return std::make_shared<op::v8::Gather>(data,
                                            op::v0::Constant::create(element::i64, Shape{}, {index}),
                                            op::v0::Constant::create(element::i64, Shape{}, {config.replica_axis}));
}

Output<Node> make_weights(const BranchConfig& config, float value = 0.5f) {
    if (!config.decompress_weights)
        return op::v0::Constant::create(element::f32, Shape{4, 3, 2, 2}, {value});
    auto convert = std::make_shared<op::v0::Convert>(op::v0::Constant::create(element::f16, Shape{4, 3, 2, 2}, {value}), element::f32);
    mark_as_decompression(convert);
    return convert;
}

Output<Node> make_bias() {
    return op::v0::Constant::create(element::f32, Shape{4, 1, 1}, {0.1f, 0.2f, 0.3f, 0.4f});
}

Output<Node> make_branch(const Output<Node>& input, const Output<Node>& weights, const Output<Node>& bias, const BranchConfig& config) {
    auto conv = std::make_shared<op::v1::Convolution>(input, weights, Strides{2, 2}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
    if (config.port_rt_info) {
        conv->input(0).get_rt_info()["port_info"] = std::string{"data"};
        conv->output(0).get_rt_info()["port_info"] = std::string{"conv_output"};
    }
    Output<Node> branch = conv;
    if (config.max_pool)
        branch = std::make_shared<op::v8::MaxPool>(branch, Strides{2, 2}, Strides{1, 1}, Shape{0, 0}, Shape{0, 0}, Shape{2, 2})->output(0);
    if (config.bias)
        branch = std::make_shared<op::v1::Add>(branch, bias);
    if (config.relu)
        branch = std::make_shared<op::v0::Relu>(branch);
    return branch;
}

std::shared_ptr<Model> make_concat_model(const OutputVector& branches, const BranchConfig& config, const ParameterVector& params) {
    auto concat = std::make_shared<op::v0::Concat>(branches, config.concat_axis);
    return std::make_shared<Model>(OutputVector{concat}, params);
}

// Branch i gathers slice indices[i] of the input
std::shared_ptr<Model> getModel(const BranchConfig& config, std::vector<int64_t> indices = {}) {
    if (indices.empty())
        for (size_t i = 0; i < config.replicas; ++i)
            indices.push_back(static_cast<int64_t>(i));
    auto data = make_data(config);
    const auto weights = make_weights(config);
    const auto bias = make_bias();
    OutputVector branches;
    for (const auto index : indices)
        branches.push_back(make_branch(make_gather(data, index, config), weights, bias, config));
    return make_concat_model(branches, config, {data});
}

// Gather(0..N-1) -> Concat(axis 0) -> one branch on [N, ...] -> Transpose -> Reshape
std::shared_ptr<Model> getModelRef(const BranchConfig& config) {
    auto data = make_data(config);
    OutputVector slices;
    for (size_t i = 0; i < config.replicas; ++i)
        slices.push_back(make_gather(data, static_cast<int64_t>(i), config));
    auto batched_input = std::make_shared<op::v0::Concat>(slices, 0);
    auto batched = make_branch(batched_input, make_weights(config), make_bias(), config);

    const auto branch_shape = batched.get_shape();
    std::vector<int64_t> order;
    std::vector<int64_t> output_shape{1};
    for (int64_t axis = 1; axis < 4; ++axis) {
        if (axis == config.concat_axis)
            order.push_back(0);
        order.push_back(axis);
        output_shape.push_back(static_cast<int64_t>(branch_shape[axis] * (axis == config.concat_axis ? config.replicas : 1)));
    }
    auto transpose = std::make_shared<op::v1::Transpose>(batched, op::v0::Constant::create(element::i64, Shape{4}, order));
    auto reshape = std::make_shared<op::v1::Reshape>(transpose, op::v0::Constant::create(element::i64, Shape{4}, output_shape), false);
    return std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
}

class BatchReplicatedBranchesTests : public TransformationTestsF {
public:
    BatchReplicatedBranchesTests() {
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    }
};

class BatchReplicatedBranchesParamTests : public BatchReplicatedBranchesTests, public WithParamInterface<BranchConfig> {
public:
    static std::string getTestCaseName(const TestParamInfo<BranchConfig>& info) {
        std::ostringstream ss;
        ss << info.param;
        return ss.str();
    }
};

TEST_P(BatchReplicatedBranchesParamTests, BatchesBranches) {
    model = getModel(GetParam());
    model_ref = getModelRef(GetParam());
}

BranchConfig make_config(size_t replicas,
                         int64_t replica_axis,
                         int64_t concat_axis,
                         bool decompress_weights,
                         bool max_pool,
                         bool bias,
                         bool relu,
                         bool port_rt_info = false) {
    return {replicas, replica_axis, concat_axis, decompress_weights, max_pool, bias, relu, port_rt_info};
}

INSTANTIATE_TEST_SUITE_P(BatchReplicatedBranches,
                         BatchReplicatedBranchesParamTests,
                         Values(make_config(4, 1, 3, false, false, false, false),
                                make_config(4, 1, 3, true, true, true, true),
                                make_config(2, 0, 1, false, false, true, false),
                                make_config(3, 2, 2, false, true, false, true),
                                make_config(4, 1, 3, false, false, false, false, true)),
                         BatchReplicatedBranchesParamTests::getTestCaseName);

TEST_F(BatchReplicatedBranchesTests, IsIdempotent) {
    const BranchConfig config{};
    model = getModel(config);
    model_ref = getModelRef(config);
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
}

TEST_F(BatchReplicatedBranchesTests, BatchesBranchesWithDistinctEqualWeights) {
    const BranchConfig config{};
    auto data = make_data(config);
    OutputVector branches;
    for (size_t i = 0; i < config.replicas; ++i)
        branches.push_back(make_branch(make_gather(data, static_cast<int64_t>(i), config), make_weights(config), make_bias(), config));
    model = make_concat_model(branches, config, {data});
    model_ref = getModelRef(config);
}

TEST_F(BatchReplicatedBranchesTests, RejectsNonExactReplicaOrder) {
    model = getModel(BranchConfig{}, {0, 1, 3, 2});
}

TEST_F(BatchReplicatedBranchesTests, RejectsPartialReplicaDimension) {
    model = getModel(BranchConfig{}, {0, 1, 2});
}

TEST_F(BatchReplicatedBranchesTests, RejectsConcatOnBatchAxis) {
    BranchConfig config{};
    config.concat_axis = 0;
    model = getModel(config);
}

TEST_F(BatchReplicatedBranchesTests, RejectsDifferentBranchTopology) {
    BranchConfig config{};
    auto data = make_data(config);
    const auto weights = make_weights(config);
    OutputVector branches;
    for (size_t i = 0; i < config.replicas; ++i) {
        config.relu = i == 2;
        branches.push_back(make_branch(make_gather(data, static_cast<int64_t>(i), config), weights, make_bias(), config));
    }
    model = make_concat_model(branches, config, {data});
}

TEST_F(BatchReplicatedBranchesTests, RejectsDifferentWeights) {
    const BranchConfig config{};
    auto data = make_data(config);
    OutputVector branches;
    for (size_t i = 0; i < config.replicas; ++i)
        branches.push_back(make_branch(make_gather(data, static_cast<int64_t>(i), config), make_weights(config, i == 1 ? 1.0f : 0.5f), make_bias(), config));
    model = make_concat_model(branches, config, {data});
}

TEST_F(BatchReplicatedBranchesTests, RejectsDynamicShape) {
    const BranchConfig config{};
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 4, 3, Dimension::dynamic(), 8});
    const auto weights = make_weights(config);
    OutputVector branches;
    for (size_t i = 0; i < config.replicas; ++i)
        branches.push_back(make_branch(make_gather(data, static_cast<int64_t>(i), config), weights, make_bias(), config));
    model = make_concat_model(branches, config, {data});
}

TEST_F(BatchReplicatedBranchesTests, RejectsAddWithoutNumpyBroadcast) {
    const BranchConfig config{};
    auto data = make_data(config);
    const auto weights = make_weights(config);
    const auto bias = op::v0::Constant::create(element::f32, Shape{1, 4, 4, 4}, {0.1f});
    OutputVector branches;
    for (size_t i = 0; i < config.replicas; ++i) {
        auto conv = make_branch(make_gather(data, static_cast<int64_t>(i), config), weights, bias, config);
        branches.push_back(std::make_shared<op::v1::Add>(conv, bias, op::AutoBroadcastType::NONE));
    }
    model = make_concat_model(branches, config, {data});
}

TEST_F(BatchReplicatedBranchesTests, RejectsExternalConsumer) {
    const BranchConfig config{};
    model = getModel(config);
    const auto branch = model->get_results()[0]->get_input_node_shared_ptr(0)->input_value(0);
    model->add_results({std::make_shared<op::v0::Result>(std::make_shared<op::v0::Relu>(branch))});
}

TEST_F(BatchReplicatedBranchesTests, RejectsShapeOfConsumer) {
    const BranchConfig config{};
    model = getModel(config);
    const auto branch = model->get_results()[0]->get_input_node_shared_ptr(0)->input_value(0);
    model->add_results({std::make_shared<op::v0::Result>(std::make_shared<op::v3::ShapeOf>(branch))});
}

TEST_F(BatchReplicatedBranchesTests, RejectsUsedMaxPoolIndices) {
    BranchConfig config{};
    config.max_pool = true;
    model = getModel(config);
    const auto max_pool = model->get_results()[0]->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0);
    ASSERT_TRUE(is_type<op::v8::MaxPool>(max_pool));
    model->add_results({std::make_shared<op::v0::Result>(max_pool->output(1))});
}

TEST_F(BatchReplicatedBranchesTests, RejectsControlDependency) {
    const BranchConfig config{};
    model = getModel(config);
    const auto conv = model->get_results()[0]->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(1);
    ASSERT_TRUE(is_type<op::v1::Convolution>(conv));
    conv->add_control_dependency(model->get_parameters()[0]);
}

TEST_F(BatchReplicatedBranchesTests, RejectsTypeRelaxedBranch) {
    // Comparator can't compare TypeVector attributes of TypeRelaxed nodes
    comparator.disable(FunctionsComparator::ATTRIBUTES);
    const BranchConfig config{};
    auto data = make_data(config);
    const auto weights = make_weights(config);
    OutputVector branches;
    for (size_t i = 0; i < config.replicas; ++i) {
        auto gather = make_gather(data, static_cast<int64_t>(i), config);
        branches.push_back(std::make_shared<op::TypeRelaxed<op::v1::Convolution>>(element::TypeVector{element::f32, element::f32},
                                                                                  element::TypeVector{element::f32},
                                                                                  gather,
                                                                                  weights,
                                                                                  Strides{2, 2},
                                                                                  CoordinateDiff{0, 0},
                                                                                  CoordinateDiff{0, 0},
                                                                                  Strides{1, 1}));
    }
    model = make_concat_model(branches, config, {data});
}

TEST_F(BatchReplicatedBranchesTests, RejectsTypeRelaxedConcat) {
    // Comparator can't compare TypeVector attributes of TypeRelaxed nodes
    comparator.disable(FunctionsComparator::ATTRIBUTES);
    const BranchConfig config{};
    auto data = make_data(config);
    const auto weights = make_weights(config);
    OutputVector branches;
    for (size_t i = 0; i < config.replicas; ++i)
        branches.push_back(make_branch(make_gather(data, static_cast<int64_t>(i), config), weights, make_bias(), config));
    auto concat = std::make_shared<op::TypeRelaxed<op::v0::Concat>>(element::TypeVector(config.replicas, element::f32),
                                                                    element::TypeVector{element::f32},
                                                                    branches,
                                                                    config.concat_axis);
    model = std::make_shared<Model>(OutputVector{concat}, ParameterVector{data});
}

}  // namespace
