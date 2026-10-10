// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/batch_replicated_branches.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace {

struct UnknownRuntimeInfo {
    int value;
    std::vector<std::string> labels;
};

std::shared_ptr<ov::Model> make_camera_model(const std::vector<int64_t>& indices,
                                             bool add_topology_difference = false,
                                             bool dynamic_camera_height = false,
                                             bool add_branch_bias = false,
                                             bool nonzero_weights = false,
                                             bool different_branch_runtime_info = false,
                                             ov::op::AutoBroadcastSpec add_broadcast =
                                                 ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY),
                                             bool add_unknown_runtime_info = false,
                                             bool use_nonzero_endpoint_output = false,
                                             bool add_decompression_metadata = false,
                                             bool use_distinct_endpoint_names = false,
                                             bool use_v14_max_pool = false,
                                             bool add_external_endpoint_user = false) {
    const auto input_shape = dynamic_camera_height
                                 ? ov::PartialShape{1, 4, 3, ov::Dimension::dynamic(), 640}
                                 : ov::PartialShape{1, 4, 3, 480, 640};
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {int64_t{1}});
    const ov::Shape weights_shape{512, 3, 32, 32};
    std::shared_ptr<ov::op::v0::Constant> weights;
    if (nonzero_weights) {
        std::vector<float> values(ov::shape_size(weights_shape), 0.0f);
        for (size_t output_channel = 0; output_channel < weights_shape[0]; ++output_channel) {
            values[output_channel * weights_shape[1] * weights_shape[2] * weights_shape[3]] =
                0.5f + static_cast<float>(output_channel) * 0.001f;
        }
        weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, weights_shape, values);
    } else {
        weights = ov::op::v0::Constant::create(ov::element::f32, weights_shape, {0.0f});
    }

    std::shared_ptr<ov::op::v0::Constant> bias;
    if (add_branch_bias) {
        const ov::Shape bias_shape =
            add_broadcast.m_type == ov::op::AutoBroadcastType::NONE ? ov::Shape{1, 512, 15, 20}
                                                                      : ov::Shape{512, 15, 20};
        std::vector<float> bias_values(ov::shape_size(bias_shape), 0.0f);
        if (nonzero_weights) {
            for (size_t output_channel = 0; output_channel < 512; ++output_channel) {
                for (size_t y = 0; y < 15; ++y) {
                    for (size_t x = 0; x < 20; ++x) {
                        const auto index = (output_channel * 15 + y) * 20 + x;
                        bias_values[index] = 0.1f + static_cast<float>(output_channel) * 0.0001f +
                                             static_cast<float>(y) * 0.00001f + static_cast<float>(x) * 0.000001f;
                    }
                }
            }
        }
        bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, bias_shape, bias_values);
    }

    ov::OutputVector branches;
    branches.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        auto index = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {indices[i]});
        auto gather = std::make_shared<ov::op::v8::Gather>(input, index, axis, 0);
        if (use_distinct_endpoint_names) {
            gather->output(0).set_names({"camera_input_" + std::to_string(i)});
        }
        ov::Output<ov::Node> branch_weights = weights;
        if (add_decompression_metadata) {
            auto weight_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
            ov::mark_as_decompression(weight_convert);
            branch_weights = weight_convert;
        }
        auto convolution = std::make_shared<ov::op::v1::Convolution>(
            gather,
            branch_weights,
            ov::Strides{32, 32},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0},
            ov::Strides{1, 1});
        if (different_branch_runtime_info && i == 1) {
            convolution->get_rt_info()["camera_branch"] = static_cast<int64_t>(i);
        }
        ov::Output<ov::Node> branch = convolution;
        if (add_unknown_runtime_info && i < 2) {
            convolution->get_rt_info()["camera_branch"] = int64_t{7};
            convolution->get_rt_info()["camera_label"] = std::string{"camera"};
            convolution->get_rt_info()["camera_payload"] =
                UnknownRuntimeInfo{static_cast<int>(i), {"front", "wide"}};
        }
        if (use_v14_max_pool) {
            auto max_pool = std::make_shared<ov::op::v14::MaxPool>(branch,
                                                                   ov::Strides{1, 1},
                                                                   ov::Strides{1, 1},
                                                                   ov::Shape{0, 0},
                                                                   ov::Shape{0, 0},
                                                                   ov::Shape{1, 1},
                                                                   ov::op::RoundingType::FLOOR,
                                                                   ov::op::PadType::EXPLICIT,
                                                                   ov::element::i64,
                                                                   2);
            branch = max_pool->output(0);
        }
        if (use_nonzero_endpoint_output) {
            auto max_pool = std::make_shared<ov::op::v8::MaxPool>(branch,
                                                                   ov::Strides{1, 1},
                                                                   ov::Strides{1, 1},
                                                                   ov::Shape{0, 0},
                                                                   ov::Shape{0, 0},
                                                                   ov::Shape{1, 1});
            branch = max_pool->output(1);
        }
        if (add_branch_bias) {
            branch = std::make_shared<ov::op::v1::Add>(branch, bias, add_broadcast);
        }
        if (add_topology_difference && i == 2) {
            branch = std::make_shared<ov::op::v0::Relu>(branch);
        }
        if (use_distinct_endpoint_names) {
            branch.set_names({"camera_endpoint_" + std::to_string(i)});
        }
        branches.push_back(branch);
    }

    auto output_concat = std::make_shared<ov::op::v0::Concat>(branches, 3);
    output_concat->set_friendly_name("camera_output_concat");
    output_concat->output(0).set_names({"camera_output"});
    auto result = std::make_shared<ov::op::v0::Result>(output_concat);
    ov::ResultVector results{result};
    if (add_external_endpoint_user) {
        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(branches.front());
        results.push_back(std::make_shared<ov::op::v0::Result>(shape_of));
    }
    return std::make_shared<ov::Model>(results, ov::ParameterVector{input});
}

class CameraGraphEvaluator {
public:
    explicit CameraGraphEvaluator(const ov::Tensor& input) : m_input(input) {}

    ov::Tensor evaluate(const std::shared_ptr<ov::Node>& node) {
        const auto found = m_values.find(node.get());
        if (found != m_values.end()) {
            return found->second;
        }

        ov::Tensor value;
        if (ov::is_type<ov::op::v0::Parameter>(node)) {
            value = m_input;
        } else if (const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            value = constant->get_tensor_view();
        } else if (const auto convolution = ov::as_type_ptr<ov::op::v1::Convolution>(node)) {
            value = evaluate_convolution(convolution);
        } else {
            ov::TensorVector inputs;
            inputs.reserve(node->get_input_size());
            for (const auto& input : node->input_values()) {
                inputs.push_back(evaluate(input.get_node_shared_ptr()));
            }

            ov::TensorVector outputs{
                ov::Tensor(node->get_output_element_type(0), node->get_output_shape(0))};
            if (!node->evaluate(outputs, inputs)) {
                throw std::runtime_error(std::string("Camera graph evaluator does not support ") +
                                         node->get_type_name());
            }
            value = outputs.front();
        }

        m_values.emplace(node.get(), value);
        return value;
    }

private:
    struct WeightEntry {
        size_t input_channel;
        size_t y;
        size_t x;
        float value;
    };

    ov::Tensor evaluate_convolution(const std::shared_ptr<ov::op::v1::Convolution>& convolution) {
        const auto data = evaluate(convolution->input_value(0).get_node_shared_ptr());
        const auto weights = evaluate(convolution->input_value(1).get_node_shared_ptr());
        const auto data_shape = data.get_shape();
        const auto weights_shape = weights.get_shape();
        const auto output_shape = convolution->get_output_shape(0);

        if (data_shape.size() != 4 || weights_shape != ov::Shape{512, 3, 32, 32} ||
            convolution->get_strides() != ov::Strides{32, 32} ||
            convolution->get_pads_begin() != ov::CoordinateDiff{0, 0} ||
            convolution->get_pads_end() != ov::CoordinateDiff{0, 0} ||
            convolution->get_dilations() != ov::Strides{1, 1}) {
            throw std::runtime_error("Unexpected convolution in camera graph evaluator");
        }

        ov::Tensor output(convolution->get_output_element_type(0), output_shape);
        const auto* data_values = data.data<const float>();
        const auto* weight_values = weights.data<const float>();
        auto* output_values = output.data<float>();

        std::vector<std::vector<WeightEntry>> nonzero_weights(weights_shape[0]);
        for (size_t output_channel = 0; output_channel < weights_shape[0]; ++output_channel) {
            for (size_t input_channel = 0; input_channel < weights_shape[1]; ++input_channel) {
                for (size_t y = 0; y < weights_shape[2]; ++y) {
                    for (size_t x = 0; x < weights_shape[3]; ++x) {
                        const auto weight_index =
                            ((output_channel * weights_shape[1] + input_channel) * weights_shape[2] + y) *
                                weights_shape[3] +
                            x;
                        if (weight_values[weight_index] != 0.0f) {
                            nonzero_weights[output_channel].push_back(
                                {input_channel, y, x, weight_values[weight_index]});
                        }
                    }
                }
            }
        }

        for (size_t batch = 0; batch < output_shape[0]; ++batch) {
            for (size_t output_channel = 0; output_channel < output_shape[1]; ++output_channel) {
                for (size_t y = 0; y < output_shape[2]; ++y) {
                    for (size_t x = 0; x < output_shape[3]; ++x) {
                        float sum = 0.0f;
                        for (const auto& weight : nonzero_weights[output_channel]) {
                            const auto input_index =
                                ((batch * data_shape[1] + weight.input_channel) * data_shape[2] +
                                 y * 32 + weight.y) *
                                    data_shape[3] +
                                x * 32 + weight.x;
                            sum += data_values[input_index] * weight.value;
                        }
                        const auto output_index =
                            ((batch * output_shape[1] + output_channel) * output_shape[2] + y) * output_shape[3] + x;
                        output_values[output_index] = sum;
                    }
                }
            }
        }
        return output;
    }

    const ov::Tensor& m_input;
    std::unordered_map<const ov::Node*, ov::Tensor> m_values;
};

// The core reference evaluator has no convolution implementation. Evaluate
// this fixed camera graph directly, using a sparse convolution implementation
// and the actual OpenVINO evaluators for every other operation.
std::vector<float> evaluate_camera_model(const std::shared_ptr<ov::Model>& model, const ov::Tensor& input) {
    CameraGraphEvaluator evaluator(input);
    const auto output = evaluator.evaluate(model->get_results().front()->input_value(0).get_node_shared_ptr());
    return std::vector<float>(output.data<const float>(), output.data<const float>() + output.get_size());
}

}  // namespace

TEST(BatchReplicatedBranches, FusesExactCameraOrderAndRepackagesWidth) {
    auto model = make_camera_model({0, 1, 2, 3});

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    manager.run_passes(model);

    auto result_input = model->get_results().front()->input_value(0);
    auto repacked = ov::as_type_ptr<ov::op::v1::Reshape>(result_input.get_node_shared_ptr());
    ASSERT_NE(repacked, nullptr);
    EXPECT_EQ(repacked->get_friendly_name(), "camera_output_concat");
    EXPECT_TRUE(repacked->output(0).get_names().count("camera_output"));

    auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(repacked->input_value(0).get_node_shared_ptr());
    ASSERT_NE(transpose, nullptr);
    auto batched_branch = ov::as_type_ptr<ov::op::v1::Convolution>(
        transpose->input_value(0).get_node_shared_ptr());
    ASSERT_NE(batched_branch, nullptr);
    const ov::PartialShape expected_batched_shape{4, 512, 15, 20};
    EXPECT_EQ(batched_branch->get_output_partial_shape(0), expected_batched_shape);

    auto camera_batch = ov::as_type_ptr<ov::op::v0::Concat>(
        batched_branch->input_value(0).get_node_shared_ptr());
    ASSERT_NE(camera_batch, nullptr);
    ASSERT_EQ(camera_batch->get_input_size(), 4);
    for (size_t i = 0; i < 4; ++i) {
        auto gather = ov::as_type_ptr<ov::op::v8::Gather>(
            camera_batch->input_value(i).get_node_shared_ptr());
        ASSERT_NE(gather, nullptr);
        auto index = ov::as_type_ptr<ov::op::v0::Constant>(gather->input_value(1).get_node_shared_ptr());
        ASSERT_NE(index, nullptr);
        EXPECT_EQ(index->cast_vector<int64_t>()[0],
                  static_cast<int64_t>(i));
    }
}

TEST(BatchReplicatedBranches, IsIdempotentAfterCameraBatching) {
    auto model = make_camera_model({0, 1, 2, 3});

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    manager.run_passes(model);

    const auto first_result_input = model->get_results().front()->input_value(0);
    const auto first_ops = model->get_ordered_ops();
    std::vector<std::string> first_friendly_names;
    std::vector<std::vector<std::pair<const ov::Node*, size_t>>> first_inputs;
    first_friendly_names.reserve(first_ops.size());
    first_inputs.reserve(first_ops.size());
    for (const auto& node : first_ops) {
        first_friendly_names.push_back(node->get_friendly_name());
        std::vector<std::pair<const ov::Node*, size_t>> inputs;
        inputs.reserve(node->get_input_size());
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            inputs.emplace_back(node->input_value(i).get_node_shared_ptr().get(), node->input_value(i).get_index());
        }
        first_inputs.push_back(std::move(inputs));
    }

    const auto count_named_nodes = [](const std::vector<std::shared_ptr<ov::Node>>& nodes,
                                      const std::string& friendly_name) {
        size_t count = 0;
        for (const auto& node : nodes) {
            count += node->get_friendly_name() == friendly_name;
        }
        return count;
    };
    const std::string camera_batch_name = "camera_output_concat/camera_batch";
    const std::string camera_repack_name = "camera_output_concat/camera_repack_transpose";
    EXPECT_EQ(count_named_nodes(first_ops, camera_batch_name), 1u);
    EXPECT_EQ(count_named_nodes(first_ops, camera_repack_name), 1u);

    manager.run_passes(model);

    const auto second_result_input = model->get_results().front()->input_value(0);
    const auto second_ops = model->get_ordered_ops();
    ASSERT_EQ(second_ops.size(), first_ops.size());
    ASSERT_EQ(first_result_input.get_node_shared_ptr().get(), second_result_input.get_node_shared_ptr().get());
    EXPECT_EQ(first_result_input.get_index(), second_result_input.get_index());
    EXPECT_EQ(count_named_nodes(second_ops, camera_batch_name), 1u);
    EXPECT_EQ(count_named_nodes(second_ops, camera_repack_name), 1u);

    for (size_t i = 0; i < first_ops.size(); ++i) {
        ASSERT_EQ(second_ops[i].get(), first_ops[i].get());
        EXPECT_EQ(second_ops[i]->get_friendly_name(), first_friendly_names[i]);
        ASSERT_EQ(second_ops[i]->get_input_size(), first_inputs[i].size());
        for (size_t j = 0; j < first_inputs[i].size(); ++j) {
            EXPECT_EQ(second_ops[i]->input_value(j).get_node_shared_ptr().get(), first_inputs[i][j].first);
            EXPECT_EQ(second_ops[i]->input_value(j).get_index(), first_inputs[i][j].second);
        }
    }
}

TEST(BatchReplicatedBranches, ActivatesWithDistinctEndpointNamesAndDecompression) {
    auto model = make_camera_model({0, 1, 2, 3},
                                   false,
                                   false,
                                   false,
                                   false,
                                   false,
                                   ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY),
                                   false,
                                   false,
                                   true,
                                   true,
                                   true,
                                   true);

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    manager.run_passes(model);

    const auto repacked =
        ov::as_type_ptr<ov::op::v1::Reshape>(model->get_results().front()->input_value(0).get_node_shared_ptr());
    ASSERT_NE(repacked, nullptr);
    EXPECT_TRUE(repacked->output(0).get_names().count("camera_output"));

    const auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(repacked->input_value(0).get_node_shared_ptr());
    ASSERT_NE(transpose, nullptr);
    const auto batched_max_pool =
        ov::as_type_ptr<ov::op::v14::MaxPool>(transpose->input_value(0).get_node_shared_ptr());
    ASSERT_NE(batched_max_pool, nullptr);
    const auto batched_branch =
        ov::as_type_ptr<ov::op::v1::Convolution>(batched_max_pool->input_value(0).get_node_shared_ptr());
    ASSERT_NE(batched_branch, nullptr);
    EXPECT_TRUE(batched_max_pool->output(0).get_names().count("camera_endpoint_0"));
    ASSERT_EQ(model->get_results().size(), 2);
    EXPECT_TRUE(ov::is_type<ov::op::v3::ShapeOf>(model->get_results()[1]->input_value(0).get_node_shared_ptr()));

    const auto decompressed_weights =
        ov::as_type_ptr<ov::op::v0::Convert>(batched_branch->input_value(1).get_node_shared_ptr());
    ASSERT_NE(decompressed_weights, nullptr);
    EXPECT_TRUE(ov::is_decompression(decompressed_weights));
}

TEST(BatchReplicatedBranches, RejectsNonExactCameraIndices) {
    auto model = make_camera_model({0, 1, 3, 2});

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    manager.run_passes(model);

    EXPECT_NE(ov::as_type_ptr<ov::op::v0::Concat>(model->get_results().front()->input_value(0)
                                                      .get_node_shared_ptr()),
              nullptr);
}

TEST(BatchReplicatedBranches, RejectsDifferentBranchTopology) {
    auto model = make_camera_model({0, 1, 2, 3}, true);

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    manager.run_passes(model);

    EXPECT_NE(ov::as_type_ptr<ov::op::v0::Concat>(model->get_results().front()->input_value(0)
                                                      .get_node_shared_ptr()),
              nullptr);
}

TEST(BatchReplicatedBranches, RejectsDynamicCameraShape) {
    auto model = make_camera_model({0, 1, 2, 3}, false, true);

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    manager.run_passes(model);

    EXPECT_NE(ov::as_type_ptr<ov::op::v0::Concat>(model->get_results().front()->input_value(0)
                                                      .get_node_shared_ptr()),
              nullptr);
}

TEST(BatchReplicatedBranches, RejectsAddWithoutSafeNumpyBatchBroadcast) {
    auto model = make_camera_model({0, 1, 2, 3}, false, false, true, false, false,
                                   ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NONE));

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    manager.run_passes(model);

    EXPECT_NE(ov::as_type_ptr<ov::op::v0::Concat>(model->get_results().front()->input_value(0)
                                                      .get_node_shared_ptr()),
              nullptr);
}

TEST(BatchReplicatedBranches, RejectsDifferentBranchRuntimeInfo) {
    auto model = make_camera_model({0, 1, 2, 3}, false, false, false, false, true);

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    manager.run_passes(model);

    EXPECT_NE(ov::as_type_ptr<ov::op::v0::Concat>(model->get_results().front()->input_value(0)
                                                      .get_node_shared_ptr()),
              nullptr);
}

TEST(BatchReplicatedBranches, RejectsUnknownRuntimeInfoWithoutThrowing) {
    auto model = make_camera_model({0, 1, 2, 3}, false, false, false, false, false,
                                   ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY), true);

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    EXPECT_NO_THROW(manager.run_passes(model));

    EXPECT_NE(ov::as_type_ptr<ov::op::v0::Concat>(model->get_results().front()->input_value(0)
                                                      .get_node_shared_ptr()),
              nullptr);
}

TEST(BatchReplicatedBranches, RejectsNonzeroEndpointOutputPort) {
    auto model = make_camera_model({0, 1, 2, 3}, false, false, false, false, false,
                                   ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY),
                                   false,
                                   true);

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    EXPECT_NO_THROW(manager.run_passes(model));

    const auto concat =
        ov::as_type_ptr<ov::op::v0::Concat>(model->get_results().front()->input_value(0).get_node_shared_ptr());
    ASSERT_NE(concat, nullptr);
    for (size_t i = 0; i < concat->get_input_size(); ++i) {
        EXPECT_EQ(concat->input_value(i).get_index(), 1);
    }
}

TEST(BatchReplicatedBranches, RejectsMismatchedCameraDimensionSize) {
    // The replica count is derived from the graph, not fixed to four. This
    // model's camera dimension still holds 4 slots (per make_camera_model's
    // fixed input shape), but only 3 of them are gathered into the Concat,
    // so the Gather axis-dimension size (4) no longer matches the replica
    // count (3). That structural mismatch, not a hard-coded "four" check,
    // must still leave the Concat untouched.
    auto model = make_camera_model({0, 1, 2});

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    manager.run_passes(model);

    EXPECT_NE(ov::as_type_ptr<ov::op::v0::Concat>(model->get_results().front()->input_value(0)
                                                      .get_node_shared_ptr()),
              nullptr);
}

TEST(BatchReplicatedBranches, FusesNonFourWayReplicaCountAndNonDefaultAxes) {
    // Proves the pass generalizes beyond the four-camera / gather-axis-1 /
    // concat-axis-3 reference topology: 3 replicas, gathered on axis 2 of a
    // rank-5 input, concatenated on axis 1.
    constexpr int64_t camera_axis = 2;
    constexpr int64_t concat_axis = 1;
    constexpr size_t replica_count = 3;

    ov::Shape input_shape(5, 3);
    input_shape[0] = 1;
    input_shape[camera_axis] = replica_count;
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {camera_axis});

    ov::OutputVector branches;
    for (size_t i = 0; i < replica_count; ++i) {
        auto index = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(i)});
        auto gather = std::make_shared<ov::op::v8::Gather>(input, index, axis, 0);
        branches.push_back(std::make_shared<ov::op::v0::Relu>(gather)->output(0));
    }
    auto output_concat = std::make_shared<ov::op::v0::Concat>(branches, concat_axis);
    auto result = std::make_shared<ov::op::v0::Result>(output_concat);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    manager.run_passes(model);

    auto repacked = ov::as_type_ptr<ov::op::v1::Reshape>(model->get_results().front()->input_value(0)
                                                             .get_node_shared_ptr());
    ASSERT_NE(repacked, nullptr);
    const ov::PartialShape expected_output_shape{1, replica_count * 3, 3, 3};
    EXPECT_EQ(repacked->get_output_partial_shape(0), expected_output_shape);
}

TEST(BatchReplicatedBranches, NumericalRegressionPreservesCameraOrder) {
    auto reference_model = make_camera_model({0, 1, 2, 3}, false, false, true, true);
    auto transformed_model = reference_model->clone();

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::BatchReplicatedBranches>();
    manager.run_passes(transformed_model);

    auto transformed_output = transformed_model->get_results().front()->input_value(0).get_node_shared_ptr();
    ASSERT_NE(ov::as_type_ptr<ov::op::v1::Reshape>(transformed_output), nullptr);
    auto transformed_transpose =
        ov::as_type_ptr<ov::op::v1::Transpose>(transformed_output->input_value(0).get_node_shared_ptr());
    ASSERT_NE(transformed_transpose, nullptr);
    ASSERT_NE(ov::as_type_ptr<ov::op::v1::Add>(
                  transformed_transpose->input_value(0).get_node_shared_ptr()),
              nullptr);

    const ov::Shape input_shape{1, 4, 3, 480, 640};
    ov::Tensor input(ov::element::f32, input_shape);
    auto* input_values = input.data<float>();
    for (size_t camera = 0; camera < input_shape[1]; ++camera) {
        for (size_t channel = 0; channel < input_shape[2]; ++channel) {
            for (size_t y = 0; y < input_shape[3]; ++y) {
                for (size_t x = 0; x < input_shape[4]; ++x) {
                    const auto index = ((camera * input_shape[2] + channel) * input_shape[3] + y) * input_shape[4] + x;
                    input_values[index] = static_cast<float>(camera + 1) * 0.25f +
                                          static_cast<float>(channel + 1) * 0.01f +
                                          static_cast<float>(y) * 0.001f + static_cast<float>(x) * 0.00001f;
                }
            }
        }
    }

    const auto reference_values = evaluate_camera_model(reference_model, input);
    const auto transformed_values = evaluate_camera_model(transformed_model, input);
    ASSERT_EQ(reference_values.size(), transformed_values.size());
    for (size_t i = 0; i < reference_values.size(); ++i) {
        EXPECT_NEAR(reference_values[i], transformed_values[i], 1e-6f) << "at output index " << i;
    }

    // The transformed layout is [C,H,B,W] before the final reshape. The
    // final width therefore contains camera 0, 1, 2, and 3 in that order.
    constexpr size_t output_channels = 512;
    constexpr size_t output_height = 15;
    constexpr size_t output_width = 20;
    for (size_t camera = 0; camera < 4; ++camera) {
        const auto index = (0 * output_height + 0) * (4 * output_width) + camera * output_width;
        EXPECT_NEAR(reference_values[index], transformed_values[index], 1e-6f);
    }
    const auto first_camera_value = transformed_values[0];
    for (size_t camera = 1; camera < 4; ++camera) {
        const auto index = camera * output_width;
        EXPECT_GT(transformed_values[index], first_camera_value) << "camera ordering changed at camera " << camera;
    }
    EXPECT_EQ(transformed_values.size(), output_channels * output_height * 4 * output_width);
}
