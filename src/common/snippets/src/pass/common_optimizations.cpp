// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/common_optimizations.hpp"

#include "snippets/pass/fq_decomposition.hpp"
#include "snippets/pass/softmax_reshape_elimination.hpp"
#include "snippets/pass/explicit_transpose_matmul_inputs.hpp"
#include "snippets/pass/transpose_decomposition.hpp"
#include "snippets/pass/fuse_transpose_brgemm.hpp"
#include "snippets/pass/transform_convert.hpp"
#include "snippets/pass/validate.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/itt.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace pass {

namespace {
size_t get_lcm(size_t a, size_t b) {
    std::function<size_t(size_t, size_t)> get_gcd;
    get_gcd = [&get_gcd](size_t a, size_t b) {
        if (b == 0)
            return a;
        return get_gcd(b, a % b);
    };
    return a / get_gcd(a, b) * b;
}

bool is_supported_matmul_for_split_dim_m_optimization(const std::shared_ptr<const ov::Node>& node) {
    const auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
    return matmul && !matmul->get_transpose_a() && !matmul->is_dynamic() && node->get_shape().size() == 3; // It's needed only for 3D MHA patterns
}
}  // namespace

bool CommonOptimizations::CanOptimizeParallelWA(const std::shared_ptr<const ov::Node>& node, size_t concurrency) {
    if (!is_supported_matmul_for_split_dim_m_optimization(node))
        return false;
    const auto mm_shape = node->get_shape();
    const auto current_parallel_work_amount =
        std::accumulate(mm_shape.rbegin() + 2, mm_shape.rend(), size_t(1), std::multiplies<size_t>());
    const auto dim_M = *(mm_shape.rbegin() + 1);
    return (current_parallel_work_amount < concurrency) &&
           (current_parallel_work_amount * dim_M >= concurrency);
}

void CommonOptimizations::SplitDimensionM(const std::shared_ptr<ov::snippets::op::Subgraph>& subgraph, size_t concurrency) {
    // To increase parallelism work in 3D cases for MHA pattern,
    // we split 1st dimension (starting from 0th) into 2 new dimensions to get 4D Shapes where
    // - 0th and 1st dimensions are used in parallel scheduling,
    // - 2nd and 3rd dimensions are used in kernel
    // Note: 3D Patterns don't contain Transpose inside so the reshaping is valid

    // It's needed only for MHA patterns. Need to add support for common patterns
    if (!subgraph->has_domain_sensitive_ops())
        return;

    const auto& body = subgraph->body_ptr();
    const auto& parameters = body->get_parameters();
    // [107806]: If count of Parameters isn't equal to Subgraph inputs (it's possible case in general),
    //           we cannot garantee correct extraction since we don't have correct connections between body I/O and Subgraph I/O.
    OPENVINO_ASSERT(parameters.size() == subgraph->input_values().size(),
                    "Failed to extract unsupported transposes: the count of Parameters isn't equal to Subgraph inputs");

    // Need to find MatMul0 and check output shape
    const auto& ops = body->get_ordered_ops();
    const auto mm_it = std::find_if(ops.begin(), ops.end(),
                                    [](const std::shared_ptr<ov::Node>& node){ return ov::is_type<ov::op::v0::MatMul>(node); });
    if (mm_it == ops.end())
        return;

    const auto matmul0 = *mm_it;
    if (!is_supported_matmul_for_split_dim_m_optimization(matmul0))
        return;

    auto get_dim_M = [](const ov::Shape& shape) {
        return *(shape.rbegin() + 1);
    };

    const auto mm_shape = matmul0->get_shape();
    const auto m_dim = get_dim_M(mm_shape);  // M
    const auto batch_dim =
        std::accumulate(mm_shape.rbegin() + 2, mm_shape.rend(), size_t(1), std::multiplies<size_t>());  // B (batch)

    // We skip optimization if the current batch is optimal for concurrency
    const auto optimal_parallelism_work_amount = concurrency;
    if (batch_dim % optimal_parallelism_work_amount == 0)
        return;

    size_t batch_m_dim = 1;
    size_t new_m_dim = m_dim;

    auto is_optimized = [&](size_t batch_m_dim) {
        return batch_m_dim > 1;
    };

    // [ First Step ]
    // Need to find optimized dimension splitting: [b1..bk, m, n] -> [b1..bk, batch_m_dim, new_m_dim, n]
    // The work amount for parallelism should be divided by max thread count in ideal case
    // that all threads have the same full work amount (avoid of thread downtime)
    // If it's impossible, we select such values so that as many threads as possible have work (see [ Second Step ])
    // For example, there are 16 threads and shape [6, 512, 32]
    //              LCM(6, 16) = 48 <- ideal work amount for parallelism
    //              new_shape [6, 48 / 6, 512 / (48 / 6), 32 ] => [6, 8, 64, 32]
    //              Each thread has parallelism_work_amount = 6 * 8 / nthrs = 3
    const auto lcm = get_lcm(batch_dim, optimal_parallelism_work_amount);  // LCM(b, nthrs)
    const auto batch_dim_multiplier = lcm / batch_dim;  // LCM(b, nthrs) / b
    const auto needed_new_dim = m_dim / batch_dim_multiplier;  // m / (LCM(b, nthrs) / b) - needed factors of dimension m
    if (batch_dim_multiplier * needed_new_dim == m_dim && is_optimized(batch_dim_multiplier)) {
        batch_m_dim = batch_dim_multiplier;
        new_m_dim = needed_new_dim;
    } else {
        // [ Second Step ]
        // If we couldn't optimally split on the previous step, try the second step.
        // The algorithm finds the more optimal parallelism work amount [batch_dim * batch_m_dim],
        // where batch_m_dim is divisor of dimension M.
        // The optimal parallelism work amount means the case when as many threads as possible have work
        // For example, there are 8 threads and shape [5, 384, 32]
        //              768 = [2 x 192] = [3 x 128] = [4 x 96] = [6 x 64]
        //               - [5, 2, 192, 32] - WA = 10 = 8 + 2 (6 threads calculates once and 2 threads twice)
        //               - [5, 3, 128, 32] - WA = 15 = 8 + 7 (all threads have 2 kernel except one thread) <- the most optimal case
        //               - [5, 4,  96, 32] - WA = 20 = 8 x 2 + 4
        //               - [5, 6,  64, 32] - WA = 30 = 8 x 3 + 6
        // The most optimal and possible case is [5, 3, 128, 32] - almost all threads executes kernel twice
        // Heuristic value for a quick exit from the algorithm.
        // The value shows the number of threads in percentages that perform the most equal work
        const auto optimal_thread_num_percent = 0.8;
        size_t optimal_remainder = 1;
        auto get_remainder = [batch_dim, optimal_parallelism_work_amount](const size_t potential_batch_dim) {
            return (batch_dim * potential_batch_dim) % optimal_parallelism_work_amount;
        };

        auto update_optimal_params = [&](size_t divisor_0, size_t divisor_1) {
            const auto remainder = batch_dim * divisor_0 % optimal_parallelism_work_amount;
            if (remainder > optimal_remainder || remainder == 0) {
                optimal_remainder = remainder;
                batch_m_dim = divisor_0;
                new_m_dim = divisor_1;
            }
        };

        // Firstly we have shape [batch, 1, m_dim, smth].
        // So at the beginning we have parallel_work_amount = batch x 1
        optimal_remainder = get_remainder(1);
        const auto root = std::sqrt(m_dim) + 1;
        for (size_t divisor_0 = 2; divisor_0 < root; ++divisor_0) {
            const size_t divisor_1 = m_dim / divisor_0;
            if (divisor_0 * divisor_1 != m_dim)
                continue;

            update_optimal_params(divisor_0, divisor_1);
            update_optimal_params(divisor_1, divisor_0);
            if ((static_cast<float>(optimal_remainder) / static_cast<float>(optimal_parallelism_work_amount) > optimal_thread_num_percent) ||
                (optimal_remainder == 0)) {
                break;
            }
        }
    }

    OPENVINO_ASSERT(batch_m_dim * new_m_dim == m_dim, "Incorrect dimension M splitting!");
    // nothing to split
    if (!is_optimized(batch_m_dim))
        return;

    /***** Reshape insertion *****/

    // There are two Parameter variants:
    //  - Parameter on branches for Second input of MatMul - the shape should be only unsqueezed (add just 1)
    //  - Other Parameters (on First input of MatMuls and between) - the shape should be splitted on M dimension

    bool updated = false;
    std::set<std::shared_ptr<ov::op::v0::Parameter>> reshaped_params;

    auto insert_reshape = [&](const std::shared_ptr<ov::op::v0::Parameter>& param, const ov::Shape& new_shape) {
        const auto index = std::distance(parameters.begin(), std::find(parameters.begin(), parameters.end(), param));
        const auto shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{new_shape.size()}, new_shape);
        const auto reshape = std::make_shared<ov::op::v1::Reshape>(subgraph->input_value(index), shape_const, false);
        subgraph->input(index).replace_source_output(reshape);
        param->set_partial_shape(new_shape);
        reshaped_params.insert(param);
        updated = true;
    };

    auto get_updated_shape = [&](const ov::Shape& shape, bool split_m_dim) {
        const auto current_m_dim = get_dim_M(shape);
        OPENVINO_ASSERT(!split_m_dim || current_m_dim == 1 || current_m_dim == m_dim, "Incorrect shape for splitting!");
        ov::Shape new_shape = shape;
        if ((split_m_dim && current_m_dim == 1) || !split_m_dim) {
            new_shape.insert((new_shape.rbegin() + 2).base(), 1);
        } else {
            new_shape.insert((new_shape.rbegin() + 2).base(), batch_m_dim);
            *(new_shape.rbegin() + 1) = new_m_dim;
        }
        OPENVINO_ASSERT(ov::shape_size(new_shape) == ov::shape_size(shape), "Incorrect shape splitting!");
        return new_shape;
    };

    auto reshape_parameter = [&](const std::shared_ptr<ov::Node>& node, bool split_m_dim = true) {
        const auto param = ov::as_type_ptr<ov::op::v0::Parameter>(node);
        if (!param || reshaped_params.count(param) > 0)
            return;
        insert_reshape(param, get_updated_shape(param->get_partial_shape().get_shape(), split_m_dim));
    };

    auto update_matmul_second_branch = [&](const std::shared_ptr<ov::Node>& node) {
        auto parent = node->get_input_node_shared_ptr(1);
        while (!ov::is_type<ov::op::v0::Parameter>(parent)) {
            if (parent->get_input_size() > 1) {
                for (const auto& input_source : parent->input_values()) {
                    reshape_parameter(input_source.get_node_shared_ptr(), false);
                }
            }

            // [107731]: It's covered my MHA tokenization
            parent = parent->get_input_node_shared_ptr(0);
        }
        reshape_parameter(parent, false);
    };

    // Firstly, Unsqueeze parameters on second branches of MatMuls
    for (const auto& op : ops) {
        if (ov::is_type<ov::op::v0::MatMul>(op)) {
            update_matmul_second_branch(op);
        }
    }

    // Secondly, Update All M dimensions for remaining parameters
    for (const auto& param : parameters) {
        if (reshaped_params.count(param) == 0)
            reshape_parameter(param, true);
    }

    // Return the previous shape on outputs
    for (size_t i = 0; i < subgraph->get_output_size() && updated; ++i) {
        const auto output_shape = subgraph->get_output_shape(i);
        if (is_scalar(output_shape))
            continue;

        const auto& target_inputs = subgraph->get_output_target_inputs(i);
        const auto shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{output_shape.size()}, output_shape);
        const auto reshape = std::make_shared<ov::op::v1::Reshape>(subgraph->output(i), shape_const, false);
        // Save output name
        const auto original_output = body->get_results()[i]->get_input_node_shared_ptr(0);
        const auto original_name = original_output->get_friendly_name();
        reshape->set_friendly_name(original_name);
        original_output->set_friendly_name(original_name + "_original");

        for (const auto& input : target_inputs) {
            input.replace_source_output(reshape);
            // Result input tensor name was changed, the name has to be restored
            if (ov::is_type<ov::op::v0::Result>(input.get_node())) {
                input.get_tensor_ptr()->add_names(subgraph->output(i).get_tensor_ptr()->get_names());
            }
        }
        subgraph->output(i).get_tensor_ptr()->set_names({});
        updated = true;
    }
    subgraph->set_friendly_name(subgraph->get_friendly_name() + "_original");

    // Need to update inner Shapes and Softmax Axis
    if (updated) {
        for (const auto &op : ops) {
            if (const auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(op)) {
                softmax_v8->set_axis(-1);
            } else if (const auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(op)) {
                softmax_v1->set_axis(softmax_v1->get_output_partial_shape(0).size()); // since new_shape.size() = old_shape.size() + 1
            } else if (const auto broadcast = ov::as_type_ptr<ov::op::v1::Broadcast>(op)) {
                // Broadcast is tokenized only between MatMuls -> Split M dimension
                const auto shape_const = ov::as_type_ptr<ov::op::v0::Constant>(broadcast->input_value(1).get_node_shared_ptr());
                OPENVINO_ASSERT(shape_const, "SplitDimensionM expects Broadcast with Constant output shape");
                const auto new_shape = get_updated_shape(shape_const->cast_vector<size_t>(), true);
                broadcast->set_argument(1, std::make_shared<ov::op::v0::Constant>(shape_const->get_element_type(), ov::Shape{new_shape.size()}, new_shape));
            }
        }
        subgraph->validate_and_infer_types();
    }
}

void CommonOptimizations::ExtractConstants(const std::shared_ptr<ov::snippets::op::Subgraph>& subgraph) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ExtractConstants");
    auto body = subgraph->body_ptr();

    ParameterVector new_parameters;
    OutputVector new_external_inputs = subgraph->input_values();

    for (auto& op : body->get_ops()) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant || ov::shape_size(constant->get_shape()) == 1ul)
            continue;

        const auto child = constant->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
        if (op::Subgraph::constant_input_should_be_inside_body(child))
            continue;

        auto parameter = std::make_shared<ov::op::v0::Parameter>(constant->get_element_type(), constant->output(0).get_partial_shape());
        parameter->set_friendly_name(constant->get_friendly_name());
        ov::copy_runtime_info(constant, parameter);
        constant->output(0).replace(parameter->output(0));

        new_external_inputs.push_back(constant);
        new_parameters.push_back(parameter);
    }

    if (new_parameters.size() != 0) {
        body->add_parameters(new_parameters);
        body->validate_nodes_and_infer_types();
        subgraph->set_arguments(new_external_inputs);
    }
}

void CommonOptimizations::ExtractUnsupportedTransposes(const std::shared_ptr<op::Subgraph>& subgraph) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ExtractUnsupportedTransposes");
    const auto& body = subgraph->body_ptr();
    const auto parameters = body->get_parameters();
    // [107806]: If count of Parameters isn't equal to Subgraph inputs,
    //           we cannot guarantee correct extraction since we don't have correct connections between body I/O and Subgraph I/O.
    OPENVINO_ASSERT(parameters.size() == subgraph->input_values().size(),
                    "Failed to extract unsupported transposes: the count of Parameters isn't equal to Subgraph inputs");

    bool updated = false;
    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& parameter = parameters[i];
        const auto& consumers = parameter->get_output_target_inputs(0);
        if (consumers.size() != 1)
            continue;

        const auto transpose = ov::as_type_ptr<opset1::Transpose>(consumers.begin()->get_node()->shared_from_this());
        if (!transpose)
            continue;

        const auto& order = ov::as_type_ptr<opset1::Constant>(transpose->get_input_node_shared_ptr(1));
        if (!order)
            continue;

        const auto order_value = order->cast_vector<int>();
        const auto transpose_child = *(transpose->get_output_target_inputs(0).begin());
        const auto is_brgemm_case = ov::is_type<opset1::MatMul>(transpose_child.get_node()->shared_from_this());
        // If Transpose is supported (can be decomposed or fused into Brgemm), skip
        if ((is_brgemm_case && FuseTransposeBrgemm::supported_cases.count(order_value) != 0) ||
            (TransposeDecomposition::supported_cases.count(order_value) != 0))
            continue;

        // If the transpose isn't supported - we have to extract it from Subgraph
        transpose->set_argument(0, subgraph->input_value(i));
        subgraph->set_argument(i, transpose);
        transpose_child.replace_source_output(parameter);
        // Update shape
        parameter->set_partial_shape(transpose->get_output_partial_shape(0));
        updated = true;
    }

    if (updated) {
        subgraph->validate_and_infer_types();
    }
}

CommonOptimizations::CommonOptimizations(const SnippetsTokenization::Config& config) {
    MATCHER_SCOPE(CommonOptimizations);
    ov::graph_rewrite_callback callback = [&](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::CommonOptimizations");

        auto subgraph = ov::as_type_ptr<ov::snippets::op::Subgraph>(m.get_match_root());
        if (transformation_callback(subgraph)) {
            return false;
        }

        const auto& body = subgraph->body_ptr();
        const auto is_quantized = subgraph->is_quantized();

        // Firstly, we should transform all original Converts inside body to ConvertTruncation to save original behavior.
        // Then if Subgraph contains FakeQuantize we enable specific transformation for quantized subgraphs.
        ov::pass::Manager manager(get_pass_config());
        manager.register_pass<ov::snippets::pass::TransformConvertToConvertTruncation>();
        manager.register_pass<ov::snippets::pass::ExplicitTransposeMatMulInputs>();
        if (is_quantized) {
            manager.register_pass<ov::snippets::pass::CommonFakeQuantizeDecomposition>();
        }
        manager.register_pass<snippets::pass::SoftmaxReshapeElimination>();
        manager.run_passes(body);

        // At the moment only non-scalar Constants of FakeQuantize can be inside Subgraph
        // so we can enable ExtractConstants pass for quantized models
        if (is_quantized) {
            ExtractConstants(subgraph);
        }
        // Extract unsupported Transposes from body
        if (subgraph->has_domain_sensitive_ops()) {
            ExtractUnsupportedTransposes(subgraph);
            if (config.split_m_dimension)
                SplitDimensionM(subgraph, config.concurrency);
        }

        // Validate the body after all common optimizations
        ov::snippets::pass::Validate(get_pass_config()).run_on_model(body);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<ov::snippets::op::Subgraph>(), matcher_name);
    this->register_matcher(m, callback);
}

} // namespace pass
} // namespace snippets
} // namespace ov
