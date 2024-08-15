// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/split_dimension_m.hpp"

#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"

namespace {
size_t get_dim_M(const ov::Shape& shape) {
    return *(shape.rbegin() + 1);
}
bool is_prime_number(size_t value) {
    if (ov::snippets::utils::one_of(value, 2lu, 3lu)) return true;
    if (value == 1 || value % 2 == 0 || value % 3 == 0) return false;
    const auto root = std::sqrt(value) + 1;
    for (size_t divisor = 5; divisor < root; divisor += 6) {
        if ((value % divisor == 0) || (value % (divisor + 2) == 0))
            return false;
    }
    return true;
}
}  // namespace

namespace ov {
namespace snippets {
namespace pass {
bool SplitDimensionM::is_supported_matmul(const std::shared_ptr<const ov::Node>& node) {
    const auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
    return matmul && !matmul->get_transpose_a() && !matmul->is_dynamic();
}

std::pair<size_t, size_t> SplitDimensionM::get_splited_dimensions(size_t batch_dim, size_t m_dim, size_t optimal_parallelism_work_amount) {
    std::pair<size_t, size_t> splited = { 1, m_dim };

    const size_t lower_bound = optimal_parallelism_work_amount / batch_dim;
    if (lower_bound * batch_dim == optimal_parallelism_work_amount && m_dim % lower_bound == 0) {
        splited.first = lower_bound;
        splited.second = m_dim / lower_bound;
        OPENVINO_ASSERT(splited.first * splited.second == m_dim, "Incorrect dimension M splitting!");
        return splited;
    }

    const size_t upper_bound = utils::div_up(2 * optimal_parallelism_work_amount, batch_dim);
    for (size_t divisor_0 = upper_bound - 1; divisor_0 > 1; divisor_0--) {
        size_t divisor_1 = m_dim / divisor_0;
        if (divisor_1 * divisor_0 == m_dim) {
            splited.first = divisor_0;
            splited.second = divisor_1;
            break;
        }
    }
    OPENVINO_ASSERT(splited.first * splited.second == m_dim, "Incorrect dimension M splitting!");
    return splited;
}

bool SplitDimensionM::can_be_optimized(const std::shared_ptr<const ov::Node>& node, size_t concurrency) {
    if (!is_supported_matmul(node))
        return false;
    size_t batch_m_dim, new_m_dim;
    return split(node->get_shape(), concurrency, batch_m_dim, new_m_dim);
}

std::vector<size_t> SplitDimensionM::get_updated_order(const std::vector<size_t>& order, size_t m_index) {
    std::vector<size_t> new_order(order.size() + 1, 0);
    size_t shift_idx = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] < m_index) {
            new_order[i + shift_idx] = order[i];
        } else if (order[i] == m_index) {
            new_order[i + shift_idx++] = order[i];
            new_order[i + shift_idx] = order[i] + 1;
        } else {
            new_order[i + shift_idx] = order[i] + 1;
        }
    }
    return new_order;
}

ov::snippets::VectorDims SplitDimensionM::reshape_m_dim(ov::snippets::VectorDims shape, size_t m_index, size_t batch_m_dim, size_t new_m_dim) {
    if (shape[m_index] == 1)
        return unsqueeze_m_dim(std::move(shape), m_index);
    shape[m_index] = new_m_dim;
    shape.insert(shape.begin() + m_index, batch_m_dim);
    return shape;
}
ov::snippets::VectorDims SplitDimensionM::unsqueeze_m_dim(ov::snippets::VectorDims shape, size_t m_index) {
    shape.insert(shape.begin() + m_index, 1);
    return shape;
}

std::shared_ptr<ov::op::v0::MatMul> SplitDimensionM::get_matmul(const std::shared_ptr<op::Subgraph>& subgraph) {
    const auto& body = subgraph->body_ptr();
    const auto& parameters = body->get_parameters();
    // [107806]: If count of Parameters isn't equal to Subgraph inputs (it's possible case in general),
    //           we cannot garantee correct extraction since we don't have correct connections between body I/O and Subgraph I/O.
    OPENVINO_ASSERT(parameters.size() == subgraph->input_values().size(),
                    "Failed to extract unsupported transposes: the count of Parameters isn't equal to Subgraph inputs");

    // Need to find MatMul0 and check output shape
    const auto& ops = body->get_ordered_ops();
    const auto mm_it = std::find_if(ops.cbegin(), ops.cend(),
                                    [](const std::shared_ptr<ov::Node>& node){ return ov::is_type<ov::op::v0::MatMul>(node); });
    if (mm_it == ops.end())
        return nullptr;

    const auto matmul0 = *mm_it;
    return is_supported_matmul(matmul0) ? ov::as_type_ptr<ov::op::v0::MatMul>(matmul0) : nullptr;
}

bool SplitDimensionM::split(const ov::Shape& shape, size_t optimal_parallelism_work_amount, size_t& batch_m_dim, size_t& new_m_dim) {
    const auto batch_dim =
        std::accumulate(shape.rbegin() + 2, shape.rend(), size_t(1), std::multiplies<size_t>());  // B (batch)
    const auto m_dim = get_dim_M(shape);  // M
    if (is_prime_number(m_dim))
        return false;

    auto is_optimized = [&](size_t batch_dim) {
        return batch_dim >= optimal_parallelism_work_amount;
    };

    // We skip optimization if the current batch is optimal for concurrency
    if (is_optimized(batch_dim))
        return false;

    std::tie(batch_m_dim, new_m_dim) = get_splited_dimensions(batch_dim, m_dim, optimal_parallelism_work_amount);
    return is_optimized(batch_dim * batch_m_dim);
}

void SplitDimensionM::reshape_subgraph(const std::shared_ptr<op::Subgraph>& subgraph, const ov::Shape& shape, size_t batch_m_dim, size_t new_m_dim) {
    const auto& body = subgraph->body_ptr();
    const auto& parameters = body->get_parameters();
    const auto& results = body->get_results();
    const auto ops = body->get_ordered_ops();
    const auto m_dim = get_dim_M(shape);

    // There are two Parameter variants:
    //  - Parameter on branches for Second input of MatMul - the shape should be only unsqueezed (add just 1)
    //  - Other Parameters (on First input of MatMuls and between) - the shape should be splitted on M dimension

    std::set<std::shared_ptr<ov::op::v0::Parameter>> reshaped_params;

    auto insert_reshape = [&](const std::shared_ptr<ov::op::v0::Parameter>& param, const ov::Shape& new_shape) {
        const auto index = std::distance(parameters.begin(), std::find(parameters.begin(), parameters.end(), param));
        const auto shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{new_shape.size()}, new_shape);
        const auto reshape = std::make_shared<ov::op::v1::Reshape>(subgraph->input_value(index), shape_const, false);
        subgraph->input(index).replace_source_output(reshape);
        param->set_partial_shape(new_shape);
        reshaped_params.insert(param);
    };

    auto get_updated_shape = [&](const ov::snippets::VectorDims& shape, size_t m_index, bool split_m_dim) {
        const auto current_m_dim = shape[m_index];
        OPENVINO_ASSERT(!split_m_dim || current_m_dim == 1 || current_m_dim == m_dim, "Incorrect shape for splitting!");
        const auto new_shape = split_m_dim ? reshape_m_dim(shape, m_index, batch_m_dim, new_m_dim) : unsqueeze_m_dim(shape, m_index);
        OPENVINO_ASSERT(ov::shape_size(new_shape) == ov::shape_size(shape), "Incorrect shape splitting!");
        return new_shape;
    };

    auto reshape_transpose = [&](const std::shared_ptr<ov::Node>& transpose, bool is_input) -> size_t {
        const auto order_constant = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(order_constant != nullptr, "Transpose must have Constant order");
        const auto order = order_constant->cast_vector<size_t>();
        const auto m_index = is_input ? order[order.size() - 2] : order.size() - 2;  // Index of M dimension in the previous order
        const auto new_order = get_updated_order(order, m_index);
        transpose->set_argument(1, std::make_shared<ov::op::v0::Constant>(order_constant->get_element_type(), ov::Shape{new_order.size()}, new_order));
        return m_index;
    };

    auto reshape_parameter = [&](const std::shared_ptr<ov::Node>& node, bool split_m_dim = true) {
        const auto param = ov::as_type_ptr<ov::op::v0::Parameter>(node);
        if (!param || reshaped_params.count(param) > 0)
            return;

        const auto shape = param->get_partial_shape().get_shape();
        const auto consumers = param->get_output_target_inputs(0);
        const auto shared_consumer = consumers.begin()->get_node()->shared_from_this();
        auto m_index = shape.size() - 2;
        if (ov::is_type<ov::op::v1::Transpose>(shared_consumer)) {
            m_index = reshape_transpose(shared_consumer, true);
        }
        insert_reshape(param, get_updated_shape(shape, m_index, split_m_dim));
    };

    auto update_matmul_second_branch = [&](const std::shared_ptr<ov::op::v0::MatMul>& node) {
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
        if (const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(op)) {
            update_matmul_second_branch(matmul);
        } else if (const auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(op)) {
            softmax_v8->set_axis(-1);
        } else if (const auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(op)) {
            softmax_v1->set_axis(softmax_v1->get_output_partial_shape(0).size()); // since new_shape.size() = old_shape.size() + 1
        } else if (const auto broadcast = ov::as_type_ptr<ov::op::v1::Broadcast>(op)) {
            // Broadcast is tokenized only between MatMuls -> Split M dimension
            const auto shape_const = ov::as_type_ptr<ov::op::v0::Constant>(broadcast->get_input_node_shared_ptr(1));
            OPENVINO_ASSERT(shape_const, "SplitDimensionM expects Broadcast with Constant output shape");
            const auto m_dim_idx = broadcast->get_output_partial_shape(0).size() - 2;
            const auto new_shape = get_updated_shape(shape_const->cast_vector<size_t>(), m_dim_idx, true);
            broadcast->set_argument(1, std::make_shared<ov::op::v0::Constant>(shape_const->get_element_type(), ov::Shape{new_shape.size()}, new_shape));
        }
    }

    // Secondly, Update All M dimensions for remaining parameters
    for (const auto& param : parameters) {
        if (reshaped_params.count(param) == 0)
            reshape_parameter(param, true);
    }

    // Update Transpose order on Result
    for (const auto& res : results) {
        const auto parent = res->get_input_node_shared_ptr(0);
        if (ov::is_type<ov::op::v1::Transpose>(parent)) {
            reshape_transpose(parent, false);
        }
    }

    // Return the previous shape on outputs
    for (size_t i = 0; i < subgraph->get_output_size(); ++i) {
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
    }
    subgraph->set_friendly_name(subgraph->get_friendly_name() + "_original");
    // Need to update inner Shapes and Softmax Axis
    subgraph->validate_and_infer_types();
}

bool SplitDimensionM::run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SplitDimensionM");
    // To increase parallelism work in MHA pattern,
    // we split 1st dimension (starting from 0th) into 2 new dimensions to get 4D Shapes where
    // - 0th and 1st dimensions are used in parallel scheduling,
    // - 2nd and 3rd dimensions are used in kernel

    // It's needed only for MHA patterns. Need to add support for common patterns
    if (!subgraph->has_domain_sensitive_ops())
        return false;

    if (const auto matmul0 = get_matmul(subgraph)) {
        const auto mm_shape = matmul0->get_shape();
        size_t batch_m_dim, new_m_dim;
        if (!split(mm_shape, m_concurrency, batch_m_dim, new_m_dim))
            return false;

        reshape_subgraph(subgraph, mm_shape, batch_m_dim, new_m_dim);
        return true;
    }
    return false;
}
} // namespace pass
} // namespace snippets
} // namespace ov
