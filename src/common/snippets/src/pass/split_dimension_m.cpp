// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/split_dimension_m.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"

namespace {
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

const size_t SplitDimensionM::min_kernel_m = 32;
const size_t SplitDimensionM::dim_M_index = 1;

bool SplitDimensionM::is_supported_matmul(const std::shared_ptr<const ov::Node>& node) {
    const auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
    return matmul && !matmul->get_transpose_a() && !matmul->is_dynamic();
}

std::pair<size_t, size_t> SplitDimensionM::split_ideally(size_t batch_dim, size_t m_dim, size_t optimal_parallelism_work_amount) {
    // Ideal case #1: M can be split on the parts one of which complements the batch dimension to the optimal parallel work amount
    // In this case, each thread will execute the Snippets kernel once
    const size_t lower_bound = optimal_parallelism_work_amount / batch_dim;
    if (lower_bound * batch_dim == optimal_parallelism_work_amount && m_dim % lower_bound == 0)
        return std::make_pair(lower_bound, m_dim / lower_bound);

    // Ideal case #2: M is divisible by optimal parallel work amount, and the new_m_dim is big enough
    // In this case, each thread will execute the Snippets kernel 'batch_dim' times
    if (m_dim % optimal_parallelism_work_amount == 0) {
        const auto new_m_dim = m_dim / optimal_parallelism_work_amount;
        if (new_m_dim >= min_kernel_m)
            return std::make_pair(optimal_parallelism_work_amount, new_m_dim);
    }

    return std::make_pair(1, m_dim);
}

std::pair<size_t, size_t> SplitDimensionM::split_fallback_increase_parallel_wa(size_t batch_dim, size_t m_dim, size_t optimal_parallelism_work_amount) {
    std::pair<size_t, size_t> splited = { 1, m_dim };
    const size_t upper_bound = utils::div_up(2 * optimal_parallelism_work_amount, batch_dim);
    for (size_t divisor_0 = upper_bound - 1; divisor_0 > 1; divisor_0--) {
        size_t divisor_1 = m_dim / divisor_0;
        if (divisor_1 * divisor_0 == m_dim)
            return divisor_0 * batch_dim >= optimal_parallelism_work_amount ? std::make_pair(divisor_0, divisor_1) : splited;
    }
    return splited;
}

std::pair<size_t, size_t> SplitDimensionM::split_minimize_kernel_wa(size_t batch_dim, size_t m_dim, size_t optimal_parallelism_work_amount) {
    // This heuristic minimizes 'm_kernel' (=> maximizes 'm_batch') with a limitation that 'm_kernel >= min_kernel_m'.
    // In other words, it tries to find 'm_kernel' bigger than 'min_kernel_m' and at the same time as close as possible to this value.
    std::pair<size_t, size_t> best_result = {1, m_dim};
    for (size_t divisor = 2; divisor < std::sqrt(m_dim); ++divisor) {
        if (m_dim % divisor != 0)
            continue;
        // If divisor is more than 'min_kernel_m', divisor becomes 'm_kernel',
        // guaranteeing the most optimal implementation from 'm_kernel' minimization perspective.
        if (divisor >= min_kernel_m)
            return std::make_pair(m_dim / divisor, divisor);

        // If divisor is less than 'min_kernel_m', divisor becomes m_batch.
        // However, it is not guaranteed that the current 'm_kernel = m_dim / divisor' is minimized, as one of the next divisors can be more optimal.
        // So in this case the best result is remembered
        const size_t m_kernel = m_dim / divisor;
        if (m_kernel >= min_kernel_m) {
            best_result.first = divisor;
            best_result.second = m_kernel;
        }
    }
    if (best_result.first * batch_dim >= optimal_parallelism_work_amount)
        return best_result;
    return std::make_pair(1, m_dim);
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
    OPENVINO_ASSERT(m_index < shape.size(), "Incorrect M index: it should be less than target shape rank");
    if (shape[m_index] == 1)
        return unsqueeze_m_dim(std::move(shape), m_index);
    shape[m_index] = new_m_dim;
    shape.insert(shape.begin() + m_index, batch_m_dim);
    return shape;
}
ov::snippets::VectorDims SplitDimensionM::unsqueeze_m_dim(ov::snippets::VectorDims shape, size_t m_index) {
    OPENVINO_ASSERT(m_index < shape.size(), "Incorrect M index: it should be less than target shape rank");
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

    // We skip optimization if the current batch is optimal for concurrency
    if (batch_dim % optimal_parallelism_work_amount == 0)
        return false;

    auto split_is_done = [&batch_m_dim]() {
        return batch_m_dim != 1;
    };

    std::tie(batch_m_dim, new_m_dim) = split_ideally(batch_dim, m_dim, optimal_parallelism_work_amount);
    if (split_is_done())
        return true;

    std::tie(batch_m_dim, new_m_dim) = split_minimize_kernel_wa(batch_dim, m_dim, optimal_parallelism_work_amount);
    if (split_is_done())
        return true;
    // If all the previous heuristics failed, fallback heuristic is used, which reflects the old splitting behavior
    if (batch_dim < optimal_parallelism_work_amount)
        std::tie(batch_m_dim, new_m_dim) = split_fallback_increase_parallel_wa(batch_dim, m_dim, optimal_parallelism_work_amount);
    return split_is_done();
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
        OPENVINO_ASSERT(m_index < shape.size(), "Dimension index must be less than shape rank");
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
        const auto forward_index = order.size() - 1 - dim_M_index;
        const auto m_index = is_input ? order[forward_index] : forward_index;  // Index of M dimension in the previous order
        const auto new_order = get_updated_order(order, m_index);
        transpose->set_argument(1, std::make_shared<ov::op::v0::Constant>(order_constant->get_element_type(), ov::Shape{new_order.size()}, new_order));
        return m_index;
    };

    auto reshape_parameter = [&](const std::shared_ptr<ov::Node>& node, bool split_m_dim = true) {
        const auto param = ov::as_type_ptr<ov::op::v0::Parameter>(node);
        if (!param || reshaped_params.count(param) > 0)
            return;

        const auto shape = param->get_partial_shape().get_shape();
        // if the index of dimension M is equal or greater than Shape rank, no need to reshape it.
        if (shape.size() <= dim_M_index)
            return;

        const auto consumers = param->get_output_target_inputs(0);
        const auto shared_consumer = consumers.begin()->get_node()->shared_from_this();
        auto m_index = shape.size() - 1 - dim_M_index;
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

    // The pass supports only static shapes on Subgraph inputs due to static `Reshape` insertion around Subgraph.
    const auto& params = subgraph->body_ptr()->get_parameters();
    const auto is_dynamic = [](const std::shared_ptr<ov::Node>& p) { return p->get_output_partial_shape(0).is_dynamic(); };
    if (std::any_of(params.cbegin(), params.cend(), is_dynamic))
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
