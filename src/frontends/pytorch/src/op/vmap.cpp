// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <map>

#include "batched_tensor.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch {
namespace {
using namespace ov::op;

std::shared_ptr<v0::Constant> axis_constant(int64_t axis) {
    return v0::Constant::create(element::i64, Shape{1}, {axis});
}

struct LoopInput {
    std::shared_ptr<v0::Parameter> parameter;
    Output<Node> value;
    int64_t axis = -1;
};

class VmapBody {
public:
    explicit VmapBody(int64_t level) : m_level(level) {}

    bool depends_on_level(const Output<Node>& value) {
        const auto node = value.get_node();
        const auto found = m_dependencies.find(node);
        if (found != m_dependencies.end()) {
            return found->second;
        }
        const auto batch = ov::as_type_ptr<BatchedTensor>(value.get_node_shared_ptr());
        bool depends = batch && batch->get_level() == m_level;
        for (const auto& input : node->input_values()) {
            depends = depends || depends_on_level(input);
        }
        m_dependencies.emplace(node, depends);
        return depends;
    }

    Output<Node> clone_value(const Output<Node>& value) {
        const auto found = m_values.find(value);
        if (found != m_values.end()) {
            return found->second;
        }
        const auto node = value.get_node_shared_ptr();
        if (!depends_on_level(value)) {
            if (ov::is_type<v0::Constant>(node)) {
                return value;
            }
            auto parameter = std::make_shared<v0::Parameter>(value.get_element_type(), value.get_partial_shape());
            m_inputs.push_back({parameter, value});
            m_values[value] = parameter;
            return parameter;
        }
        const auto batch = ov::as_type_ptr<BatchedTensor>(node);
        if (batch && batch->get_level() == m_level) {
            const auto source = batch->input_value(0);
            FRONT_END_OP_CONVERSION_CHECK(!depends_on_level(source),
                                          "A tensor cannot be mapped twice at one vmap level.");
            auto shape = source.get_partial_shape();
            shape[batch->get_batch_dim()] = 1;
            auto parameter = std::make_shared<v0::Parameter>(source.get_element_type(), shape);
            m_inputs.push_back({parameter, source, batch->get_batch_dim()});
            auto logical = std::make_shared<v0::Squeeze>(parameter, axis_constant(batch->get_batch_dim()));
            m_values[value] = logical;
            return logical;
        }
        OutputVector inputs;
        for (const auto& input : node->input_values()) {
            inputs.push_back(clone_value(input));
        }
        auto cloned = node->clone_with_new_inputs(inputs);
        cloned->set_friendly_name(node->get_friendly_name());
        copy_runtime_info(node, cloned);
        for (size_t i = 0; i < node->get_output_size(); ++i) {
            m_values[node->output(i)] = cloned->output(i);
        }
        return cloned->output(value.get_index());
    }

    const std::vector<LoopInput>& get_inputs() const {
        return m_inputs;
    }

private:
    int64_t m_level;
    std::map<Node*, bool> m_dependencies;
    std::map<Output<Node>, Output<Node>> m_values;
    std::vector<LoopInput> m_inputs;
};

bool is_vmap_scaffolding(const std::string& type) {
    return type == "torch._functorch.predispatch::_vmap_increment_nesting" ||
           type == "torch._functorch.predispatch::_vmap_decrement_nesting" ||
           type == "torch._functorch.predispatch::lazy_load_decompositions";
}
}  // namespace

void check_vmap_node(const NodeContext& context) {
    auto session = context.get_session();
    if (session->m_vmap_depth == 0 || is_vmap_scaffolding(context.get_op_type())) {
        return;
    }
    if (session->m_vmap_error.empty()) {
        const auto& schema = context.get_decoder()->get_schema();
        const auto type = normalize_op_type(context.get_op_type());
        if (schema.find('!') != std::string::npos || context.has_attribute("out") ||
            (type.size() > 1 && type.back() == '_' && type[type.size() - 2] != '_' && type.rfind("aten::", 0) == 0)) {
            session->m_vmap_error = "Mutation inside vmap is not supported.";
        }
        if (type.find("Generator") != std::string::npos || type.find("manual_seed") != std::string::npos) {
            session->m_vmap_error = "Random generators inside vmap are not supported.";
        }
        static const std::vector<std::string> random_prefixes{"aten::rand",
                                                              "aten::normal",
                                                              "aten::bernoulli",
                                                              "aten::multinomial",
                                                              "aten::dropout",
                                                              "aten::native_dropout",
                                                              "aten::_fused_dropout",
                                                              "aten::poisson",
                                                              "aten::rrelu"};
        for (const auto& prefix : random_prefixes) {
            if (type.rfind(prefix, 0) == 0) {
                session->m_vmap_error = "Random operations inside vmap are not supported.";
            }
        }
    }
    FRONT_END_OP_CONVERSION_CHECK(session->m_vmap_error.empty(), session->m_vmap_error);
}

namespace op {
using namespace ov::op;

OutputVector translate_add_batch_dim(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    const auto value = context.get_input(0);
    const auto rank = value.get_partial_shape().rank();
    PYTORCH_OP_CONVERSION_CHECK(rank.is_static(), "vmap requires a statically known input rank.");
    auto dim = context.const_input<int64_t>(1);
    dim = dim < 0 ? dim + rank.get_length() : dim;
    PYTORCH_OP_CONVERSION_CHECK(dim >= 0 && dim < rank.get_length(), "Invalid vmap input axis.");
    const auto level = context.const_input<int64_t>(2);
    return {context.mark_node(std::make_shared<BatchedTensor>(value, dim, level))};
}

OutputVector translate_remove_batch_dim(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    const auto value = context.get_input(0);
    const auto level = context.const_input<int64_t>(1);
    auto batch_size = context.get_input(2);
    if (batch_size.get_element_type() != element::i64) {
        batch_size = std::make_shared<v0::Convert>(batch_size, element::i64);
    }
    batch_size = std::make_shared<v1::Reshape>(batch_size, v0::Constant::create(element::i64, Shape{0}, {}), false);
    const auto rank = value.get_partial_shape().rank();
    PYTORCH_OP_CONVERSION_CHECK(rank.is_static(), "vmap requires a statically known output rank.");
    auto out_dim = context.const_input<int64_t>(3);
    out_dim = out_dim < 0 ? out_dim + rank.get_length() + 1 : out_dim;
    PYTORCH_OP_CONVERSION_CHECK(out_dim >= 0 && out_dim <= rank.get_length(), "Invalid vmap output axis.");
    const auto axis = axis_constant(out_dim);
    VmapBody builder(level);
    if (!builder.depends_on_level(value)) {
        auto expanded = std::make_shared<v0::Unsqueeze>(value, axis);
        auto shape = std::make_shared<v3::ShapeOf>(expanded, element::i64);
        auto batch_vector = std::make_shared<v0::Unsqueeze>(batch_size, axis_constant(0));
        auto target = std::make_shared<v3::ScatterUpdate>(shape,
                                                          axis,
                                                          batch_vector,
                                                          v0::Constant::create(element::i64, Shape{}, {0}));
        return {context.mark_node(std::make_shared<v3::Broadcast>(expanded, target))};
    }

    auto logical = builder.clone_value(value);
    auto stacked = std::make_shared<v0::Unsqueeze>(logical, axis);
    auto condition = v0::Constant::create(element::boolean, Shape{}, {true});
    ParameterVector parameters;
    for (const auto& input : builder.get_inputs()) {
        parameters.push_back(input.parameter);
    }
    auto body = std::make_shared<Model>(OutputVector{condition, stacked}, parameters);
    auto loop = std::make_shared<v5::Loop>(batch_size, condition);
    loop->set_function(body);
    loop->set_special_body_ports({-1, 0});
    for (const auto& input : builder.get_inputs()) {
        if (input.axis < 0) {
            loop->set_invariant_input(input.parameter, input.value);
        } else {
            loop->set_sliced_input(input.parameter, input.value, 0, 1, 1, -1, input.axis);
        }
    }
    auto result = loop->get_concatenated_slices(stacked, 0, 1, 1, -1, out_dim);
    loop->validate_and_infer_types();
    context.mark_node(loop);
    return {result};
}

OutputVector translate_vmap_increment_nesting(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto session = context.get_session();
    ++session->m_vmap_depth;
    if (context.const_input<std::string>(1) != "error") {
        session->m_vmap_error = "vmap supports only randomness='error'.";
    }
    return {v0::Constant::create(element::i64, Shape{}, {session->m_vmap_depth})};
}

OutputVector translate_vmap_decrement_nesting(const NodeContext& context) {
    num_inputs_check(context, 0, 0);
    auto session = context.get_session();
    PYTORCH_OP_CONVERSION_CHECK(session->m_vmap_depth > 0, "Unbalanced vmap nesting.");
    if (--session->m_vmap_depth == 0) {
        session->m_vmap_error.clear();
    }
    return {v0::Constant::create(element::i64, Shape{}, {session->m_vmap_depth})};
}

OutputVector translate_vmap_lazy_load_decompositions(const NodeContext& context) {
    num_inputs_check(context, 0, 0);
    return {v0::Constant::create(element::i64, Shape{}, {0})};
}

}  // namespace op
}  // namespace ov::frontend::pytorch
