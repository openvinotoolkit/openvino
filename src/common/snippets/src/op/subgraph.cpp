// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "remarks.hpp"

#include "snippets/op/subgraph.hpp"
#include "snippets/pass/insert_load_store.hpp"
#include "snippets/pass/insert_movebroadcast.hpp"
#include "snippets/pass/load_movebroadcast_to_broadcastload.hpp"
#include "snippets/pass/assign_registers.hpp"

#include <ngraph/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>

#include <algorithm>
#include <memory>
#include <array>

using namespace std;
using namespace ngraph;

void snippets::op::Subgraph::set_generator(std::shared_ptr<ngraph::snippets::Generator> generator) {
    m_generator = generator;
}

snippets::op::Subgraph::Subgraph(const OutputVector& args, std::shared_ptr<ov::Model> body)
    : Op(args), m_body(body), m_generator(nullptr) {
    constructor_validate_and_infer_types();
}

snippets::op::Subgraph::Subgraph(const NodeVector& args, std::shared_ptr<ov::Model> body)
    : Subgraph(as_output_vector(args), body) {}

std::shared_ptr<Node> snippets::op::Subgraph::clone_with_new_inputs(const OutputVector& inputs) const {
    INTERNAL_OP_SCOPE(Subgraph);
    return make_shared<Subgraph>(inputs, ov::clone_model(*m_body.get()));
}

void snippets::op::Subgraph::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::validate_and_infer_types")
    ngraph::ParameterVector old_parameters;
    for (auto op : m_body->get_parameters()) {
        old_parameters.push_back(op);
    }

    for (size_t i = 0; i < get_input_size(); ++i) {
        m_body->replace_parameter(i, std::make_shared<opset1::Parameter>(get_input_element_type(i), get_input_partial_shape(i)));
    }

    m_body->validate_nodes_and_infer_types();

    for (size_t i = 0; i < m_body->get_parameters().size(); i++) {
        m_body->get_parameters()[i]->set_friendly_name(old_parameters[i]->get_friendly_name());
    }

    set_output_size(m_body->get_output_size());
    for (size_t i = 0; i < get_output_size(); ++i) {
        set_output_type(i, m_body->get_output_element_type(i), m_body->get_output_partial_shape(i));
    }
}

bool snippets::op::Subgraph::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

auto snippets::op::Subgraph::wrap_node_as_subgraph(const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<op::Subgraph> {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::wrap_node_as_subgraph")
    ngraph::ParameterVector body_parameters;
    ngraph::OutputVector body_inputs;

    ngraph::OutputVector subgraph_inputs;

    for (const auto& input : node->input_values()) {
        if (is_scalar_constant(input.get_node_shared_ptr())) {
            body_inputs.push_back(input);
        } else {
            auto parameter = std::make_shared<ngraph::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
            body_parameters.push_back(parameter);
            body_parameters.back()->set_friendly_name(input.get_node()->get_friendly_name());
            body_inputs.push_back(parameter->output(0));

            subgraph_inputs.push_back(input);
        }
    }

    auto body_node = node->clone_with_new_inputs(body_inputs);
    body_node->set_friendly_name(node->get_friendly_name());

    if (node->get_output_size() != body_node->get_output_size()) {
        throw ngraph::ngraph_error("original node outputs size and extracted subgraph node outputs size doesn't much");
    }

    // Clear the node dependencies so graph::topological_sort will not find any extra ops in get_ordered_ops()
    //  This is needed so the model body will be created correctly
    body_node->clear_control_dependencies();
    ngraph::ResultVector body_results;
    for (auto output : node->outputs()) {
        body_results.push_back(std::make_shared<ngraph::opset1::Result>(body_node->output(output.get_index())));
    }

    auto body = create_body(node->get_friendly_name(), body_results, body_parameters);
    auto subgraph = build_subgraph(node, subgraph_inputs, body);

    for (size_t i = 0; i < body->get_parameters().size(); i++) {
        body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
    }

    if (subgraph->get_output_size() != body->get_results().size()) {
        throw ngraph::ngraph_error("newly create subgraph doesn't much number of original node results");
    }

    return subgraph;
}

std::shared_ptr<snippets::op::Subgraph> snippets::op::Subgraph::make_canonical_from_this() {
    INTERNAL_OP_SCOPE(Subgraph);
    ngraph::OutputVector subgraph_node_inputs;
    for (auto input : this->input_values()) {
        subgraph_node_inputs.push_back(input);
    }
    auto new_body = ov::clone_model(*this->get_body().get());
    auto snippet = std::make_shared<op::Subgraph>(subgraph_node_inputs, new_body);
    ngraph::copy_runtime_info(this->shared_from_this(), snippet);
    snippet->set_friendly_name(this->get_friendly_name());
    snippet->set_generator(this->m_generator);

    return snippet;
}

// We also can think of canonization as of pass to copy original subgraph and transforming it to canonical form suitable for code generation
// pass actual parameters and results shapes to generate for as well as channel mapping,
// Todo: we need to distinguish between 5d tensors that represents <N, C, H, W, c> and <N, C, D, H, W> somehow like locked dimensions
//  ngraph::AxisVector to code
void snippets::op::Subgraph::canonicalize(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes) {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::canonicalize")
    NODE_VALIDATION_CHECK(this, input_shapes.size() == m_body->get_parameters().size(),
        "Number of parameters for snippet doesn't match passed to generate method: ", input_shapes.size(), " vs ", m_body->get_parameters().size(), ".");

    NODE_VALIDATION_CHECK(this, output_shapes.size() == m_body->get_results().size(),
        "number of results for snippet doesn't match passed to generate method: ", output_shapes.size(), " vs ", m_body->get_results().size(), ".");

    // replace only constants which are actually should be represented as scalars during code generation and probably move this step a bit later
    for (auto op : m_body->get_ordered_ops()) {
        if (auto constant = ngraph::as_type_ptr<opset1::Constant>(op)) {
            auto scalar = std::make_shared<snippets::op::Scalar>(*constant);
            scalar->set_friendly_name(constant->get_friendly_name());
            ngraph::copy_runtime_info(constant, scalar);
            ngraph::replace_node(constant, scalar);
        }
    }



    // it should be in subgraph node to be aligned with internal and external parameter list, but adding this for testing
    // TODO: store blocking into to Parameter's rt_info for future propagation
    for (size_t i = 0; i < m_body->get_parameters().size(); i++) {
        auto param = m_body->get_parameters()[i];
        if (param->get_shape().size() < 4) {
            std::vector<size_t> shape(4, 1);
            std::copy(param->get_shape().begin(), param->get_shape().end(), &shape.at(4 - (param->get_shape().size() == 0 ? 1 : param->get_shape().size())) );
            m_body->replace_parameter(i, std::make_shared<opset1::Parameter>(param->get_element_type(), ngraph::Shape(shape)));
        } else if (param->get_shape().size() >= 4) {
            if (param->get_element_type() != std::get<2>(input_shapes[i])) {
                throw ngraph::ngraph_error("changes in presision. Is it legal??");
            }
            m_body->replace_parameter(i, std::make_shared<opset1::Parameter>(std::get<2>(input_shapes[i]), std::get<0>(input_shapes[i])));
        }
    }

    m_body->validate_nodes_and_infer_types();

    for (size_t i = 0; i < m_body->get_results().size(); i++) {
        auto result = m_body->get_results()[i];
        PartialShape partial(result->get_shape());
        bool isCompatible = ngraph::PartialShape::broadcast_merge_into(partial, std::get<0>(output_shapes[i]), ::ngraph::op::AutoBroadcastType::NUMPY);
        // equality check won't pass since we reshape without changes on external snippet edges
        NODE_VALIDATION_CHECK(this, isCompatible, "Inferend and passed results shapes are difference for snippet : ",
                                                  result->get_shape(), " vs ", std::get<0>(output_shapes[i]), ".");
    }
}

void snippets::op::Subgraph::convert_to_snippet_dialect() {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::convert_to_snippet_dialect")
    ngraph::pass::Manager manager;
    manager.register_pass<snippets::pass::InsertLoad>();
    manager.register_pass<snippets::pass::InsertStore>();
    manager.register_pass<snippets::pass::InsertMoveBroadcast>();
    manager.register_pass<snippets::pass::LoadMoveBroadcastToBroadcastLoad>();
    manager.run_passes(m_body);
}

snippets::Schedule snippets::op::Subgraph::generate(const BlockedShapeVector& output_shapes,
                                                    const BlockedShapeVector& input_shapes,
                                                    const void* compile_params) {
    return generate(output_shapes, input_shapes, ngraph::pass::Manager(), compile_params);
}

snippets::Schedule snippets::op::Subgraph::generate(const BlockedShapeVector& output_shapes,
                                                    const BlockedShapeVector& input_shapes,
                                                    ngraph::pass::Manager opt,
                                                    const void* compile_params) {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::generate")
    NGRAPH_CHECK(m_generator != nullptr, "generate is called while generator is not set");

    canonicalize(output_shapes, input_shapes);

    // Todo: ngraph::pass::Manager introduces appreciable overheads, especially while used on small graphs.
    // So don't wrap this transformation as a MatcherPass, but rewrite convert_to_snippet_dialect() as a
    // for loop to improve first-inference time.
    // replace power with power static

    for (auto op : m_body->get_ordered_ops()) {
        if (ov::is_type<opset1::Power>(op) &&
            ov::is_type<snippets::op::Scalar>(op->get_input_node_shared_ptr(1)) &&
            ov::shape_size(op->get_input_shape(1)) == 1) {
            auto power = ov::as_type_ptr<opset1::Power>(op);
            auto scalar = ov::as_type_ptr<snippets::op::Scalar>(op->get_input_node_shared_ptr(1));
            auto value = scalar->cast_vector<float>()[0];;
            auto power_static = std::make_shared<snippets::op::PowerStatic>(power->input(0).get_source_output(), value);
            power_static->set_friendly_name(power->get_friendly_name());
            ngraph::copy_runtime_info(power, power_static);
            ngraph::replace_node(power, power_static);
        }
    }


    convert_to_snippet_dialect();
    opt.run_passes(m_body);

    // generation flow
    snippets::pass::AssignRegisters().run_on_model(m_body);

    // schedule generation should go here and be target agnostic

    // actual code emission
    ngraph::snippets::code ptr = m_generator->generate(m_body, compile_params);

    // check that body doesn't have constants for scheduling
    std::vector<std::shared_ptr<opset1::Constant>> constants;
    for (auto op : m_body->get_ordered_ops()) {
        if (auto constant = ov::as_type_ptr<opset1::Constant>(op)) {
            if (ngraph::shape_size(constant->get_shape()) != 1 && constant->get_shape() != Shape()) {
                constants.push_back(constant);
            }
        }
    }
    NGRAPH_CHECK(!constants.size(), "External constants detected. Snippet is illigal for scheduling");

    // check resulting shapes are broadcastable to each other so can be scheduled
    Shape work_size = m_body->output(0).get_shape();
    for (size_t k = 0; k < m_body->get_output_size(); k++) {
        auto shape = m_body->output(k).get_shape();

        if (work_size.size() != shape.size()) {
            throw ngraph_error("rank for all outputs of a snippet should match");
        }

        for (size_t i = 0; i < work_size.size(); i++) {
            if (work_size[i] != shape[i]) {
                if (work_size[i] == 1 || shape[i] == 1) {
                    work_size[i] = max(work_size[i], shape[i]);
                } else {
                    throw ngraph_error("incompatible shapes for output graphs");
                }
            }
        }
    }

    return {work_size, false /*canBeLinearized*/, ptr};
}

void snippets::op::Subgraph::print() const {
    INTERNAL_OP_SCOPE(Subgraph);
    remark(13) << "subgraph " << this->get_friendly_name() << " "
        << this->get_type_name()
        << " which contains " << this->get_body()->get_ops().size() << " nodes" << std::endl;

    int qqq = 0;
    for (auto op : this->get_body()->get_ordered_ops()) {
        remark(13) << "op " << qqq++ << " " << op->get_friendly_name() << " (" << op->get_type_name() << ") " << op << std::endl;
    }

    for (auto& in : this->inputs()) {
        remark(13) << "  -> " << in.get_source_output().get_node_shared_ptr()->get_friendly_name() << " "
            << in.get_source_output().get_node_shared_ptr() << std::endl;
    }

    for (auto& out : this->outputs()) {
        for (auto& user : out.get_target_inputs()) {
            remark(13) << " <- " << user.get_node()->get_friendly_name() << " "  << user.get_node() << std::endl;
        }
        remark(13) << std::endl;
    }
}

void snippets::op::Subgraph::print_statistics(bool verbose) {
    INTERNAL_OP_SCOPE(Subgraph);
    auto getNodeInventory = [](std::shared_ptr<ov::Node> n) -> size_t {
        size_t total = 0;

        for (auto input : n->inputs()) {
            total += input.get_tensor().size();
        }

        for (auto output : n->outputs()) {
            total += output.get_tensor().size();
        }

        if (auto subgraph = ngraph::as_type_ptr<op::Subgraph>(n)) {
            for (auto op : subgraph->get_body()->get_ordered_ops()) {
                if (ngraph::as_type_ptr<ngraph::opset1::Constant>(op)) {
                    total += op->output(0).get_tensor().size();
                }
            }
        }

        return total;
    };

    auto getModelInventory = [getNodeInventory](std::shared_ptr<ov::Model> f) -> size_t {
        size_t total = 0;
        for (auto op : f->get_ordered_ops()) {
            // Results and parameters are artificially introduced,
            // while Constants are already considered if they are inputs of other operation
            // this should lead to 1:1 inventory for single node operations
            if (!ngraph::as_type_ptr<ngraph::opset1::Parameter>(op)
             && !ngraph::as_type_ptr<ngraph::opset1::Result>(op)
             && !ngraph::as_type_ptr<ngraph::opset1::Constant>(op)) {
                total += getNodeInventory(op);
            }
        }
        return total;
    };

    auto countConstants = [](std::shared_ptr<ov::Model> f) -> size_t {
        size_t count = 0;
        for (auto op : f->get_ordered_ops()) {
            count += !!ngraph::as_type_ptr<ngraph::opset1::Constant>(op) ? 1 : 0;
        }
        return count;
    };

    auto body = this->get_body();

    std::cout << this->get_friendly_name()
                << ";" << this
                << ";" << body->get_ops().size()
                << ";" << body->get_parameters().size()
                << ";" << body->get_results().size()
                << ";" << countConstants(body)
                << ";" << getModelInventory(body)
                << ";" << getNodeInventory(this->shared_from_this()) << std::endl;

    if (verbose) {
        this->print();
    }
}

void snippets::op::Subgraph::serialize() const {
    std::stringstream xmlFile, binFile;
    ov::pass::Serialize serializer(xmlFile, xmlFile, ov::pass::Serialize::Version::IR_V10);
    serializer.run_on_model(get_body());
    auto m_constants = binFile.str();
    auto m_model = xmlFile.str();
    std::cout << m_model << std::endl;
}
