// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/op/subgraph.hpp"
#include "snippets/pass/insert_load_store.hpp"
#include "snippets/pass/insert_movebroadcast.hpp"
#include "snippets/pass/load_movebroadcast_to_broadcastload.hpp"
#include "snippets/pass/assign_registers.hpp"
#include "snippets/pass/convert_constants_to_scalars.hpp"
#include "snippets/pass/convert_power_to_powerstatic.hpp"
#include "snippets/pass/vector_to_scalar.hpp"

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
///
/// \brief  Canonization transforms original subgraph and to canonical form suitable for code generation. In particular,
///         it handles supported layout conversions, broadcasts inputs and outputs to a single rank and layout. Canonicalization
///         returns master-shape (max rank + max dimensions over all outputs) that can be used for scheduling.
///         Canonicalization currently supports only the following layout conversions:
///             * None: all inputs have the same layout
///             * Planar + blocked: some inputs have blocked, and some have planar layouts, e.g. <N, C, H, W, c> + <N, C, H, W>
Shape snippets::op::Subgraph::canonicalize(const BlockedShapeVector& outputShapes, const BlockedShapeVector& inputShapes) {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::canonicalize")
    NODE_VALIDATION_CHECK(this, inputShapes.size() == m_body->get_parameters().size(),
        "Number of parameters for snippet doesn't match passed to generate method: ", inputShapes.size(), " vs ", m_body->get_parameters().size(), ".");

    NODE_VALIDATION_CHECK(this, outputShapes.size() == m_body->get_results().size(),
        "number of results for snippet doesn't match passed to generate method: ", outputShapes.size(), " vs ", m_body->get_results().size(), ".");

    auto getMaxRankBlockedShape = [](const BlockedShapeVector& blockedShapes) -> const BlockedShape& {
        return *std::max_element(blockedShapes.begin(), blockedShapes.end(),
                         [&](const BlockedShape& lhs, const BlockedShape& rhs) {
                            return std::get<0>(lhs).size() < std::get<0>(rhs).size();
                         });
    };
    Shape baseShape;
    AxisVector baseOrder;
    std::tie(baseShape, baseOrder, std::ignore) = getMaxRankBlockedShape(inputShapes);
    const auto baseRank = baseShape.size();
    const bool baseIsBlocked = baseOrder.size() != std::set<size_t>(baseOrder.begin(), baseOrder.end()).size();
    for (size_t i = 0; i < inputShapes.size(); i++) {
        const auto &blockedShape = inputShapes[i];
        Shape inShape;
        AxisVector inOrder;
        element::Type inType;
        std::tie(inShape, inOrder, inType) = blockedShape;
        const auto inRank = inShape.size();
        NODE_VALIDATION_CHECK(this, inRank <= baseRank, "Input rank can't be larger than output rank in snippets.");
        if (inRank < baseRank) {
            Shape newShape(baseRank, 1);
            // todo: more complicated logics is needed if we want to merge smth else than blocked and planar
            // could be done by PartialShape::broadcast_merge_into, but this way is faster
            size_t startOffset = baseRank - inRank;
            if (baseIsBlocked) {
                const bool inIsNotBlocked = inOrder.size() == std::set<size_t>(inOrder.begin(), inOrder.end()).size();
                NODE_VALIDATION_CHECK(this, inIsNotBlocked, "Snippets don't support conversion between blocked layouts of different ranks");
                startOffset--;
            }
            std::copy(inShape.begin(), inShape.end(), &newShape[startOffset]);
            inShape = move(newShape);
        } else {
            // todo: 4d blocked + 5d planar layouts are not supported: <N, C, H, W, c> + <N, C, D, H, W>
            NODE_VALIDATION_CHECK(this,
                                  equal(baseOrder.begin(), baseOrder.end(), inOrder.begin()),
                                  "Snippets canonicalization got input shapes of equal ranks but different layouts, which is not supported");
        }
        ov::PartialShape tmpPShape(baseShape);
        NODE_VALIDATION_CHECK(this,
                              PartialShape::broadcast_merge_into(tmpPShape, inShape, ::ngraph::op::AutoBroadcastType::NUMPY),
                              "Failed to create broadcastable shapes in snippets canonicalization");
        const auto paramShape = m_body->get_parameters()[i]->get_shape();
        if (paramShape.size() != inShape.size() || !equal(paramShape.begin(), paramShape.end(), inShape.begin()))
                m_body->replace_parameter(i, std::make_shared<opset1::Parameter>(inType, inShape));
    }

    m_body->validate_nodes_and_infer_types();
    auto skipStartEndOnes = [](const Shape& shape) {
        auto begin = shape.begin();
        auto end = shape.end();
        while (begin != end && *begin == 1)
            begin++;
        while (begin != end && *(end-1) == 1)
            end--;
        Shape trimmedShape(end - begin, 1);
        std::copy(begin, end, trimmedShape.begin());
        return trimmedShape;
    };

    // Check that output shapes are broadcastable => can be scheduled
    const auto& body_results = m_body->get_results();
    PartialShape outPShape = body_results[0]->get_shape();
    for (size_t i = 0; i < body_results.size(); i++) {
        auto shape_i = body_results[i]->get_shape();
        auto outputShape_i = std::get<0>(outputShapes[i]);
        // Check that the produced output shape corresponds to the passed shape
        // Some produced shapes may have been changed to be broadcastable (e.g. blocked + planar outputs),
        // so we need to remove leading and trailing "1" before the comparison
        PartialShape pShape_i(skipStartEndOnes(shape_i));
        bool compatibleWithPassedShape = PartialShape::broadcast_merge_into(pShape_i, skipStartEndOnes(outputShape_i),
                                                                              ::ngraph::op::AutoBroadcastType::NUMPY);
        NODE_VALIDATION_CHECK(this, ov::shape_size(shape_i) == ov::shape_size(outputShape_i) &&
                              compatibleWithPassedShape, "Inferred and passed results shapes are incompatible for snippet ",
                              get_friendly_name(), " : ", shape_i, " vs ", outputShape_i, ".");
        // Check that output shapes are broadcastable to each other => can be scheduled
        bool compatibleWithOtherOutputs = PartialShape::broadcast_merge_into(outPShape, shape_i,
                                                               ::ngraph::op::AutoBroadcastType::NUMPY);
        NODE_VALIDATION_CHECK(this, compatibleWithOtherOutputs, "Snippets output shapes must be numpy broadcastable");
    }
    exec_domain = outPShape.get_shape();
    return exec_domain;
}

void snippets::op::Subgraph::convert_to_snippet_dialect() {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::convert_to_snippet_dialect")
    auto skip_matching_domain = [](const std::shared_ptr<const ov::Node>& n) -> bool {
        return n->get_input_shape(0).back() != 1;
    };
    ngraph::pass::Manager manager;
    manager.register_pass<snippets::pass::ConvertConstantsToScalars>();
    manager.register_pass<snippets::pass::ConvertPowerToPowerStatic>();
    manager.register_pass<snippets::pass::InsertLoad>();
    manager.register_pass<snippets::pass::InsertStore>();
    manager.register_pass<snippets::pass::InsertMoveBroadcast>();
    manager.register_pass<snippets::pass::LoadMoveBroadcastToBroadcastLoad>();
    // Note that, BrodacastMove is typically inserted right after the Load. Such cases are typical for
    // simple subgraphs where one of the ngraph::op's inputs is broadcasted to match the larger one. However, BroadcastMove
    // could also be inserted after the ngraph::op, if the op input don't need broadcasting, but the the output does
    // (for example, to match the larger output of a child node). In such cases, Loads (and Stores) should be replaced
    // with ScalarLoads (ScalarStores) to avoid invalid read in vector Tile. Graph example:
    // Parameter_0    Parameter_1        Parameter_2
    // [1,2,5,16]      [1,2,5,1]          [1,2,5,1]
    //   Load        BroadcastLoad         Load*       Scalar
    //          Add                             Subtract
    //            \___________     ___________BroadcastMove
    //                        \   /
    //                       Multiply
    //                         Store
    //                        Result
    // Note: Load* should be replaced with ScalarLoad in this example to avoid invalid read in vector Tile.
    if (!exec_domain.empty() && exec_domain.back() != 1) {
        manager.register_pass<snippets::pass::ReplaceLoadsWithScalarLoads>();
        manager.register_pass<snippets::pass::ReplaceStoresWithScalarStores>();
        manager.get_pass_config()->
        set_callback<ngraph::snippets::pass::ReplaceLoadsWithScalarLoads>(skip_matching_domain);
        manager.get_pass_config()->
        set_callback<ngraph::snippets::pass::ReplaceStoresWithScalarStores>(skip_matching_domain);
    }
    manager.run_passes(m_body);
}

snippets::Schedule snippets::op::Subgraph::generate(const BlockedShapeVector& output_shapes,
                                                    const BlockedShapeVector& input_shapes,
                                                    const void* compile_params) {
    canonicalize(output_shapes, input_shapes);
    return generate(compile_params);
}

snippets::Schedule snippets::op::Subgraph::generate(const BlockedShapeVector& output_shapes,
                                                    const BlockedShapeVector& input_shapes,
                                                    ngraph::pass::Manager& opt,
                                                    const void* compile_params) {
    canonicalize(output_shapes, input_shapes);
    return generate(opt, compile_params);
}

snippets::Schedule snippets::op::Subgraph::generate(const void* compile_params) {
    auto mngr = ngraph::pass::Manager();
    return generate(mngr, compile_params);
}

snippets::Schedule snippets::op::Subgraph::generate(ngraph::pass::Manager& opt, const void* compile_params) {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::generate")
    NGRAPH_CHECK(m_generator != nullptr, "generate is called while generator is not set");
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

    return {exec_domain, false /*canBeLinearized*/, ptr};
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
