// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <openvino/core/model.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "snippets/generator.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Subgraph
 * @brief An operation that is implemented by a model
 * @ingroup snippets
 */
class Subgraph : public ngraph::op::Op {
public:
    OPENVINO_OP("Subgraph", "SnippetsOpset");

    // < 1, 42, 17, 15, 16> < 0, 1, 2, 3, 1>
    // should be:
    // A = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 1>
    // B = < 1,  1, 17, 15> -> < 1, 1, 17, 15, 16> < 0, 1, 2, 3, 1>
    // D = < 1, 42,  1, 1 > -> < 1, 3,  1,  1, 16> < 0, 1, 2, 3, 1> ???
    // C = A + B
    // C = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 1>
    //
    // how it works now (multi-demention broadcast):
    // [BroadcastLoad] doesn't perform post increment
    // [Load] performs += vlan
    // [ScalarLoad] performs += 1
    // A = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 1>
    // B = < 1,  1, 17, 15> -> < 1, 1, 17, 15,  1> < 0, 1, 2, 3, 1>
    // [A]     [B]
    // [Load]  [ScalarLoad] <- should consider AxisVector to choose right type of load
    //         [Broadcast]
    //   [Add]
    //  [Store]
    //    [C]
    // C = A + B
    // C = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 1>
    //
    // Multiple-dimension broadcasts support?
    // A = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 4>
    // B = < 1,  1, 17, 15> -> < 1, 1, 17, 15,  1> < 0, 1, 2, 3, 4>
    //
    // A = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 4>
    // B = < 1,  1, 17, 15> -> < 1, 3, 17, 15,  1> < 0, 1, 2, 3, 4>
    //
    // Collapse moat varying dimensions with broadcast
    // A = < 1, 42, 17, 15> -> < 1, 3, 17, 15, 16> < 0, 1, 2, 3, 1>
    // B = < 1,  1, 17, 15> -> < 1, 3, 17, 15,  1> < 0, 1, 2, 3, 1>
    //
    // Collapse for mixed broadcast
    // A = < 1, 3, 17, 15, 32> < 0, 1, 2, 3, 4>
    // B = < 1, 3, 17,  1, 32> < 0, 1, 2, 3, 4>
    // C = < 1, 3,  1, 15, 32> < 0, 1, 2, 3, 4>
    //
    // D = < 1, 3, 17, 15, 32> < 0, 1, 2, 3, 4>
    // E = < 1, 3, 17,  1, 32> < 0, 1, 2, 3, 4>
    using BlockedShape = std::tuple<ngraph::Shape, ngraph::AxisVector, ngraph::element::Type>;
    using BlockedShapeVector = std::vector<BlockedShape>;

    Subgraph(const OutputVector& args, std::shared_ptr<ov::Model> body);

    Subgraph(const NodeVector& args, std::shared_ptr<ov::Model> body);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    std::shared_ptr<ov::Model> get_body() const {
        return m_body;
    }

    std::shared_ptr<ngraph::snippets::Generator> get_generator() const {
        return m_generator;
    }

    size_t get_non_scalar_constants_count() const {
        return m_non_scalar_constants_count;
    }

    bool is_quantized() const {
        return config.m_is_quantized;
    }

    bool has_type_relaxed_ops() const {
        return config.m_has_type_relaxed_ops;
    }

    snippets::Schedule generate(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes, ngraph::pass::Manager& opt,
                                const void* compile_params = nullptr);
    snippets::Schedule generate(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes, const void* compile_params = nullptr);
    snippets::Schedule generate(ngraph::pass::Manager &opt, const void* compile_params = nullptr);
    snippets::Schedule generate(const void* compile_params = nullptr);
    Shape canonicalize(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes);

    // plugin sets generator for a snippet to some specific generator.
    // it's going to be replaced with Jitters table later
    void set_generator(std::shared_ptr<ngraph::snippets::Generator> generator);
    void set_non_scalar_constants_count(const size_t count);

    void print() const;
    void print_statistics(bool verbose);

    void serialize() const;

    static auto wrap_node_as_subgraph(const std::shared_ptr<ngraph::Node>& node) -> std::shared_ptr<Subgraph>;
    static void fill_empty_output_names(const Output<Node>& target_output_node, const Output<Node>& replacement_output_node);

private:
    void align_element_types(const BlockedShapeVector& outputShapes, const BlockedShapeVector& inputShapes);
    void convert_to_snippet_dialect();
    // Count of potentional non-scalar Consants that will be created after some tranformations
    // At the moment it's relevant only for FakeQuantize decomposition
    // NOTE: To avoid overheads in each calcution of this count (for example, in validate_and_type_infer()),
    //       we should MANUALLY calculate it where it needed.
    size_t m_non_scalar_constants_count = 0;
    Shape exec_domain = {};
    std::shared_ptr<ov::Model> m_body = nullptr;
    std::shared_ptr<ngraph::snippets::Generator> m_generator = nullptr;

    // TODO: Change logic of insert Converts. This exec element type can be different for plugins
    const ov::element::Type execution_element_type = ov::element::f32;

    // Config to know which transformations should be called.
    // It helps to avoid overheads of extra transformation calls
    struct {
        // True if Subgraph contains FakeQuantize -> FQ decomposition should be called
        bool m_is_quantized = false;
        // True if we should align element types indise body
        bool m_is_needed_to_align_precision = false;
        // True if Subgraph contains TypeRelaxed nodes -> for several streams in tp mode we should copy body using mutexes
        // because TypeRelaxed::copy_with_new_inputs() isn't save-thread method
        bool m_has_type_relaxed_ops = false;
    } config;
};

static inline std::ostream& operator<<(std::ostream& os, const op::Subgraph::BlockedShape& blocked_shape) {
    os << std::get<0>(blocked_shape) << " " << std::get<1>(blocked_shape) << " " << std::get<2>(blocked_shape);
    return os;
}

static inline auto create_body(std::string name, const ngraph::ResultVector& results, const ngraph::ParameterVector& parameters) ->
    std::shared_ptr<ov::Model> {
    auto body = std::make_shared<ov::Model>(results, parameters, name);
    return body;
};

static inline auto build_subgraph(const std::shared_ptr<ngraph::Node>& node, const ngraph::OutputVector& inputs,
                                  const std::shared_ptr<ov::Model>& body, const std::string name = "")
    -> std::shared_ptr<Subgraph>{
    auto subgraph = std::make_shared<Subgraph>(inputs, body);
    copy_runtime_info(node, subgraph);
    subgraph->set_friendly_name(name.empty() ? node->get_friendly_name() : name);
    return subgraph;
};

}  // namespace op
}  // namespace snippets
}  // namespace ngraph
