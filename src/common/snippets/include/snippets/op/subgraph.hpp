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
    enum {DYNAMIC_DIMENSION = 0xffffffffffffffff};

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
    using BlockedShape = std::tuple<ngraph::PartialShape, ngraph::AxisVector, ngraph::element::Type>;
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

    // Return common memory size for all buffers in body. Should be called only after tileRank setting
    size_t get_buffer_scratchpad_size() const;
    size_t get_virtual_port_count() const { return m_virtual_port_count; }
    bool is_buffer_needed() const { return m_buffer_needed; }
    bool is_quantized() const { return config.m_is_quantized; }
    bool has_type_relaxed_ops() const { return config.m_has_type_relaxed_ops; }
    bool has_domain_sensitive_ops() const { return config.m_has_domain_sensitive_ops; }

    snippets::Schedule generate(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes, ngraph::pass::Manager& opt,
                                const void* compile_params = nullptr);
    snippets::Schedule generate(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes, const void* compile_params = nullptr);
    snippets::Schedule generate(ngraph::pass::Manager &opt, const void* compile_params = nullptr);
    snippets::Schedule generate(const void* compile_params = nullptr);
    ov::PartialShape canonicalize(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes);
    std::vector<PartialShape> reshape_body(const std::vector<PartialShape>& input_shapes);
    std::vector<Shape> reshape_body(const std::vector<Shape>& input_shapes);

    // plugin sets generator for a snippet to some specific generator.
    // it's going to be replaced with Jitters table later
    void set_generator(std::shared_ptr<ngraph::snippets::Generator> generator);
    void set_tile_rank(size_t newRank) {tileRank = newRank;}
    void set_virtual_port_count(const size_t count);
    void set_buffer_needed(const bool need);

    void print() const;
    void print_statistics(bool verbose);

    void serialize() const;
    void set_master_shape(ov::PartialShape new_shape) {master_shape = std::move(new_shape);}

    static auto wrap_node_as_subgraph(const std::shared_ptr<ngraph::Node>& node) -> std::shared_ptr<Subgraph>;
    static void fill_empty_output_names(const Output<Node>& target_output_node, const Output<Node>& replacement_output_node);

    // Non-scalar Constants are tokenized as Parameters inside Subgraph body but some operations with constant inputs
    // should have explicit Constants even if they're non-scalar (Reshape, Transpose, Broadcast)
    // This check returns True if Constant op which is input of this op should be inside Subgraph body
    static auto constant_input_should_be_inside_body(const std::shared_ptr<ov::Node>& node) -> bool;

private:
    void align_element_types(const BlockedShapeVector& outputShapes, const BlockedShapeVector& inputShapes);
    void convert_to_snippet_dialect();
    void init_config();
    // Count of Subgraph virtual ports:
    //  - Potential non-scalar Constants that will be created after some transformations (At the moment it's relevant only for FakeQuantize decomposition)
    // Need Buffer op or not
    //  - Buffers. All Buffers are considered as one common additional virtual port. So we cannot summarize them as potential non-scalar Constants
    // NOTE: To avoid overheads in each calculation of this count (for example, in validate_and_type_infer()),
    //       we should MANUALLY calculate it where it needed.
    size_t m_virtual_port_count = 0;
    bool m_buffer_needed = false;
    Shape exec_domain = {};
    std::shared_ptr<ov::Model> m_body = nullptr;
    std::shared_ptr<ngraph::snippets::Generator> m_generator = nullptr;

    // TODO: Change logic of insert Converts. This exec element type can be different for plugins
    const ov::element::Type execution_element_type = ov::element::f32;

    ov::PartialShape master_shape;
    size_t tileRank = 0; // set by plugin to specify the number of dimensions processed in a single kernel call

    /**
    * @interface SubgraphConfig
    * @brief Config to optimize IR transformation pipeline. It indicates which transformations are necessary
    *       so the irrelevant ones could be skipped.
    */
    class SubgraphConfig {
    public:
        // True if Subgraph contains FakeQuantize -> FQ decomposition should be called
        bool m_is_quantized = false;
        // True if we should align element types indise body
        bool m_is_needed_to_align_precision = false;
        // True if Subgraph contains TypeRelaxed nodes -> for several streams in tp mode we should copy body using mutexes
        // because TypeRelaxed::copy_with_new_inputs() isn't save-thread method
        bool m_has_type_relaxed_ops = false;
        // True if body has operations that don't support plugin-side domain optimizations
        // (e.g. Transpose, Softmax, MatMul in general doesn't support dimensions collapsing)
        bool m_has_domain_sensitive_ops = false;
        // True if we should go through whole body to check for where loops should be explicitly inserted.
        // Otherwise, we insert Loops on Parameters and Results - for example, it's optimized out for subgraph with only Eltwise ops
        bool m_explicit_loop_insertion = false;
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
