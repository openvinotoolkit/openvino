// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/op/util/sub_graph_base.hpp>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/op.hpp"
#include "snippets/generator.hpp"
#include "snippets/runtime_configurator.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/pass/manager.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/pass/positioned_pass.hpp"


namespace ov {
namespace snippets {
namespace op {

/**
 * @interface Subgraph
 * @brief An operation that is implemented by a model
 * @ingroup snippets
 */
class Subgraph : public ov::op::util::SubGraphOp {
public:
    OPENVINO_OP("Subgraph", "SnippetsOpset", ov::op::util::SubGraphOp);
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
    using Layout = std::vector<size_t>;
    using BlockedShape = std::pair<VectorDims, Layout>;
    using BlockedShapeVector = std::vector<BlockedShape>;

    Subgraph() = default;

    Subgraph(const OutputVector& args, const std::shared_ptr<ov::Model>& body);

    Subgraph(const NodeVector& args, const std::shared_ptr<ov::Model>& body);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    // we introduce this method instead of using SubGraphOp::get_function()
    // to align naming with other methods
    const std::shared_ptr<ov::Model>& body_ptr() const { return m_bodies[0]; }
    std::shared_ptr<ov::Model>& body_ptr() { return m_bodies[0]; }

    const ov::Model& body() const { return *m_bodies[0]; }
    ov::Model& body() { return *m_bodies[0]; }

    const std::shared_ptr<ov::snippets::Generator>& get_generator() const { return m_generator; }
    std::shared_ptr<ov::snippets::Generator>& get_generator() { return m_generator; }

    size_t get_virtual_port_count() const { return m_virtual_port_count; }
    bool is_quantized() const { return config.m_is_quantized; }
    bool has_domain_sensitive_ops() const { return config.m_has_domain_sensitive_ops; }

    // plugin sets generator for a snippet to some specific generator.
    // it's going to be replaced with Jitters table later
    void set_generator(std::shared_ptr<ov::snippets::Generator> generator);
    void set_tile_rank(size_t rank) {tile_rank = rank;}
    void set_virtual_port_count(size_t count);

    void print() const;

    IShapeInferSnippets::Result shape_infer(const std::vector<VectorDimsRef>& input_shapes);
    VectorDims infer_master_shape();

    std::shared_ptr<Subgraph> clone() const;

    const std::shared_ptr<RuntimeConfigurator>& get_runtime_configurator() const;
    const std::shared_ptr<RuntimeConfig>& update_runtime_config() const;

    static auto wrap_node_as_subgraph(const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<Subgraph>;
    static void fill_empty_output_names(const Output<Node>& target_output_node, const Output<Node>& replacement_output_node);

    // Non-scalar Constants are tokenized as Parameters inside Subgraph body but some operations with constant inputs
    // should have explicit Constants even if they're non-scalar (Reshape, Transpose, Broadcast)
    // This check returns True if Constant op which is input of this op should be inside Subgraph body
    static auto constant_input_should_be_inside_body(const std::shared_ptr<ov::Node>& node) -> bool;
    static bool check_broadcast(const std::shared_ptr<const ov::Node>& node) noexcept;
    // Return estimated unique buffer count (upper bound). It's needed for tokenization
    static auto get_estimated_buffer_count(const ov::NodeVector& ops) -> size_t;
    static auto is_domain_sensitive_op(const std::shared_ptr<ov::Node>& op) -> bool;
    static auto is_shape_infer_op(const std::shared_ptr<ov::Node>& op) -> bool;

    void data_flow_transformations(const BlockedShapeVector& blocked_input_shapes = {},
                                   const std::vector<ov::element::Type>& input_precisions = {},
                                   const std::vector<ov::element::Type>& output_precisions = {},
                                   const std::vector<snippets::pass::Manager::PositionedPassBase>& = {}) const;

    void control_flow_transformations(size_t min_parallel_work_amount = 8, size_t min_kernel_work_amount = 256,
                                      const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory = std::make_shared<IShapeInferSnippetsFactory>(),
                                      const std::shared_ptr<lowered::pass::PassConfig>& lowered_pass_config = std::make_shared<lowered::pass::PassConfig>(),
                                      const std::vector<snippets::lowered::pass::PassPipeline::PositionedPassLowered>& lowered_backend_passes = {});

    Schedule generate(const void* compile_params = nullptr) const;
    Schedule generate(const BlockedShapeVector& blocked_input_shapes = {},
                      const std::vector<ov::element::Type>& input_precisions = {},
                      const std::vector<ov::element::Type>& output_precisions = {},
                      const std::vector<snippets::pass::Manager::PositionedPassBase>& data_flow_passes = {},
                      const std::shared_ptr<lowered::pass::PassConfig>& lowered_pass_config = std::make_shared<lowered::pass::PassConfig>(),
                      const std::vector<snippets::lowered::pass::PassPipeline::PositionedPassLowered>& lowered_backend_passes = {},
                      size_t min_parallel_work_amount = 8, size_t min_kernel_work_amount = 256,
                      const std::shared_ptr<IShapeInferSnippetsFactory>& factory = nullptr,
                      const void* compile_params = nullptr);

private:
    std::shared_ptr<lowered::LinearIR>
    convert_body_to_linear_ir(size_t min_parallel_work_amount = 8, size_t min_kernel_work_amount = 256,
                              const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory = std::make_shared<IShapeInferSnippetsFactory>());

    void init_config();
    // Count of Subgraph virtual ports:
    //  - Potential non-scalar Constants that will be created after some transformations (At the moment it's relevant only for FakeQuantize decomposition)
    // NOTE: To avoid overheads in each calculation of this count (for example, in validate_and_type_infer()),
    //       we should MANUALLY calculate it where it needed.
    size_t m_virtual_port_count = 0;
    size_t tile_rank = 0; // set by plugin to specify the number of dimensions processed in a single kernel call

    std::shared_ptr<lowered::LinearIR> m_linear_ir = nullptr;
    // This LinearIR is used for ShapeInfer and based on LinearIR state after ControlFlow transformations
    std::shared_ptr<lowered::LinearIR> m_shape_infer_linear_ir = nullptr;

    std::shared_ptr<ov::snippets::Generator> m_generator = nullptr;

    /**
    * @interface SubgraphConfig
    * @brief Config to optimize IR transformation pipeline. It indicates which transformations are necessary
    *       so the irrelevant ones could be skipped.
    */
    class SubgraphConfig {
    public:
        // True if Subgraph contains FakeQuantize -> FQ decomposition should be called
        bool m_is_quantized = false;
        // True if body has operations that don't support plugin-side domain optimizations
        // (e.g. Transpose, Softmax, MatMul in general doesn't support dimensions collapsing)
        bool m_has_domain_sensitive_ops = false;
        // True if Subgraph contains ops that are not applicable to auto broadcast rule.
        // (e.g. GroupNormalization, reshape)
        bool m_has_broadcast_sensitive_ops = false;
    } config;

    std::shared_ptr<ShapeInferSnippetsNode> m_shape_infer = nullptr;

    class OVShapeInfer : public ShapeInferSnippetsNode {
        std::shared_ptr<ov::Model> m_ov_body;
    public:
        explicit OVShapeInfer(const std::shared_ptr<ov::Model>& body);
        Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
    };
};

static inline auto create_body(const std::string& name, const ov::ResultVector& results, const ov::ParameterVector& parameters) ->
    std::shared_ptr<ov::Model> {
    auto body = std::make_shared<ov::Model>(results, parameters, name);
    return body;
}

static inline auto build_subgraph(const std::shared_ptr<ov::Node>& node, const ov::OutputVector& inputs,
                                  const std::shared_ptr<ov::Model>& body, const std::string& name = "")
    -> std::shared_ptr<Subgraph>{
    auto subgraph = std::make_shared<Subgraph>(inputs, body);
    copy_runtime_info(node, subgraph);
    subgraph->set_friendly_name(name.empty() ? node->get_friendly_name() : name);
    return subgraph;
}

// Need to update tensor name manually, since intel_cpu::Graph::Replicate() looks at input.get_shape().get_name();
// If subgraph->get_output_size() == 1, then the name will be restored correctly from the node name
auto inline update_out_tensor_name(const std::shared_ptr<ov::snippets::op::Subgraph>& subgraph) -> void {
    bool not_set = true;
    for (unsigned int i = 0; i < subgraph->get_output_size() && not_set; i++) {
        for (const auto& in : subgraph->get_output_target_inputs(i)) {
            if (ov::is_type<ov::op::v0::Result>(in.get_node())) {
                const auto& body_result = subgraph->body_ptr()->get_output_op(i);
                const auto& body_result_input = body_result->get_input_source_output(0);
                ov::snippets::op::Subgraph::fill_empty_output_names(
                        subgraph->output(i), body_result_input);
                not_set = false;
                break;
            }
        }
    }
}

}  // namespace op
}  // namespace snippets
}  // namespace ov
