// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_MLIR_FOR_GPU

#    include "mlir_primitive.hpp"

#    include <memory>
#    include <vector>

#    include "intel_gpu/op/mlir_op.hpp"
#    include "intel_gpu/primitives/mlir_primitive.hpp"
#    include "mlir_primitive_inst.h"
#    include "plugin/transformations/mlir/interface/gpu_runtime.hpp"
#    include "register.hpp"
#    include "registry/implementation_map.hpp"

namespace cldnn::common {

namespace {

struct mlir_primitive_impl : typed_primitive_impl<mlir_primitive> {
    using parent = typed_primitive_impl<mlir_primitive>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::common::mlir_primitive_impl)

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<mlir_primitive_impl>(*this);
    }

    mlir_primitive_impl() : parent() {}

    explicit mlir_primitive_impl(const mlir_primitive_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& /*arg*/) override {}

    event::ptr execute_impl(const std::vector<event::ptr>& dependent_events, mlir_primitive_inst& instance) override {
        const auto& prim = instance.node->get_primitive();
        const auto& mlir_op = prim->op;
        OPENVINO_ASSERT(mlir_op != nullptr, "[GPU] MLIROp is not set for mlir_primitive '", prim->id, "'");

        // The memref types are baked into the compiled program, so the physical layout must match exactly.
        for (size_t i = 0; i < instance.inputs_memory_count(); ++i) {
            const auto& l = instance.input_memory_ptr(i)->get_layout();
            const auto required = prim->get_required_input_format(i, l.get_rank());
            OPENVINO_ASSERT(l.format == required && !l.data_padding,
                            "[GPU] mlir_primitive '",
                            prim->id,
                            "': input ",
                            i,
                            " is expected in '",
                            format(required).to_string(),
                            "' format without padding, got ",
                            l.to_short_string());
        }

        const auto& program = mlir_op->get_program();
        auto* runtime = instance.get_network().gc_runtime();
        OPENVINO_ASSERT(runtime != nullptr, "[GPU] MLIR runtime is not available for mlir_primitive '", prim->id, "'");
        return program->execute(*runtime, *mlir_op, instance, dependent_events, instance.needs_completion_event() || instance.is_output());
    }

    static std::unique_ptr<primitive_impl> create(const mlir_primitive_node& arg, const kernel_impl_params& /*params*/) {
        return std::make_unique<mlir_primitive_impl>(arg);
    }

    void init_kernels(const kernels_cache& /*kernels_cache*/, const kernel_impl_params& /*params*/) override {}

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }
    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    [[nodiscard]] bool is_cpu() const override {
        return false;
    }

    [[nodiscard]] bool supports_replay() const override {
        return false;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MLIRPrimitiveImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<mlir_primitive>());
    return mlir_primitive_impl::create(static_cast<const mlir_primitive_node&>(node), params);
}

in_out_fmts_t MLIRPrimitiveImplementationManager::query_formats(const program_node& node) const {
    OPENVINO_ASSERT(node.is_type<mlir_primitive>());
    const auto& prim = node.as<mlir_primitive>().get_primitive();

    // Inputs: the format the compiled MLIR function expects
    std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
    for (size_t i = 0; i < in_fmts.size(); ++i) {
        in_fmts[i] = prim->get_required_input_format(i, node.get_input_layout(i).get_rank());
    }

    // Outputs: always plain, must match mlir_primitive_inst::calc_output_layouts.
    std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);
    for (size_t i = 0; i < out_fmts.size(); ++i) {
        out_fmts[i] = format::get_default_format(node.get_output_layout(i).get_rank());
    }
    return {in_fmts, out_fmts};
}

namespace detail {

attach_mlir_primitive_common::attach_mlir_primitive_common() {
    implementation_map<mlir_primitive>::add(impl_types::common, shape_types::dynamic_shape, mlir_primitive_impl::create, {}, {});
    implementation_map<mlir_primitive>::add(impl_types::common, mlir_primitive_impl::create, {});
}

}  // namespace detail

}  // namespace cldnn::common

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::common::mlir_primitive_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::mlir_primitive)

#endif  // ENABLE_MLIR_FOR_GPU
