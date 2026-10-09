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
        auto* op = prim->op.get();
        auto* mlir_op = op && ov::is_type<ov::intel_gpu::op::MLIROp>(op) ? static_cast<ov::intel_gpu::op::MLIROp*>(op) : nullptr;
        OPENVINO_ASSERT(mlir_op != nullptr, "[GPU] MLIROp is not set for mlir_primitive '", prim->id, "'");
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
};

}  // namespace

std::unique_ptr<primitive_impl> MLIRPrimitiveImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<mlir_primitive>());
    return mlir_primitive_impl::create(static_cast<const mlir_primitive_node&>(node), params);
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
