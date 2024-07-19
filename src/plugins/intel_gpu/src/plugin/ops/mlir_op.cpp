// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/tensor_accessor.hpp"
#include "openvino/core/partial_shape.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/generic_primitive.hpp"

namespace ov {
namespace op {
namespace mlir {
using MLIRSubgraph = ov::op::Op;
}  // namespace mlir
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

void CreateMLIRSubgraphOp(ProgramBuilder& p, const std::shared_ptr<ov::op::mlir::MLIRSubgraph>& op) {
    cldnn::generic_primitive::execute_function execute_f = [op](
            const std::vector<cldnn::event::ptr>& dependent_events,
            cldnn::stream& stream,
            const std::vector<cldnn::memory::ptr>& inputs,
            const std::vector<cldnn::memory::ptr>& outputs) {
        // Synchronization as evalute() may be a CPU code
        if (stream.get_queue_type() == cldnn::QueueTypes::out_of_order) {
            for (auto& ev : dependent_events) {
                ev->wait();
            }
        } else {
            stream.finish();
        }

        cldnn::event::ptr ev = stream.create_user_event(false);

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        for (size_t i = 0; i < inputs.size(); i++)
            input_host_tensors.push_back(make_tensor(inputs[i]->get_layout(), inputs[i]->lock(stream, cldnn::mem_lock_type::read)));

        for (size_t i = 0; i < outputs.size(); i++)
            output_host_tensors.push_back(make_tensor(outputs[i]->get_layout(), outputs[i]->lock(stream, cldnn::mem_lock_type::write)));

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute MLIROp ", op->get_friendly_name());

        for (size_t i = 0; i < inputs.size(); i++)
            inputs[i]->unlock(stream);

        for (size_t i = 0; i < outputs.size(); i++)
            outputs[i]->unlock(stream);

        ev->set();
        return ev;
    };
    cldnn::generic_primitive::shape_infer_function shape_infer_f = [&op](
            const std::vector<ov::PartialShape>& input_shapes) -> std::vector<ov::PartialShape> {
        // Dummy shape infer
        return {input_shapes[0]};
    };

    auto inputs = p.GetInputInfo(op);
    const std::string layerName = layer_type_name_ID(op);
    const size_t num_outputs = op->get_output_size();

    const cldnn::generic_primitive primitive(layerName,
                                             inputs,
                                             execute_f,
                                             shape_infer_f,
                                             num_outputs,
                                             get_output_data_types(op));

    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(mlir, MLIRSubgraph);

}  // namespace intel_gpu
}  // namespace ov
