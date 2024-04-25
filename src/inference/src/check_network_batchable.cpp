#include "check_network_batchable.hpp"

#include "openvino/core/dimension.hpp"
#include "openvino/op/detection_output.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/dimension_tracking.hpp"
#include "transformations/init_node_info.hpp"

namespace ov {
namespace details {
namespace {
bool model_has_suitable_do(const std::shared_ptr<const ov::Model>& model) {
    bool bDetectionOutput = false;
    for (auto& result_node : model->get_results()) {
        auto do_node = result_node->input_value(0).get_node_shared_ptr();
        std::shared_ptr<ov::Node> convert_node;
        if (ov::is_type<ov::op::v0::Convert>(do_node)) {  // cases with do->convert->result
            convert_node = do_node;
            do_node = convert_node->get_input_node_shared_ptr(0);
        }
        auto detectionOutputBase = std::dynamic_pointer_cast<ov::op::util::DetectionOutputBase>(do_node);
        if (detectionOutputBase) {
            bDetectionOutput = true;
        }
    }
    return bDetectionOutput;
}
}  // namespace

NetworkBatchAbility is_model_batchable(const std::shared_ptr<const ov::Model>& model,
                                       const std::string& deviceNameWithoutBatch,
                                       bool strictly_track_dims) {
    auto function = model->clone();
    // find the batch dim
    ov::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::FindBatch>(true, strictly_track_dims);
    m.run_passes(function);
    bool any_batched_inputs = false;
    // do not reshape/re-batch originally batched networks and when there are no inputs with the N* layouts
    // input(s) should have the batch dim as the first dim or none (current limitation of the auto-batching impl)
    const auto& params = function->get_parameters();
    for (size_t input_id = 0; input_id < params.size(); input_id++) {
        const auto& input = params[input_id];
        const auto& shape = input->get_partial_shape();
        // currently no plugin support batched execution for dynamic networks
        if (shape.is_dynamic())
            return NetworkBatchAbility::NO;
        // check the batch dim: either 0th (and the original batch size of 1) or none
        if (shape.size() && shape[0].has_symbol()) {
            const auto& static_shape = input->get_shape();
            if (static_shape[0] != 1)
                return NetworkBatchAbility::NO;
            else
                any_batched_inputs = true;
        } else {
            // if the 0-th dim is not for the batch, then we support only the case when NONE dimension is batch
            for (size_t s = 1; s < shape.size(); s++)
                if (shape[s].has_symbol())
                    return NetworkBatchAbility::NO;
        }
    }
    if (!any_batched_inputs)
        return NetworkBatchAbility::NO;

    return model_has_suitable_do(model) ? NetworkBatchAbility::WITH_HETERO : NetworkBatchAbility::AS_IS;
}

std::shared_ptr<const ov::Model> apply_batch_affinity(const std::shared_ptr<const ov::Model>& model_,
                                                      const std::string& deviceNameWithoutBatch) {
    auto model = model_->clone();
    for (auto&& node : model->get_ops())
        node->get_rt_info()["affinity"] = "BATCH";  // default affinity (ignored if HETERO is not triggered)
    // have to execute the DetectionOutput separately (without batching)
    // as this layer does mix-in the values from the different inputs (batch id)
    for (auto& result_node : model->get_results()) {
        auto do_node = result_node->input_value(0).get_node_shared_ptr();
        std::shared_ptr<ov::Node> convert_node;
        if (ov::is_type<ov::op::v0::Convert>(do_node)) {  // cases with do->convert->result
            convert_node = do_node;
            do_node = convert_node->get_input_node_shared_ptr(0);
        }
        // the code below doesn't need to separate the versions (opsets) of the DetectionOutput
        // so base class  check is enough
        auto detectionOutputBase = std::dynamic_pointer_cast<ov::op::util::DetectionOutputBase>(do_node);
        if (detectionOutputBase) {
            result_node->get_rt_info()["affinity"] = deviceNameWithoutBatch;
            do_node->get_rt_info()["affinity"] = deviceNameWithoutBatch;
            if (convert_node)
                convert_node->get_rt_info()["affinity"] = deviceNameWithoutBatch;
        }
    }
    return model;
}

}  // namespace details
}  // namespace ov
