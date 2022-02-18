#include "check_network_batchable.hpp"

#include "dimension_tracker.hpp"
#include "ie_ngraph_utils.hpp"
#include "ngraph/opsets/opset.hpp"
#include "openvino/op/detection_output.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/dimension_tracking.hpp"
#include "transformations/init_node_info.hpp"

namespace InferenceEngine {
namespace details {

NetworkBatchAbility isNetworkBatchable(const CNNNetwork& orig_network, const std::string& deviceNameWithoutBatch) {
    CNNNetwork clonedNetwork(cloneNetwork(orig_network));
    auto function = clonedNetwork.getFunction();
    // find the batch dim
    ov::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<ov::pass::FindBatch>();  // TODO: disable DetectionOutput
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
        if (shape.size() && ov::DimensionTracker::get_label(shape[0])) {
            const auto& static_shape = input->get_shape();
            if (static_shape[0] != 1)
                return NetworkBatchAbility::NO;
            else
                any_batched_inputs = true;
        } else {
            // if the 0-th dim is not for the batch, then we support only the case when NONE dimension is batch
            for (size_t s = 1; s < shape.size(); s++)
                if (ov::DimensionTracker::get_label(shape[s]))
                    return NetworkBatchAbility::NO;
        }
    }
    if (!any_batched_inputs)
        return NetworkBatchAbility::NO;

    const std::string detectionOutputOpName = ov::op::v0::DetectionOutput::get_type_info_static().name;
    const std::string resultOpName = ngraph::op::Result::get_type_info_static().name;
    // have to execute the DetectionOutput separately (without batching)
    // as this layer mix-in the values from the different inputs (batch id)
    bool bDetectionOutput = false;
    for (auto&& node : orig_network.getFunction()->get_ops()) {
        auto isDetectionOutputParent = [&detectionOutputOpName](decltype(node)& nd) {
            for (size_t n = 0; n < nd->get_input_size(); n++) {
                // the code below doesn't need to separate the versions (opsets) of the DetectionOutput
                // so type_info name check is enough
                // (if in a future there will be a new ver that doesn't mix the batch, this will be new op)
                if (detectionOutputOpName == nd->get_input_node_ptr(n)->get_type_info().name)
                    return true;
            }
            return false;
        };

        if ((detectionOutputOpName == node->get_type_info().name) ||
            ((resultOpName == node->get_type_info().name) && isDetectionOutputParent(node))) {
            node->get_rt_info()["affinity"] = deviceNameWithoutBatch;
            bDetectionOutput = true;
        } else {
            node->get_rt_info()["affinity"] = "BATCH";
        }
    }
    return bDetectionOutput ? NetworkBatchAbility::WITH_HETERO : NetworkBatchAbility::AS_IS;
}

}  // namespace details
}  // namespace InferenceEngine