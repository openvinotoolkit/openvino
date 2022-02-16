#include "check_network_batchable.hpp"

#include "dimension_tracker.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/dimension_tracking.hpp"
#include "transformations/init_node_info.hpp"

namespace InferenceEngine {
namespace details {

bool isNetworkBatchable(std::shared_ptr<ngraph::Function> function) {
    // find the batch dim
    ov::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<ov::pass::FindBatch>();
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
            return false;
        // check the batch dim: either 0th (and the original batch size of 1) or none
        if (shape.size() && ov::DimensionTracker::get_label(shape[0])) {
            const auto& static_shape = input->get_shape();
            if (static_shape[0] != 1)
                return false;
            any_batched_inputs = true;
        } else {
            // if the 0-th dim is not for the batch, then we support only the case when NONE dimension is batch
            for (size_t s = 1; s < shape.size(); s++)
                if (ov::DimensionTracker::get_label(shape[s]))
                    return false;
        }
    }
    return any_batched_inputs;
}

}  // namespace details
}  // namespace InferenceEngine