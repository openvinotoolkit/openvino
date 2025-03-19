// copyright (c) 2023 intel corporation
// spdx-license-identifier: apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"

#include "concatenation_inst.h"
#include "crop_inst.h"

#include <utility>
#include <list>
#include <vector>

using namespace cldnn;
namespace cldnn {
struct concat_noop_optimization : pattern_match_optimization_typed<concat_noop_optimization, concatenation> {
    // Removes concatenation nodes with single input.
    using base = pattern_match_optimization_typed<concat_noop_optimization, concatenation>;
    using base::base;

    bool match(concatenation_node& node);
    bool optimize(concatenation_node& node);
};
struct concat_in_place_optimization : pattern_match_optimization_typed<concat_in_place_optimization, concatenation> {
    // Performs in-place concat optimization.
    // Padding of predecessors is updated to use single buffer by all, which is output from concatenation.
    // Then concatenation can be optimized out, as memory will be correctly filled by previous nodes.
    // If one of the dependencies is also optimized-out concatenation, then cascade adjusment is performed to update it.
    // This optimization is expected to be executed in some topological order, as cascade adjustment is performed backwards.
    using base = pattern_match_optimization_typed<concat_in_place_optimization, concatenation>;
    using base::base;

    // Runs concat in-place optimization and adds already optimized concatenations that need re-optimization to
    // `needs_reoptimization`.
    void optimize_cascade(concatenation_node& node, std::list<concatenation_node*>& need_reoptimization);
    static void update_in_place_concat_paddings(layout& concat_layout,
                                 std::vector<layout>& preds_layouts,
                                 size_t concat_axis,
                                 bool is_runtime);
    bool match(concatenation_node& node);
    static bool match(const program_node& concat_node,
                      kernel_impl_params& concat_params,
                      std::vector<kernel_impl_params>& pred_params,
                      bool is_runtime = false);
    bool optimize(concatenation_node& node) {
        std::list<concatenation_node*> need_reopt;
        optimize_cascade(node, need_reopt);
        while (!need_reopt.empty()) {
            auto& prop = *need_reopt.front();
            need_reopt.pop_front();
            if (match(prop))
                optimize_cascade(prop, need_reopt);
            else
                // TODO: Revert extra padding when cascade adjustment failed.
                prop.can_be_optimized(false);
        }
        return false;  // node not invalidated
    }
};

struct crop_in_place_optimization : pattern_match_optimization_typed<crop_in_place_optimization, crop> {
    // Performs in-place crop optimization.
    using base = pattern_match_optimization_typed<crop_in_place_optimization, crop>;
    using base::base;

    static bool can_crop_be_optimized_along_feature(const layout& crop_layout,
                                                    const layout& input_layout);
    static bool can_crop_be_optimized_simple_data_format(const layout& crop_layout,
                                                         const layout& input_layout);
    bool match(crop_node& node);
    static bool match(const program_node& node,
                      kernel_impl_params& crop_params,
                      layout& input_layout,
                      bool is_runtime = false);
    bool optimize(crop_node& node);
    static void update_in_place_crop_padding_along_feature(const program_node& node,
                                                           layout& crop_layout,
                                                           layout& pred_layout,
                                                           const tensor offsets,
                                                           size_t crop_axis,
                                                           bool is_runtime);
    static void update_in_place_crop_padding_simple_data_format(layout& crop_layout,
                                                                layout& pred_layout,
                                                                std::pair<const program_node*, layout>& user_info,
                                                                const tensor offsets,
                                                                size_t crop_axis,
                                                                bool is_runtime);
};

} // namespace cldnn