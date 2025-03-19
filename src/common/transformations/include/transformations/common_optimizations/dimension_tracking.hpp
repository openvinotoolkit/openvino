// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

using P2Btype =
    std::unordered_map<std::shared_ptr<ov::op::v0::Parameter>, std::unordered_set<std::shared_ptr<ov::Symbol>>>;

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FindBatch;

}  // namespace pass
}  // namespace ov

class ov::pass::FindBatch : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("FindBatch");
    FindBatch(bool detach_detection_output = false, bool track = true)
        : track(track),
          detach_do(detach_detection_output) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

protected:
    bool track = true, detach_do = false;
};

namespace ov {

namespace batch_util {
void mark_batch(const std::shared_ptr<ov::op::v0::Parameter>& parameter,
                P2Btype& map,
                const std::unordered_set<std::shared_ptr<Symbol>>& batches);
void mark_no_batch(const std::shared_ptr<ov::op::v0::Parameter>& parameter, P2Btype& map);
void mark_layout_independent_batch(const std::shared_ptr<ov::op::v0::Parameter>& parameter,
                                   const std::shared_ptr<ov::Node>& result,
                                   P2Btype& map);
void mark_with_unique_dimension_symbols(const std::shared_ptr<Model>& m);
void restore_original_dimensions(
    const std::shared_ptr<ov::Model>& model,
    const std::map<std::shared_ptr<ov::op::v0::Parameter>, ov::PartialShape>& parameter_to_shape,
    bool leave_batch_dynamic = true,
    bool clear_symbols = false);
bool check_batch_tracks_through_all_the_nodes(const std::shared_ptr<ov::Model>& m);
P2Btype find_batch(const std::shared_ptr<ov::Model>& m);
bool detach_detection_output(const std::shared_ptr<ov::Model>& f);
}  // namespace batch_util
}  // namespace ov
