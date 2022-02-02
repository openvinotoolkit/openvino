// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <transformations_visibility.hpp>

#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/opsets/opset1.hpp>

using P2Btype = std::unordered_map<std::shared_ptr<ov::opset1::Parameter>, std::unordered_set<size_t>>;

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FindBatch;
class TRANSFORMATIONS_API FindBatchDontTrack;

}  // namespace pass
}  // namespace ov

class ov::pass::FindBatch: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("FindBatch");
    FindBatch(bool track = true) : track(track) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
protected:
    bool track = true;
};

class ov::pass::FindBatchDontTrack: public ov::pass::FindBatch {
public:
    FindBatchDontTrack() : FindBatch(false) {}
};

namespace ov {
class DimensionTracker;

namespace batch_util {
    void mark_batch(const std::shared_ptr<ov::opset1::Parameter> &parameter, P2Btype &map, const std::unordered_set<size_t> &batches);
    void mark_no_batch(const std::shared_ptr<ov::opset1::Parameter> &parameter, P2Btype &map);
    void mark_layout_independent_batch(const std::shared_ptr<ov::opset1::Parameter> &parameter, const std::shared_ptr<ov::Node> & result, P2Btype &map);
    void mark_with_unique_dimension_labels(const std::shared_ptr<Model> &m, const ov::DimensionTracker &dt);
    void restore_original_dimensions(
            const std::map<std::shared_ptr<ov::opset1::Parameter>, ov::PartialShape>& parameter_to_shape, bool leave_batch_dynamic = true);
    bool check_batch_tracks_through_all_the_nodes(const std::shared_ptr<ov::Model>& m);
    P2Btype find_batch(const std::shared_ptr<ov::Model> &m);
} // namespace batch_util
} // namespace ov
