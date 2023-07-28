// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

using P2Btype = std::unordered_map<std::shared_ptr<ov::opset1::Parameter>, std::unordered_set<ov::label_t>>;

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FindBatch;

class TRANSFORMATIONS_API SymbolicOptimizations;
class TRANSFORMATIONS_API SymbolicPOC;
class TRANSFORMATIONS_API ChainedMaximumOptimization;
class TRANSFORMATIONS_API NopBroadcast;
class TRANSFORMATIONS_API BroadcastOnes;
class TRANSFORMATIONS_API RemoveSliceBeforeGatherElements;
class TRANSFORMATIONS_API ReshapeUpThroughBEAWithConstScalar;

class TRANSFORMATIONS_API SharedTransposeOptimization;

class TRANSFORMATIONS_API GroupedSliceToVSplitOptimization;
class TRANSFORMATIONS_API ChainedReshapeOptimization;

class TRANSFORMATIONS_API RPE_Optimization;
class TRANSFORMATIONS_API Fused_RPE_MHA_Replacer;

class TRANSFORMATIONS_API ApplyTableOfEquivalence;
class TRANSFORMATIONS_API OptimizeLabelsUsedAsValues;
class TRANSFORMATIONS_API LabelResolvingThroughSelect;

class TRANSFORMATIONS_API DeReshapeMatMul;
class TRANSFORMATIONS_API DeReshapeMatMulWithComplications;

}  // namespace pass
}  // namespace ov

class ov::pass::FindBatch : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("FindBatch");
    FindBatch(bool detach_detection_output = false, bool track = true)
        : track(track),
          detach_do(detach_detection_output) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

protected:
    bool track = true, detach_do = false;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief runs optimizations which are based on symbolic shape inference
 */
class ov::pass::SymbolicOptimizations : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("SymbolicOptimizations", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class ov::pass::SymbolicPOC : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("SymbolicPOC");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class ov::pass::ChainedMaximumOptimization : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ChainedMaximumOptimization", "0");
    ChainedMaximumOptimization();
};

class ov::pass::NopBroadcast : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("NopBroadcast", "0");
    NopBroadcast();
};

class ov::pass::BroadcastOnes : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("BroadcastOnes", "0");
    BroadcastOnes();
};

class ov::pass::DeReshapeMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DeReshapeMatMul", "0");
    DeReshapeMatMul();
};

class ov::pass::DeReshapeMatMulWithComplications : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DeReshapeMatMulWithComplications", "0");
    DeReshapeMatMulWithComplications();
};

class ov::pass::LabelResolvingThroughSelect : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("LabelResolvingThroughSelect", "0");
    LabelResolvingThroughSelect();
};

class ov::pass::RemoveSliceBeforeGatherElements : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveSliceBeforeGatherElements", "0");
    RemoveSliceBeforeGatherElements();
};

class ov::pass::ChainedReshapeOptimization : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ChainedReshapeOptimization", "0");
    ChainedReshapeOptimization();
};

class ov::pass::ReshapeUpThroughBEAWithConstScalar : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapeUpThroughBEAWithConstScalar", "0");
    ReshapeUpThroughBEAWithConstScalar();
};

class ov::pass::RPE_Optimization : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoEP_Optimization", "0");
    RPE_Optimization();
};

class ov::pass::Fused_RPE_MHA_Replacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("Fused_RPE_MHA_Replacer", "0");
    Fused_RPE_MHA_Replacer();
};

class ov::pass::SharedTransposeOptimization : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("SharedTransposeOptimization", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class ov::pass::GroupedSliceToVSplitOptimization : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GroupedSliceToVSplitOptimization", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class ov::pass::ApplyTableOfEquivalence : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ApplyTableOfEquivalence", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class ov::pass::OptimizeLabelsUsedAsValues : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("OptimizeLabelsUsedAsValues", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

namespace ov {
class DimensionTracker;

namespace batch_util {
void mark_batch(const std::shared_ptr<ov::opset1::Parameter>& parameter,
                P2Btype& map,
                const std::unordered_set<label_t>& batches);
void mark_no_batch(const std::shared_ptr<ov::opset1::Parameter>& parameter, P2Btype& map);
void mark_layout_independent_batch(const std::shared_ptr<ov::opset1::Parameter>& parameter,
                                   const std::shared_ptr<ov::Node>& result,
                                   P2Btype& map);
void mark_with_unique_dimension_labels(const std::shared_ptr<Model>& m, const ov::DimensionTracker& dt);
void restore_original_dimensions(
    const std::map<std::shared_ptr<ov::opset1::Parameter>, ov::PartialShape>& parameter_to_shape,
    bool leave_batch_dynamic = true);
bool check_batch_tracks_through_all_the_nodes(const std::shared_ptr<ov::Model>& m);
P2Btype find_batch(const std::shared_ptr<ov::Model>& m);
bool detach_detection_output(const std::shared_ptr<ov::Model>& f);
}  // namespace batch_util
}  // namespace ov
