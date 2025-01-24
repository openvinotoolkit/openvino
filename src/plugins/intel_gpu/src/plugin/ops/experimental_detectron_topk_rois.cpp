// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/experimental_detectron_topkrois.hpp"

#include "intel_gpu/primitives/experimental_detectron_topk_rois.hpp"
#include "intel_gpu/primitives/arg_max_min.hpp"

namespace ov::intel_gpu {

namespace {

using namespace cldnn;

void CreateExperimentalDetectronTopKROIsOp(ProgramBuilder &p,
                                           const std::shared_ptr<ov::op::v6::ExperimentalDetectronTopKROIs> &op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    auto max_rois = static_cast<uint32_t>(op->get_max_rois());
    auto layer_name = layer_type_name_ID(op);
    auto argmax_layer_name = layer_name + "_topk";
    auto top_k_indices = arg_max_min(argmax_layer_name,
                                     {inputs[1]}, ov::op::TopKMode::MAX, max_rois, 0,
                                     ov::op::TopKSortType::SORT_VALUES, false, false, cldnn::data_types::i32);


    p.add_primitive(*op, top_k_indices);

    auto experimental_detectron_topk_layer = cldnn::experimental_detectron_topk_rois(layer_name,
                                                                                     {inputs[0], cldnn::input_info(argmax_layer_name)},
                                                                                      max_rois);

    p.add_primitive(*op, experimental_detectron_topk_layer);
}

} // namespace

REGISTER_FACTORY_IMPL(v6, ExperimentalDetectronTopKROIs);

}  // namespace ov::intel_gpu
