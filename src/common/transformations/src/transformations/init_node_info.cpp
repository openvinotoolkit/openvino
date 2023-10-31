// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/init_node_info.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/common_optimizations/remove_concat_zero_dim_input.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/old_api_map_element_type_attribute.hpp"
#include "transformations/rt_info/old_api_map_order_attribute.hpp"
#include "transformations/rt_info/preprocessing_attribute.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"

namespace {

bool can_erase_key(const std::string& key) {
#undef TYPE_INFO
#define TYPE_INFO(name) ov::name::get_type_info_static()
    static const std::unordered_set<std::string> rt_keys = {
            TYPE_INFO(Decompression),
            TYPE_INFO(DisableFP16Compression),
            TYPE_INFO(FusedNames),
            TYPE_INFO(LayoutAttribute),
            TYPE_INFO(NoTransposeSinkingAttr),
            TYPE_INFO(OldApiMapElementType),
            TYPE_INFO(OldApiMapOrder),
            TYPE_INFO(PreprocessingAttribute),
            TYPE_INFO(pass::DisableConstantFolding),
            TYPE_INFO(pass::DisableRemoveConcatZeroDimInput),
            TYPE_INFO(preprocess::TensorInfoMemoryType),
    };
#undef TYPE_INFO
    return rt_keys.find(key) == rt_keys.end();
}

void clear_rt_info(ov::RTMap& rtInfo) {
    for (auto it = rtInfo.cbegin(); it != rtInfo.cend();) {
        if (can_erase_key(it->first)) {
            it = rtInfo.erase(it);
        } else {
            ++it;
        }
    }
}
}  // namespace

bool ov::pass::InitNodeInfo::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(InitNodeInfo);

    for (auto& node : f->get_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                run_on_model(sub_graph);
            }
        }
        auto& rtInfo = node->get_rt_info();
        clear_rt_info(rtInfo);
        rtInfo.emplace(FusedNames::get_type_info_static(), FusedNames{node->get_friendly_name()});
    }
    return false;
}
