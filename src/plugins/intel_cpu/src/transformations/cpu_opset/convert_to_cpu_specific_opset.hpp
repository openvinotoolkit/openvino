// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/pass/align_matmul_input_ranks.hpp"
#include "common/pass/convert_matmul_to_fc.hpp"
#include "common/pass/convert_tile_to_seq_tiles.hpp"
#include "common/pass/convert_to_leaky_relu.hpp"
#include "common/pass/convert_to_power_static.hpp"
#include "common/pass/convert_to_swish_cpu.hpp"
#include "common/pass/fc_bias_fusion.hpp"
#include "common/pass/move_fc_reshape_to_weights.hpp"
#include "common/pass/move_readvalue_inputs_to_subgraph.hpp"
#include "common/pass/rnn_sequences_optimization.hpp"
#include "config.h"
#include "nodes/fullyconnected.h"
#include "nodes/gathermatmul.h"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/validate.hpp"
#include "ov_ops/fully_connected.hpp"
#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/reshape_sequence_fusion.hpp"
#include "transformations/common_optimizations/transpose_to_reshape.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/defs.hpp"
#include "transformations/op_conversions/convert_fc_to_compressed.hpp"
#include "transformations/op_conversions/convert_fc_to_quantized_legacy.hpp"
#include "transformations/op_conversions/convert_gather_matmul_to_compressed.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

namespace ov::intel_cpu {

inline bool is_topk_data_consumer(const ov::Input<ov::Node>& consumer) {
    return consumer.get_index() == 0 &&
           ov::is_type_any_of<ov::op::v1::TopK, ov::op::v3::TopK, ov::op::v11::TopK>(consumer.get_node());
}

inline bool is_integer_topk_data_source(const ov::Output<ov::Node>& data) {
    const auto precision = data.get_element_type();
    if (precision != ov::element::i64 && precision != ov::element::u64) {
        return false;
    }

    const auto source = data.get_node_shared_ptr();
    return ov::is_type_any_of<ov::op::v0::Constant, ov::op::v0::Parameter>(source) && source->get_output_size() == 1;
}

inline bool is_precision_conversion_boundary(const ov::Input<ov::Node>& consumer) {
    return ov::is_type_any_of<ov::op::v0::Convert, ov::op::v0::Result>(consumer.get_node());
}

inline bool split_non_topk_consumers_to_i32(const ov::Output<ov::Node>& data,
                                           std::unordered_set<const ov::Node*>& converted_consumer_nodes) {
    std::shared_ptr<ov::op::v0::Convert> convert;
    bool converted_consumers = false;

    for (const auto& consumer : data.get_target_inputs()) {
        if (is_topk_data_consumer(consumer) || is_precision_conversion_boundary(consumer)) {
            continue;
        }

        if (!convert) {
            convert = std::make_shared<ov::op::v0::Convert>(data, ov::element::i32);
            convert->set_friendly_name(data.get_node_shared_ptr()->get_friendly_name() + "_i32");
        }

        consumer.replace_source_output(convert);
        converted_consumer_nodes.insert(consumer.get_node());
        converted_consumers = true;
    }

    return converted_consumers;
}

inline bool has_converted_consumer_ancestor(const std::shared_ptr<ov::Node>& node,
                                            const std::unordered_set<const ov::Node*>& converted_consumer_nodes,
                                            std::unordered_set<const ov::Node*>& visited_nodes) {
    if (!node || !visited_nodes.insert(node.get()).second) {
        return false;
    }

    if (converted_consumer_nodes.count(node.get()) > 0) {
        return true;
    }

    for (const auto& input : node->inputs()) {
        if (has_converted_consumer_ancestor(input.get_source_output().get_node_shared_ptr(),
                                            converted_consumer_nodes,
                                            visited_nodes)) {
            return true;
        }
    }

    return false;
}

inline bool restore_result_precisions(const std::shared_ptr<ov::Model>& model,
                                      const std::vector<ov::element::Type>& result_precisions,
                                      const std::unordered_set<const ov::Node*>& converted_consumer_nodes) {
    bool restored_results = false;
    const auto& results = model->get_results();

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        std::unordered_set<const ov::Node*> visited_nodes;
        if (!has_converted_consumer_ancestor(result->input_value(0).get_node_shared_ptr(),
                                             converted_consumer_nodes,
                                             visited_nodes)) {
            continue;
        }

        const auto result_input = result->input_value(0);
        if (result_input.get_element_type() == result_precisions[i] &&
            ov::is_type<ov::op::v0::Convert>(result_input.get_node_shared_ptr())) {
            continue;
        }

        const auto convert = std::make_shared<ov::op::v0::Convert>(result_input, result_precisions[i]);
        convert->set_friendly_name(result_input.get_node_shared_ptr()->get_friendly_name() + "." +
                                   std::to_string(result_input.get_index()));
        result->input(0).replace_source_output(convert);
        result->revalidate_and_infer_types();
        restored_results = true;
    }

    return restored_results;
}

inline bool keep_integer_topk_data_precision(const std::shared_ptr<ov::Model>& model) {
    const auto& results = model->get_results();
    std::vector<ov::element::Type> result_precisions;
    result_precisions.reserve(results.size());
    for (const auto& result : results) {
        result_precisions.push_back(result->get_input_element_type(0));
    }

    bool changed = false;
    bool converted_consumers = false;
    std::unordered_set<const ov::Node*> converted_consumer_nodes;
    ov::traverse_nodes(model, [&](const std::shared_ptr<ov::Node>& node) {
        if (!ov::is_type_any_of<ov::op::v1::TopK, ov::op::v3::TopK, ov::op::v11::TopK>(node)) {
            return;
        }

        const auto data = node->input_value(0);
        if (!is_integer_topk_data_source(data)) {
            return;
        }

        converted_consumers = split_non_topk_consumers_to_i32(data, converted_consumer_nodes) || converted_consumers;
        ov::enable_keep_const_precision(data.get_node_shared_ptr());
        changed = true;
    });

    if (converted_consumers) {
        changed = restore_result_precisions(model, result_precisions, converted_consumer_nodes) || changed;
    }

    return changed;
}

class KeepIntegerTopKDataPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("KeepIntegerTopKDataPrecision");
    KeepIntegerTopKDataPrecision() = default;

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        return keep_integer_topk_data_precision(model);
    }
};

inline void ConvertToCPUSpecificOpset(std::shared_ptr<ov::Model>& model, const Config& config) {
    RUN_ON_FUNCTION_SCOPE(ConvertToCPUSpecificOpset);

    ov::pass::Manager manager("CPU:ConvertToCPUSpecificOpset");
    manager.set_per_pass_validation(false);

    // TransformMoeBlockToGatherMatmuls
    CPU_REGISTER_PASS_X64(manager, ov::pass::ConvertTiledMoeBlockToGatherMatmuls);
    CPU_REGISTER_PASS_X64(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_X64(
        manager,
        ov::pass::ConvertGatherMatmulToGatherMatmulCompressed,
        ov::intel_cpu::node::GatherMatmul::getSupportedCompressedActivationsTypes(),
        ov::intel_cpu::node::GatherMatmul::getSupportedCompressedWeightsTypes(),
        [&](const std::shared_ptr<ov::op::internal::GatherMatmulCompressed>& gather_matmul,
            size_t IC,
            size_t OC,
            size_t G) {
            return ov::intel_cpu::node::GatherMatmul::isSupportedCompressedOperation(gather_matmul, IC, OC, G, config);
        });

    CPU_REGISTER_PASS_COMMON(manager, ConvertMatMulToFC);
    CPU_REGISTER_PASS_COMMON(manager, FullyConnectedBiasFusion);

    CPU_REGISTER_PASS_COMMON(
        manager,
        pass::ConvertFullyConnectedToFullyConnectedCompressed,
        ov::intel_cpu::node::FullyConnected::getSupportedCompressedActivationsTypes(),
        ov::intel_cpu::node::FullyConnected::getSupportedCompressedWeightsTypes(),
        [&config](const std::shared_ptr<ov::op::internal::FullyConnected>& fc, size_t IC, size_t OC, size_t G) {
            return ov::intel_cpu::node::FullyConnected::isSupportedCompressedOperation(fc, IC, OC, G, config);
        });

    CPU_REGISTER_PASS_X64(manager, pass::ConvertFCToFCQuantizedLegacy);
    CPU_REGISTER_PASS_COMMON(manager, MoveFCReshapeToWeights);
    CPU_REGISTER_PASS_COMMON(manager, AlignMatMulInputRanks);
    CPU_REGISTER_PASS_COMMON(manager, ConvertTileToSeqTiles);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToPowerStatic);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToLeakyRelu);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToSwishCPU);
    CPU_REGISTER_PASS_COMMON(manager, OptimizeSequenceTransposes);
    // TransposeToReshape is also registered in MOC, but plugin-specific transformations
    // can introduce new Transpose nodes after MOC runs.
    CPU_REGISTER_PASS_COMMON(
        manager,
        ov::pass::TransposeToReshape);  // Should be after all transformations that can produce transposes
    CPU_REGISTER_PASS_COMMON(
        manager,
        ov::pass::ReshapeSequenceFusion);  // after transformation "MoveEltwiseUpThroughDataMov" there can be reshaped
                                           // sequences that should be eliminated or fused
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConstantFolding);
    CPU_REGISTER_PASS_COMMON(manager, KeepIntegerTopKDataPrecision);
    CPU_REGISTER_PASS_COMMON(manager,
                             ov::pass::ConvertPrecision,
                             precisions_map{{ov::element::i64, ov::element::i32}},
                             type_to_fuse_map{{}},
                             false,
                             false);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::EliminateConvert);  // Need to clean up after the ConvertPrecision.
    CPU_REGISTER_PASS_COMMON(manager, MoveReadValueInputsToSubgraph);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);

    manager.run_passes(model);
}

}  // namespace ov::intel_cpu
