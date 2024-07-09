// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/manager.hpp"
#include "common/pass/align_matmul_input_ranks.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/reshape_prelu.hpp"
#include "common/pass/convert_broadcast_to_tiles.hpp"
#include "common/pass/convert_tile_to_seq_tiles.hpp"
#include "common/pass/convert_matmul_to_fc.hpp"
#include "common/pass/convert_fc_to_compressed_fc.hpp"
#include "common/pass/convert_to_power_static.hpp"
#include "common/pass/convert_to_leaky_relu.hpp"
#include "common/pass/convert_to_swish_cpu.hpp"
#include "common/pass/move_fc_reshape_to_weights.hpp"
#include "common/pass/split_fc.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/cpu_opset/common/op/submodel.hpp"
#include "transformations/cpu_opset/common/pass/duplicate_model.hpp"
#include "transformations/cpu_opset/common/pass/move_convert_through_variadic_split.hpp"
#include "transformations/utils/utils.hpp"
#include "common/pass/rnn_sequences_optimization.hpp"
#include "transformations/common_optimizations/reshape_sequence_fusion.hpp"
#include "transformations/defs.hpp"

#include "itt.hpp"

namespace ov {
namespace intel_cpu {

static std::string extraDump() {
    static auto env = std::getenv("EXTRA_DUMP");
    static std::string result = env ? std::string(env) : std::string{};

    return result;
}

inline void ConvertToCPUSpecificOpset(std::shared_ptr<ov::Model> &nGraphFunc, int subStreamNum) {
    RUN_ON_FUNCTION_SCOPE(ConvertToCPUSpecificOpset);

    // std::cout << "ConvertToCPUSpecificOpset: " << nGraphFunc->get_friendly_name() << "\n";

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    CPU_REGISTER_PASS_COMMON(manager, ConvertMatMulToFC);
    CPU_REGISTER_PASS_COMMON(manager, ConvertFullyConnectedToFullyConnectedCompressed);
    CPU_REGISTER_PASS_X64(manager, MoveFCReshapeToWeights);
    CPU_REGISTER_PASS_X64(manager, ov::pass::Validate);

    if ((subStreamNum >= 1 || std::getenv("FORCE_SPLIT")) && !std::getenv("DISABLE_SPLIT")) {
        subStreamNum = 1;

        CPU_REGISTER_PASS_COMMON(manager, SplitFC, subStreamNum);
        if (std::getenv("EXTRA_DUMP")) {
            manager.run_passes(nGraphFunc);
            std::cout << "### Dumping graph after SplitFC" << "\n";
            ov::pass::Serialize("after_split.xml", "/dev/null").run_on_model(nGraphFunc);
            CPU_DISABLE_PASS_COMMON(manager, SplitFC);
        }

        // else {
        //     CPU_REGISTER_PASS_COMMON(manager, SplitFCbyK, subStreamNum);
        //     if (std::getenv("EXTRA_DUMP")) {
        //         manager.run_passes(nGraphFunc);
        //         std::cout << "### Dumping graph after SplitFCbyK" << "\n";
        //         ov::pass::Serialize("after_split.xml", "/dev/null").run_on_model(nGraphFunc);
        //         CPU_DISABLE_PASS_COMMON(manager, SplitFCbyK);
        //     }
        // }

        CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);

        if (!std::getenv("DISABLE_DUP") && !std::getenv("ENABLE_SUBGRAPH")) {
            CPU_REGISTER_PASS_COMMON(manager, DuplicateModel);
            // manager.register_pass<pass::PrintModel>("print_model.txt");
            CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);

            if (std::getenv("EXTRA_DUMP")) {
                manager.run_passes(nGraphFunc);
                std::cout << "### Dumping graph after DuplicateModel" << "\n";
                ov::pass::Serialize("after_duplicate.xml", "/dev/null").run_on_model(nGraphFunc);
                CPU_DISABLE_PASS_COMMON(manager, DuplicateModel);
            }
        }

        if (!extraDump().empty()) {
            for (const auto& op : nGraphFunc->get_ordered_ops()) {
                if (const auto submodel = ov::as_type_ptr<SubModel>(op)) {
                    if (extraDump() == "all" || extraDump() == submodel->get_friendly_name())
                        ov::pass::Serialize(submodel->get_friendly_name() + ".xml", "/dev/null").run_on_model(submodel->get_function());
                }
            }
        }

        // if (std::getenv("ENABLE_SUBGRAPH")) {
        //     CPU_REGISTER_PASS_COMMON(manager, FormParallelSubgraphs);

        //     if (std::getenv("EXTRA_DUMP")) {
        //         manager.run_passes(nGraphFunc);
        //         ov::pass::Serialize("after_form.xml", "/dev/null").run_on_model(nGraphFunc);
        //         CPU_DISABLE_PASS_COMMON(manager, FormParallelSubgraphs);
        //     }
        // }

        CPU_REGISTER_PASS_COMMON(manager, MoveConvertThroughVariadicSplit);
        CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    }
    CPU_REGISTER_PASS_COMMON(manager, AlignMatMulInputRanks);
    CPU_REGISTER_PASS_COMMON(manager, ConvertTileToSeqTiles);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToPowerStatic);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToLeakyRelu);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToSwishCPU);
    CPU_REGISTER_PASS_COMMON(manager, OptimizeSequenceTransposes);
    // after transformation "MoveEltwiseUpThroughDataMov" there can be reshaped sequences that should be eliminated or fused
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ReshapeSequenceFusion);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConstantFolding);
    CPU_REGISTER_PASS_COMMON(manager,
                             ov::pass::ConvertPrecision,
                             precisions_map{{ov::element::i64, ov::element::i32}},
                             type_to_fuse_map{{}},
                             false,
                             false);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::EliminateConvert); // Need to clean up after the ConvertPrecision.

    manager.run_passes(nGraphFunc);
}

}   // namespace intel_cpu
}   // namespace ov
