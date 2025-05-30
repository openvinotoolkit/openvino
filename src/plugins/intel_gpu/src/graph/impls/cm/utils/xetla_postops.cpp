// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xetla_postops.hpp"

namespace ov::intel_gpu::cm {

std::vector<std::tuple<std::string, std::string>> generate_post_ops(const std::vector<std::unique_ptr<XeTLAPostOP>>& post_ops) {
    std::string post_op_kernel_args = "";
    std::string post_op_args = "";
    std::string post_op_args_pass = "";
    std::string post_op_definitions = "";
    std::string post_op_list = "";
    std::string post_op_shape_definitions = "";
    std::string post_op_epilogue_init_args = "";

    bool first_epilogue_arg = true;
    bool first_post_op_list = true;

    for (const auto& post_op : post_ops) {
        auto kernel_arg_definition = post_op->get_kernel_arg_definition();
        if (!kernel_arg_definition.empty()) {
            post_op_kernel_args += ", " + kernel_arg_definition;
        }

        auto arg_definition = post_op->get_arg_definition();
        if (!arg_definition.empty()) {
            post_op_args += ", " + arg_definition;
        }

        auto arg_definition_pass = post_op->get_arg_name();
        if (!arg_definition_pass.empty()) {
            post_op_args_pass += ", " + arg_definition_pass;
        }

        post_op_definitions += post_op->get_definition();

        if (first_post_op_list) {
            post_op_list += post_op->get_definition_name();
            first_post_op_list = false;
        } else {
            post_op_list += ", " + post_op->get_definition_name();
        }

        post_op_shape_definitions += post_op->get_shape_definition();

        if (first_epilogue_arg) {
            post_op_epilogue_init_args += post_op->get_epilogue_init();
            first_epilogue_arg = false;
        } else {
            post_op_epilogue_init_args += ", " + post_op->get_epilogue_init();
        }
    }
    std::vector<std::tuple<std::string, std::string>> definitions;
    definitions.push_back({"LORA_POST_OP_KERNEL_ARGS", post_op_kernel_args});
    definitions.push_back({"LORA_POST_OP_ARGS", post_op_args});
    definitions.push_back({"LORA_POST_OP_ARGS_PASS", post_op_args_pass});
    definitions.push_back({"LORA_POST_OP_DEFINITIONS", post_op_definitions});
    definitions.push_back({"LORA_POST_OP_LIST", post_op_list});
    definitions.push_back({"LORA_POST_OP_SHAPE_DEFINITIONS", post_op_shape_definitions});
    definitions.push_back({"LORA_POST_OP_EPILOGUE_INIT_ARGS", post_op_epilogue_init_args});

    return definitions;
}

}  // namespace ov::intel_gpu::cm
