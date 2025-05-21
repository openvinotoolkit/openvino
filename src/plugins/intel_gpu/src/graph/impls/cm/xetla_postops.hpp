// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ov::intel_gpu::cm {
    namespace {
class XeTLAPostOP {
protected:
    const uint32_t index;
    const std::string dtype;

public:
    XeTLAPostOP(uint32_t index, std::string dtype) : index{index}, dtype{dtype} {};
    virtual ~XeTLAPostOP() = default;
    virtual std::string get_arg_name() const = 0;
    virtual std::string get_arg_definition() const {
        return dtype + " *" + get_arg_name();
    }
    virtual std::string get_kernel_arg_definition() const {
        return get_arg_definition() + " [[type(\"svmptr_t\")]]";
    }
    virtual std::string get_definition_name() const = 0;
    virtual std::string get_definition() const = 0;
    virtual std::string get_shape_name() const {
        return get_arg_name() + "_shape";
    }
    virtual std::string get_shape_definition() const = 0;
    virtual std::string get_epilogue_init() const {
        return "{" + get_arg_name() + ", " + get_shape_name() + "}";
    }
};

class ScaleChannels : public XeTLAPostOP {
public:
    ScaleChannels(uint32_t index, std::string dtype) : XeTLAPostOP(index, dtype) {}
    virtual std::string get_arg_name() const {
        return "scale_input" + std::to_string(index);
    }
    virtual std::string get_definition_name() const {
        return "scale_op_t" + std::to_string(index);
    }
    virtual std::string get_definition() const {
        return "using " + get_definition_name() + " = subgroup::scale_v_op_t<" + dtype + ", arch_tag>;";
    }
    virtual std::string get_shape_definition() const {
        return "typename " + get_definition_name() + "::scale_shape_t " + get_shape_name() + "(mat_n, 1, mat_n);";
    }
};

class ShiftChannels : public XeTLAPostOP {
public:
    ShiftChannels(uint32_t index, std::string dtype) : XeTLAPostOP(index, dtype) {}
    virtual std::string get_arg_name() const {
        return "shift_input" + std::to_string(index);
    }
    virtual std::string get_definition_name() const {
        return "shift_op_t" + std::to_string(index);
    }
    virtual std::string get_definition() const {
        return "using " + get_definition_name() + " = subgroup::bias_add_op_t<" + dtype + ", arch_tag>;";
    }
    virtual std::string get_shape_definition() const {
        return "typename " + get_definition_name() + "::shape_t " + get_shape_name() + "(mat_n, 1, mat_n);";
    }
};

class Eltwise : public XeTLAPostOP {
public:
    enum class EltwiseOp { none, sum, prod };

private:
    EltwiseOp op;

public:
    Eltwise(uint32_t index, std::string dtype, EltwiseOp op) : XeTLAPostOP(index, dtype), op{op} {}
    virtual std::string get_arg_name() const {
        return "eltwise_input" + std::to_string(index);
    }
    virtual std::string get_definition_name() const {
        return "eltwise_op_t" + std::to_string(index);
    }
    virtual std::string get_definition() const {
        return "using " + get_definition_name() + " = subgroup::elemwise_reduce_op_t<reduce_op::" + get_op_name() + ", " + dtype + ", arch_tag>;";
    }
    virtual std::string get_shape_definition() const {
        return "typename " + get_definition_name() + "::shape_t " + get_shape_name() + "(mat_n, mat_m, ldc);";
    }
    std::string get_op_name() const {
        switch (op) {
        case EltwiseOp::sum:
            return "sum";
        case EltwiseOp::prod:
            return "prod";
        default:
            throw std::runtime_error("Unknown EltwiseOp");
        }
    }
};

class Activation : public XeTLAPostOP {
public:
    enum class ActivationOp { none, ReLU, Tanh, Sigmoid, SiLU, GeLU };

private:
    ActivationOp op;

public:
    Activation(uint32_t index, std::string dtype, ActivationOp op) : XeTLAPostOP(index, dtype), op{op} {}
    virtual std::string get_arg_name() const {
        return "";
    }
    virtual std::string get_arg_definition() const {
        return "";
    }
    virtual std::string get_kernel_arg_definition() const {
        return "";
    }
    virtual std::string get_definition_name() const {
        return "activation_op_t" + std::to_string(index);
    }
    virtual std::string get_definition() const {
        return "using " + get_definition_name() + " = subgroup::" + get_op_name() + ";";
    }
    virtual std::string get_shape_definition() const {
        return "";
    }
    virtual std::string get_shape_name() const {
        return "";
    }
    virtual std::string get_epilogue_init() const {
        return "{}";
    }
    std::string get_op_name() const {
        switch (op) {
        case ActivationOp::ReLU:
            return "relu_op_t";
        case ActivationOp::Tanh:
            return "tanh_op_t";
        case ActivationOp::Sigmoid:
            return "sigmoid_op_t";
        case ActivationOp::SiLU:
            return "silu_precise_op_t";
        case ActivationOp::GeLU:
            return "gelu_fwd_op_t";
        default:
            throw std::runtime_error("Unknown ActivationOp");
        }
    }
};

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

Activation::ActivationOp get_xetla_activation_op(cldnn::activation_func func) {
    switch (func) {
    case cldnn::activation_func::relu:
        return Activation::ActivationOp::ReLU;
    case cldnn::activation_func::tan:
        return Activation::ActivationOp::Tanh;
    case cldnn::activation_func::swish:
        return Activation::ActivationOp::SiLU;
    case cldnn::activation_func::gelu:
        return Activation::ActivationOp::GeLU;
    default:
        return Activation::ActivationOp::none;
    }
}

Eltwise::EltwiseOp get_xetla_eltwise_op(cldnn::eltwise_mode mode) {
    switch (mode) {
    case cldnn::eltwise_mode::sum:
        return Eltwise::EltwiseOp::sum;
    case cldnn::eltwise_mode::prod:
        return Eltwise::EltwiseOp::prod;
    default:
        return Eltwise::EltwiseOp::none;
    }
}
    }
}  // namespace ov::intel_gpu::cm
