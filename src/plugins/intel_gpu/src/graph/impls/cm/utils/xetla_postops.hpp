// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "openvino/core/except.hpp"

namespace ov::intel_gpu::cm {

class XeTLAPostOP {
protected:
    const size_t index;
    const std::string dtype;

public:
    XeTLAPostOP(size_t index, std::string dtype) : index{index}, dtype{dtype} {};
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
    ScaleChannels(size_t index, std::string dtype) : XeTLAPostOP(index, dtype) {}
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
    ShiftChannels(size_t index, std::string dtype) : XeTLAPostOP(index, dtype) {}
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
    Eltwise(size_t index, std::string dtype, EltwiseOp op) : XeTLAPostOP(index, dtype), op{op} {}
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
            OPENVINO_THROW("Unknown XeTLA EltwiseOp");
        }
    }
};

class Activation : public XeTLAPostOP {
public:
    enum class ActivationOp { none, ReLU, Tanh, Sigmoid, SiLU, GeLU };

private:
    ActivationOp op;

public:
    Activation(size_t index, std::string dtype, ActivationOp op) : XeTLAPostOP(index, dtype), op{op} {}
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
            OPENVINO_THROW("Unknown XeTLA ActivationOp");
        }
    }
};

class XeTLAPostOPs {
    size_t post_op_index = 0;
    std::vector<std::unique_ptr<XeTLAPostOP>> postops;

public:
    template <typename T, typename... Args>
    void add_post_op(Args&&... args) {
        postops.push_back(std::make_unique<T>(post_op_index++, std::forward<Args>(args)...));
    }

    size_t add_post_ops(const RuntimeParams& params, size_t post_op_arg_index);
    std::vector<std::tuple<std::string, std::string>> get_definitions();
};

std::vector<std::tuple<std::string, std::string>> generate_post_ops(const std::vector<std::unique_ptr<XeTLAPostOP>>& post_ops);

inline Activation::ActivationOp get_xetla_activation_op(cldnn::activation_func func) {
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

inline Eltwise::EltwiseOp get_xetla_eltwise_op(cldnn::eltwise_mode mode) {
    switch (mode) {
    case cldnn::eltwise_mode::sum:
        return Eltwise::EltwiseOp::sum;
    case cldnn::eltwise_mode::prod:
        return Eltwise::EltwiseOp::prod;
    default:
        return Eltwise::EltwiseOp::none;
    }
}

}  // namespace ov::intel_gpu::cm
