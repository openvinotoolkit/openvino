// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/cpu/cpu_impl_helpers.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "register.hpp"
#include "activation_inst.h"
#include "registry/implementation_map.hpp"

#include "openvino/op/power.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/selu.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/tan.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/sqrt.hpp"

namespace cldnn {
namespace cpu {

struct activation_impl : public typed_primitive_impl<activation> {
    using parent = typed_primitive_impl<activation>;
    using parent::parent;

    activation_func activation_function = activation_func::none;
    activation_additional_params additional_params = {0.f, 0.f};

    std::shared_ptr<ov::op::Op> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::activation_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<activation_impl>(*this);
    }

    activation_impl() : parent("activation_cpu_impl") {}

    explicit activation_impl(const activation_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<activation>(), "[GPU] Incorrect program_node type");
        const auto& node = arg.as<activation>();
        activation_function = node.get_primitive()->activation_function;
        additional_params = node.get_primitive()->additional_params;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << make_data(&activation_function, sizeof(activation_func));
        ob << make_data(&additional_params, sizeof(activation_additional_params));
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> make_data(&activation_function, sizeof(activation_func));
        ib >> make_data(&additional_params, sizeof(activation_additional_params));
    }

    template <data_types DT>
    static void execute_activation(std::shared_ptr<ov::op::Op> op,
                                   const activation_inst& instance,
                                   const activation_func& activation_function,
                                   const activation_additional_params& additional_params) {
        auto& stream = instance.get_network().get_stream();

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        auto params = instance.get_impl_params();

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); i++)
            input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

        // TODO: consider to re-implement lock/unlock in more exception-safetest way
        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_host_tensors.push_back(make_tensor(params->input_layouts[i], input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

        // Most of the evaluate functions expect same data type for all inputs, so we need to convert params from float
        auto param_a = static_cast<typename ov::element_type_traits<DT>::value_type>(additional_params.a);

        auto input_dt = instance.get_input_layout().data_type;

        if (activation_function == activation_func::pow) {
            input_host_tensors.push_back(ov::Tensor(input_dt, {}, &param_a));
        } else if (activation_function == activation_func::relu_negative_slope) {
            if (input_host_tensors.size() < 2) {
                input_host_tensors.push_back(ov::Tensor(input_dt, {}, &param_a));
            }
        } else if (activation_function == activation_func::swish) {
            if (additional_params.a != 1.0f)
                input_host_tensors.push_back(ov::Tensor(input_dt, {}, &param_a));
        }

        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<uint8_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);
        output_host_tensors.push_back(make_tensor(params->output_layouts[0], output_lock.data()));

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute activation primitive with id ", instance.id());

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_mem_ptrs[i]->unlock(stream);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, activation_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "activation::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        if (!pass_through_events) {
            stream.wait_for_events(events);
        }

        if (!op) {
            switch (activation_function) {
            case activation_func::pow:
                op = std::make_shared<ov::op::v1::Power>(); break;
            case activation_func::hyperbolic_tan:
                op = std::make_shared<ov::op::v0::Tanh>(); break;
            case activation_func::elu:
                op = std::make_shared<ov::op::v0::Elu>(); break;
            case activation_func::logistic:
                op = std::make_shared<ov::op::v0::Sigmoid>(); break;
            case activation_func::relu:
                op = std::make_shared<ov::op::v0::Relu>(); break;
            case activation_func::relu_negative_slope:
                op = std::make_shared<ov::op::v0::PRelu>(); break;
            case activation_func::clamp: {
                auto clamp_op = std::make_shared<ov::op::v0::Clamp>();
                clamp_op->set_min(additional_params.a);
                clamp_op->set_max(additional_params.b);
                op = clamp_op;
                break;
            }
            case activation_func::exp:
                op = std::make_shared<ov::op::v0::Exp>(); break;
            case activation_func::negation:
                op = std::make_shared<ov::op::v1::LogicalNot>(); break;
            case activation_func::asin:
                op = std::make_shared<ov::op::v0::Asin>(); break;
            case activation_func::asinh:
                op = std::make_shared<ov::op::v3::Asinh>(); break;
            case activation_func::acos:
                op = std::make_shared<ov::op::v0::Acos>(); break;
            case activation_func::acosh:
                op = std::make_shared<ov::op::v3::Acosh>(); break;
            case activation_func::atan:
                op = std::make_shared<ov::op::v0::Atan>(); break;
            case activation_func::atanh:
                op = std::make_shared<ov::op::v3::Atanh>(); break;
            case activation_func::abs:
                op = std::make_shared<ov::op::v0::Abs>(); break;
            case activation_func::floor:
                op = std::make_shared<ov::op::v0::Floor>(); break;
            case activation_func::ceil:
                op = std::make_shared<ov::op::v0::Ceiling>(); break;
            case activation_func::erf:
                op = std::make_shared<ov::op::v0::Erf>(); break;
            case activation_func::log:
                op = std::make_shared<ov::op::v0::Log>(); break;
            case activation_func::negative:
                op = std::make_shared<ov::op::v0::Negative>(); break;
            case activation_func::softplus:
                op = std::make_shared<ov::op::v4::SoftPlus>(); break;
            case activation_func::tan:
                op = std::make_shared<ov::op::v0::Tan>(); break;
            case activation_func::sin:
                op = std::make_shared<ov::op::v0::Sin>(); break;
            case activation_func::sinh:
                op = std::make_shared<ov::op::v0::Sinh>(); break;
            case activation_func::cos:
                op = std::make_shared<ov::op::v0::Cos>(); break;
            case activation_func::cosh:
                op = std::make_shared<ov::op::v0::Cosh>(); break;
            case activation_func::swish:
                op = std::make_shared<ov::op::v4::Swish>(); break;
            case activation_func::hswish:
                op = std::make_shared<ov::op::v4::HSwish>(); break;
            case activation_func::mish:
                op = std::make_shared<ov::op::v4::Mish>(); break;
            case activation_func::gelu:
            case activation_func::gelu_tanh: {
                auto gelu_op = std::make_shared<ov::op::v7::Gelu>();
                auto approximation_mode =
                    activation_function == cldnn::activation_func::gelu ? ov::op::GeluApproximationMode::ERF
                                                                        : ov::op::GeluApproximationMode::TANH;
                gelu_op->set_approximation_mode(approximation_mode);
                op = gelu_op;
                break;
            }
            case activation_func::sign:
                op = std::make_shared<ov::op::v0::Sign>(); break;
            case activation_func::hsigmoid:
                op = std::make_shared<ov::op::v5::HSigmoid>(); break;
            case activation_func::round_half_to_even:
            case activation_func::round_half_away_from_zero: {
                auto round_op = std::make_shared<ov::op::v5::Round>();
                auto round_mode =
                    activation_function == cldnn::activation_func::round_half_to_even ? ov::op::v5::Round::RoundMode::HALF_TO_EVEN
                                                                                      : ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO;
                round_op->set_mode(round_mode);
                op = round_op;
                break;
            }
            case activation_func::sqrt:
                op = std::make_shared<ov::op::v0::Sqrt>(); break;
            case activation_func::hard_sigmoid:
            case activation_func::selu:
            default:
                OPENVINO_THROW("[GPU] Couldn't create activation operation: unsupported activation type ",
                               "(", static_cast<size_t>(activation_function), ") for primitive with id ", instance.id());
            }
        }

        auto params = instance.get_impl_params();

        switch (params->input_layouts[0].data_type) {
        case data_types::f32:
            execute_activation<data_types::f32>(op, instance, activation_function, additional_params);
            break;
        case data_types::f16:
            execute_activation<data_types::f16>(op, instance, activation_function, additional_params);
            break;
        case data_types::i64:
            execute_activation<data_types::i64>(op, instance, activation_function, additional_params);
            break;
        case data_types::i32:
            execute_activation<data_types::i32>(op, instance, activation_function, additional_params);
            break;
        case data_types::i8:
            execute_activation<data_types::i8>(op, instance, activation_function, additional_params);
            break;
        case data_types::u8:
            execute_activation<data_types::u8>(op, instance, activation_function, additional_params);
            break;
        default:
            OPENVINO_THROW("[GPU] Couldn't execute activation operation: unsupported input data type: ",
                           params->input_layouts[0].data_type);
        }

        if (pass_through_events) {
            return stream.group_events(events);
        }

        return make_output_event(stream, instance.is_output());
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const activation_node& arg, const kernel_impl_params& impl_param) {
        return std::make_unique<activation_impl>();
    }
};


namespace detail {

attach_activation_impl::attach_activation_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
        format::bfuwzyx,
        format::bfvuwzyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i64,
        data_types::i8,
        data_types::u8,
    };

    implementation_map<activation>::add(impl_types::cpu, shape_types::static_shape, activation_impl::create, types, formats);
    implementation_map<activation>::add(impl_types::cpu, shape_types::dynamic_shape, activation_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::activation_impl)
