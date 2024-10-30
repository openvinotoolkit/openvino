// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_inst.h"
#include "impls/registry/implementation_map.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_left_shift.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/is_finite.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/xor.hpp"
#include "register.hpp"

namespace cldnn {
namespace cpu {

struct eltwise_impl : public typed_primitive_impl<eltwise> {
    using parent = typed_primitive_impl<eltwise>;
    using parent::parent;

    eltwise_mode mode = eltwise_mode::sum;
    std::vector<float> coefficients;

    std::shared_ptr<ov::op::Op> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::eltwise_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<eltwise_impl>(*this);
    }

    eltwise_impl() : parent("eltwise_cpu_impl") {}

    explicit eltwise_impl(const eltwise_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<eltwise>(), "[GPU] Incorrect program_node type");
        const auto& node = arg.as<eltwise>();
        mode = node.get_primitive()->mode;
        coefficients = node.get_primitive()->coefficients;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << make_data(&mode, sizeof(eltwise_mode));
        ob << coefficients;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> make_data(&mode, sizeof(eltwise_mode));
        ib >> coefficients;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, eltwise_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "eltwise::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }

        auto params = instance.get_impl_params();

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        if (!op) {
            switch (mode) {
            case eltwise_mode::sum:
                op = std::make_shared<ov::op::v1::Add>();
                break;
            case eltwise_mode::prod:
                op = std::make_shared<ov::op::v1::Multiply>();
                break;
            case eltwise_mode::max:
                op = std::make_shared<ov::op::v1::Maximum>();
                break;
            case eltwise_mode::min:
                op = std::make_shared<ov::op::v1::Minimum>();
                break;
            case eltwise_mode::sub:
                op = std::make_shared<ov::op::v1::Subtract>();
                break;
            case eltwise_mode::div:
                op = std::make_shared<ov::op::v1::Divide>();
                break;
            case eltwise_mode::squared_diff:
                op = std::make_shared<ov::op::v0::SquaredDifference>();
                break;
            case eltwise_mode::eq:
                op = std::make_shared<ov::op::v1::Equal>();
                break;
            case eltwise_mode::ne:
                op = std::make_shared<ov::op::v1::NotEqual>();
                break;
            case eltwise_mode::lt:
                op = std::make_shared<ov::op::v1::Less>();
                break;
            case eltwise_mode::le:
                op = std::make_shared<ov::op::v1::LessEqual>();
                break;
            case eltwise_mode::gt:
                op = std::make_shared<ov::op::v1::Greater>();
                break;
            case eltwise_mode::ge:
                op = std::make_shared<ov::op::v1::GreaterEqual>();
                break;
            case eltwise_mode::logic_and:
                op = std::make_shared<ov::op::v1::LogicalAnd>();
                break;
            case eltwise_mode::logic_or:
                op = std::make_shared<ov::op::v1::LogicalOr>();
                break;
            case eltwise_mode::logic_xor:
                op = std::make_shared<ov::op::v1::LogicalXor>();
                break;
            case eltwise_mode::pow:
                op = std::make_shared<ov::op::v1::Power>();
                break;
            case eltwise_mode::floor_mod:
                op = std::make_shared<ov::op::v1::FloorMod>();
                break;
            case eltwise_mode::mod:
                op = std::make_shared<ov::op::v1::Mod>();
                break;
            case eltwise_mode::is_finite:
                op = std::make_shared<ov::op::v10::IsFinite>();
                break;
            case eltwise_mode::is_inf: {
                auto is_inf_op = std::make_shared<ov::op::v10::IsInf>();

                OPENVINO_ASSERT(coefficients.size() == 2, "[GPU] Incorrect configuration of eltwise is_inf_op operation. ",
                                                          "Expected number of coefficients is 2, but got ", coefficients.size());

                is_inf_op->set_attributes({ static_cast<bool>(coefficients[0]),
                                            static_cast<bool>(coefficients[1]) });
                op = is_inf_op;
                break;
            }
            case eltwise_mode::is_nan:
                op = std::make_shared<ov::op::v10::IsNaN>();
                break;
            case eltwise_mode::right_shift:
                op = std::make_shared<ov::op::v15::BitwiseRightShift>();
                break;
            case eltwise_mode::left_shift:
                op = std::make_shared<ov::op::v15::BitwiseLeftShift>();
                break;
            case eltwise_mode::bitwise_and:
                op = std::make_shared<ov::op::v13::BitwiseAnd>();
                break;
            case eltwise_mode::bitwise_or:
                op = std::make_shared<ov::op::v13::BitwiseOr>();
                break;
            case eltwise_mode::bitwise_xor:
                op = std::make_shared<ov::op::v13::BitwiseXor>();
                break;
            default:
                OPENVINO_THROW("[GPU] Couldn't create eltwise operation: unsupported eltwise operation (", static_cast<size_t>(mode), ")");
            }
        }

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); i++)
            input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<uint8_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_host_tensors.push_back(make_tensor(params->input_layouts[i], input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

        output_host_tensors.push_back(make_tensor(params->output_layouts[0], output_lock.data()));

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute eltwise primitive with id ", instance.id());

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_mem_ptrs[i]->unlock(stream);

        if (pass_through_events) {
            if (events.size() > 1) {
                return stream.group_events(events);
            } else if (events.size() == 1) {
                return events[0];
            }
        }

        return stream.create_user_event(true);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const eltwise_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<eltwise_impl>();
    }
};


namespace detail {

attach_eltwise_impl::attach_eltwise_impl() {
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

    implementation_map<eltwise>::add(impl_types::cpu, shape_types::static_shape, eltwise_impl::create, types, formats);
    implementation_map<eltwise>::add(impl_types::cpu, shape_types::dynamic_shape, eltwise_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::eltwise_impl)
