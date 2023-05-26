// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "eltwise_inst.h"
#include "implementation_map.hpp"

#include "intel_gpu/runtime/error_handler.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/squared_difference.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/floor_mod.hpp"
#include "ngraph/op/mod.hpp"
#include "ngraph/op/is_finite.hpp"
#include "ngraph/op/is_inf.hpp"
#include "ngraph/op/is_nan.hpp"

namespace cldnn {
namespace cpu {

struct eltwise_impl : public typed_primitive_impl<eltwise> {
    using parent = typed_primitive_impl<eltwise>;
    using parent::parent;

    eltwise_mode mode;
    std::vector<float> coefficients;

    std::shared_ptr<ov::op::Op> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<eltwise_impl>(*this);
    }

    eltwise_impl() : parent("eltwise_cpu_impl") {}

    explicit eltwise_impl(const eltwise_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<eltwise>());
        const auto& node = arg.as<eltwise>();
        mode = node.get_primitive()->mode;
        coefficients = node.get_primitive()->coefficients;
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob << make_data(&mode, sizeof(eltwise_mode));
        ob << coefficients;
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> make_data(&mode, sizeof(eltwise_mode));
        ib >> coefficients;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, eltwise_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "eltwise::execute_impl");
        auto& stream = instance.get_network().get_stream();

        for (auto e : events) {
            e->wait();
        }

        auto ev = stream.create_user_event(false);

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
            input_host_tensors.push_back(make_tensor(input_mem_ptrs[i]->get_layout(), input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

        output_host_tensors.push_back(make_tensor(output_mem_ptr->get_layout(), output_lock.data()));

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute eltwise primitive with id ", instance.id());

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_mem_ptrs[i]->unlock(stream);

        ev->set();

        return ev;
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update_dispatch_data(const kernel_impl_params& impl_param) override {}

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
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
        data_types::i32,
        data_types::i64,
    };

    implementation_map<eltwise>::add(impl_types::cpu, shape_types::static_shape, eltwise_impl::create, types, formats);
    implementation_map<eltwise>::add(impl_types::cpu, shape_types::dynamic_shape, eltwise_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::eltwise_impl)
