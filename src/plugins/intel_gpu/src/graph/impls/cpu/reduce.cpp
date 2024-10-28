// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "reduce_inst.h"
#include "impls/registry/implementation_map.hpp"

#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"

namespace cldnn {
namespace cpu {

namespace {

template<typename T>
std::shared_ptr<ov::op::Op> make_reduce(bool keep_dims) {
    auto op = std::make_shared<T>();
    op->set_keep_dims(keep_dims);
    return op;
}
}  // namespace

struct reduce_impl : public typed_primitive_impl<reduce> {
    using parent = typed_primitive_impl<reduce>;
    using parent::parent;

    reduce_mode mode = reduce_mode::sum;
    std::vector<int64_t> axes = {};
    bool keep_dims = false;

    std::shared_ptr<ov::op::Op> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::reduce_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reduce_impl>(*this);
    }

    reduce_impl() : parent("reduce_cpu_impl") {}

    explicit reduce_impl(const reduce_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<reduce>(), "[GPU] Incorrect program_node type");
        const auto& node = arg.as<reduce>();
        mode = node.get_primitive()->mode;
        axes = node.get_primitive()->axes;
        keep_dims = node.get_primitive()->keep_dims;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << make_data(&mode, sizeof(reduce_mode));
        ob << axes;
        ob << keep_dims;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> make_data(&mode, sizeof(reduce_mode));
        ib >> axes;
        ib >> keep_dims;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, reduce_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "reduce::execute_impl");
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
            op = make_reduce<ov::op::v1::ReduceProd>(keep_dims);
            switch (mode) {
            case reduce_mode::max:
                op = make_reduce<ov::op::v1::ReduceMax>(keep_dims);
                break;
            case reduce_mode::min:
                op = make_reduce<ov::op::v1::ReduceMin>(keep_dims);
                break;
            case reduce_mode::mean:
                op = make_reduce<ov::op::v1::ReduceMean>(keep_dims);
                break;
            case reduce_mode::prod:
                op = make_reduce<ov::op::v1::ReduceProd>(keep_dims);
                break;
            case reduce_mode::sum:
                op = make_reduce<ov::op::v1::ReduceSum>(keep_dims);
                break;
            case reduce_mode::logical_and:
                op = make_reduce<ov::op::v1::ReduceLogicalAnd>(keep_dims);
                break;
            case reduce_mode::logical_or:
                op = make_reduce<ov::op::v1::ReduceLogicalOr>(keep_dims);
                break;
            case reduce_mode::l1:
                op = make_reduce<ov::op::v4::ReduceL1>(keep_dims);
                break;
            case reduce_mode::l2:
                op = make_reduce<ov::op::v4::ReduceL2>(keep_dims);
                break;
            default:
                OPENVINO_THROW("[GPU] Couldn't create reduce operation: unsupported reduce mode (", static_cast<size_t>(mode), ")");
            }
        }

        cldnn::mem_lock<uint8_t, mem_lock_type::write> output_lock(instance.output_memory_ptr(), stream);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> input_lock(instance.dep_memory_ptr(0), stream);

        input_host_tensors.push_back(make_tensor(params->input_layouts[0], input_lock.data()));
        input_host_tensors.push_back(ov::Tensor(ov::element::i64, ov::Shape{axes.size()}, static_cast<void*>(axes.data())));

        output_host_tensors.push_back(make_tensor(params->output_layouts[0], output_lock.data()));

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute reduce primitive with id ", instance.id());

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
    static std::unique_ptr<primitive_impl> create(const reduce_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<reduce_impl>();
    }
};


namespace detail {

attach_reduce_impl::attach_reduce_impl() {
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

    implementation_map<reduce>::add(impl_types::cpu, shape_types::static_shape, reduce_impl::create, types, formats);
    implementation_map<reduce>::add(impl_types::cpu, shape_types::dynamic_shape, reduce_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::reduce_impl)
