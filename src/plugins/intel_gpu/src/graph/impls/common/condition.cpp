// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "condition_inst.h"
#include "implementation_map.hpp"
#include "register.hpp"

#include <algorithm>
#include <vector>

namespace cldnn {
namespace common {

struct condition_impl : typed_primitive_impl<condition> {
    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<condition_impl>(*this);
    }

    explicit condition_impl(const condition_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<condition>());
        const auto& node = arg.as<condition>();
        _node_id = node.id();
    }

    template <class T>
    bool compare_data(memory::ptr mem, stream& stream) {
        mem_lock<T, mem_lock_type::read> lock_compare_data{mem, stream};
        return (static_cast<float>(*lock_compare_data.data()) != 0.f);
    }

    bool get_compare_data(memory::ptr mem, stream& stream) {
        auto mem_dt = mem->get_layout().data_type;
        switch (mem_dt) {
            case cldnn::data_types::f32:
                return compare_data<float>(mem, stream);
            case cldnn::data_types::f16:
                return compare_data<half_t>(mem, stream);
            case cldnn::data_types::i64:
                return compare_data<int64_t>(mem, stream);
            case cldnn::data_types::i32:
                return compare_data<int32_t>(mem, stream);
            case cldnn::data_types::i8:
                return compare_data<int8_t>(mem, stream);
            case cldnn::data_types::u8:
                return compare_data<uint8_t>(mem, stream);
            case cldnn::data_types::bin:
            default:
                return compare_data<uint32_t>(mem, stream);
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, condition_inst& instance) override {
        for (auto& a : events) {
            a->wait();
        }
        auto ev = instance.get_network().get_stream().create_user_event(false);
        set_node_params(instance.get_node());

        auto compare_data = get_compare_data(instance.compare_memory_ptr(), instance.get_network().get_stream());
        network::ptr executed_net = instance.get_inner_networks(compare_data);
        executed_net->execute({});

        ev->set();
        return ev;
    }

    static std::unique_ptr<primitive_impl> create(const condition_node& arg, const kernel_impl_params&) {
        return make_unique<condition_impl>(arg);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

private:
    primitive_id _node_id;
};

namespace detail {

attach_condition_common::attach_condition_common() {
    implementation_map<condition>::add(impl_types::common, condition_impl::create, {});
}

}  // namespace detail
}  // namespace common
}  // namespace cldnn

// TODO: Change code like cldnn::loop
ASSIGN_TYPE_NAME(cldnn::common::condition_impl)
