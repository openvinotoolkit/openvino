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
        std::cout << "COMPARE_DATA RESULT: " << static_cast<float>(*lock_compare_data.data()) << std::endl;
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
        // std::cout << "------------------------------------------------------------------" << std::endl;
        // std::cout << "------------------------------------------------------------------" << std::endl;
        // {
        //     auto test_ptr = instance.dep_memory_ptr(1);
        //     auto pid = instance.dependencies()[1].first->id();
        //     mem_lock<uint8_t, mem_lock_type::read> lock_compare_data{test_ptr, instance.get_network().get_stream()};
        //     std::cout << "- [" << pid << "] insance. num_deps : " << instance.dependencies().size() << std::endl;
        //     std::cout << "- [" << pid << "] instance.compare_memory_ptr() for compare : " << test_ptr->buffer_ptr() << std::endl;
        //     std::cout << "- [" << pid << "] instance.mem_lock : " << static_cast<void*>(lock_compare_data.data()) << std::endl;
        //     std::cout << "- [" << pid << "] mem_size : " << test_ptr->count() << std::endl;
        //     std::cout << static_cast<int>(*lock_compare_data.data())<< std::endl;
        // }

        auto compare_data = get_compare_data(instance.compare_memory_ptr(), instance.get_network().get_stream());
        // mem_lock<half_t, mem_lock_type::read> lock_compare_data{instance.compare_memory_ptr(), instance.get_network().get_stream()};
        // auto compare_data_ptr = lock_compare_data.data();
        // auto pid = instance.dependencies()[0].first->id();
        // auto compare_data = (static_cast<float>(*lock_compare_data.data()) != 0.f);
        // std::cout << "- [" << pid << "] insance. num_deps : " << instance.dependencies().size() << std::endl;
        // std::cout << "- [" << pid << "] instance.compare_memory_ptr() for compare : " << instance.compare_memory_ptr()->buffer_ptr() << std::endl;
        // std::cout << "- [" << pid << "] instance.mem_lock : " << static_cast<void*>(lock_compare_data.data()) << std::endl;
        // std::cout << "- [" << pid << "] mem_size : " << instance.compare_memory_ptr()->count() << std::endl;
        // std::cout << "- [" << pid << "] mem_size : " << instance.compare_memory_ptr()->get_layout().to_short_string() << std::endl;
        // std::cout << "- [" << pid << "] Result of compare_data " << (compare_data? "True" : "False") << ", value: ";
        // std::cout << static_cast<int>(lock_compare_data.data()[0]) << " " << static_cast<float>(compare_data_ptr[0]) << std::endl;

        // std::cout << "------------------------------------------------------------------" << std::endl;
        // std::cout << "------------------------------------------------------------------" << std::endl;

        network::ptr executed_net = instance.get_inner_networks(compare_data);
        executed_net->execute({});



        // auto out_mem_ptr = executed_net->get_output_memory("condi_when_true");
        // {
        //     executed_net->get_stream().finish();
        //     const size_t size = out_mem_ptr->count();
        //     mem_lock<float, mem_lock_type::read> lock(out_mem_ptr, executed_net->get_stream());
        //     auto mem_ptr = lock.data();
        //     std::cout << "inner net - output[" << instance.id() << "] = {";
        //     for (size_t idx = 0; idx < size; idx++) {
        //         std::cout << mem_ptr[idx] << ",";
        //     }
        //     std::cout << "}" << std::endl;

        //     {
        //         auto test_mem_ptr = out_mem_ptr;
        //         std::cout << "* mem_ptr out of inner_net " << static_cast<void*>(test_mem_ptr->buffer_ptr());
        //         std::cout << " - " << static_cast<void*>(test_mem_ptr->buffer_ptr()) << std::endl;
        //     }

        //     {
        //         auto test_mem_ptr = instance.output_memory_ptr();
        //         std::cout << "* mem_ptr out of node " << static_cast<void*>(test_mem_ptr->buffer_ptr());
        //         std::cout << " - " << static_cast<void*>(test_mem_ptr->buffer_ptr()) << std::endl;
        //     }
        // }


        ev->set();
        return ev;
    }

    static std::unique_ptr<primitive_impl> create(const condition_node& arg, const kernel_impl_params&) {
        return make_unique<condition_impl>(arg);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

private:
    primitive_id _node_id;

    /*
    Add functions here.
    */
    bool check_condition(const float value_1, const float value_2, const cond_functions& func) const {
        switch (func) {
            case cond_functions::EQUAL:
                return value_1 == value_2;
                break;
            case cond_functions::GREATER:
                return value_1 > value_2;
                break;
            case cond_functions::LESS:
                return value_1 < value_2;
                break;
            default:
                throw("Unknown comparision function for: " + _node_id);
                break;
        }
    }
};

namespace detail {

attach_condition_common::attach_condition_common() {
    implementation_map<condition>::add(impl_types::common, condition_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f16, format::yxfb),
    });
}

}  // namespace detail
}  // namespace common
}  // namespace cldnn

ASSIGN_TYPE_NAME(cldnn::common::condition_impl)
