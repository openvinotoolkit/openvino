// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "data_inst.h"
#include "prior_box_inst.h"
#include "input_layout_inst.h"
#include "impls/registry/implementation_map.hpp"
#include "register.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include <vector>

namespace cldnn {
namespace common {

class wait_for_events_impl : public primitive_impl {
    using primitive_impl::primitive_impl;

public:
    explicit wait_for_events_impl(const program_node& /*node*/)
        : primitive_impl("wait_for_events") { }

    wait_for_events_impl() : primitive_impl() {}

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::common::wait_for_events_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<wait_for_events_impl>(*this);
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}
    void set_arguments(primitive_inst& /*instance*/) override {}
    void set_arguments(primitive_inst& /*instance*/, kernel_arguments_data& /*args*/) override {}
    std::vector<layout> get_internal_buffer_layouts(const kernel_impl_params& /*params*/) const override { return {}; }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();

        return events.empty() ? stream.create_user_event(true)
                              : stream.enqueue_marker(events);
    }

    static std::unique_ptr<primitive_impl> create_data(const data_node& data, const kernel_impl_params&) {
        return make_unique<wait_for_events_impl>(data);
    }

    static std::unique_ptr<primitive_impl> create_input_layout(const input_layout_node& input, const kernel_impl_params&) {
        return make_unique<wait_for_events_impl>(input);
    }

    static std::unique_ptr<primitive_impl> create_prior_box(const prior_box_node& prior_box, const kernel_impl_params&) {
        // This primitive is being executed on CPU during network compilation.
        return make_unique<wait_for_events_impl>(prior_box);
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override { }
};

namespace detail {

attach_data_common::attach_data_common() {
    implementation_map<data>::add(impl_types::common, shape_types::any, wait_for_events_impl::create_data, {});
}

attach_input_layout_common::attach_input_layout_common() {
    implementation_map<input_layout>::add(impl_types::common, shape_types::any, wait_for_events_impl::create_input_layout, {});
}

attach_prior_box_common::attach_prior_box_common() {
    implementation_map<prior_box>::add(impl_types::common, wait_for_events_impl::create_prior_box, {});
}

}  // namespace detail
}  // namespace common
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::common::wait_for_events_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::data)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::input_layout)
