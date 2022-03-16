// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "data_inst.h"
#include "prior_box_inst.h"
#include "input_layout_inst.h"
#include "impls/implementation_map.hpp"
#include "register.hpp"
#include <vector>

namespace cldnn {
namespace common {

struct wait_for_events_impl : public primitive_impl {
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<wait_for_events_impl>(*this);
    }

    static std::unique_ptr<primitive_impl> create_data(const data_node&) { return make_unique<wait_for_events_impl>(); }

    static std::unique_ptr<primitive_impl> create_input_layout(const input_layout_node&) {
        return make_unique<wait_for_events_impl>();
    }

    static std::unique_ptr<primitive_impl> create_prior_box(const prior_box_node&) {
        // This primitive is being executed on CPU during network compilation.
        return make_unique<wait_for_events_impl>();
    }

private:
    void init_kernels(const program&) override {}
    void set_arguments(primitive_inst& /*instance*/) override {}
    std::vector<layout> get_internal_buffer_layouts() const override { return {}; }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        return stream.enqueue_marker(events);
    }

    bool validate(const primitive_inst&) const override { return true; }
};

namespace detail {

attach_data_common::attach_data_common() {
    implementation_map<data>::add(impl_types::common, wait_for_events_impl::create_data, {});
}

attach_input_layout_common::attach_input_layout_common() {
    implementation_map<input_layout>::add(impl_types::common, wait_for_events_impl::create_input_layout, {});
}

attach_prior_box_common::attach_prior_box_common() {
    implementation_map<prior_box>::add(impl_types::common, wait_for_events_impl::create_prior_box, {});
}

}  // namespace detail
}  // namespace common
}  // namespace cldnn
