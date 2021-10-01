// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive_inst.h"
#include "cldnn/runtime/error_handler.hpp"
#include "cldnn/runtime/memory.hpp"
#include "to_string_utils.h"
#include "register.hpp"
#include "utils.hpp"

#include "quantize_inst.h"

#include "reorder/reorder_weights_kernel_selector.h"
#include "reorder/reorder_kernel_base.h"

#include <vector>
#include <list>
#include <utility>

#include <oneapi/dnnl/dnnl.hpp>

namespace cldnn {
namespace onednn {

template <class PType, class DescType, class PrimDescType = dnnl::primitive_desc, class PrimType = dnnl::primitive>
struct typed_primitive_onednn_impl : public typed_primitive_impl<PType> {
    const typed_program_node<PType>& _outer;
    std::shared_ptr<DescType> _desc;
    std::shared_ptr<dnnl::primitive_attr> _attrs;
    PrimDescType _pd;
    PrimType _prim;
    std::unordered_map<int, dnnl::memory> _args;

    typed_primitive_onednn_impl(const typed_program_node<PType>& arg,
                                std::shared_ptr<DescType> desc,
                                std::shared_ptr<dnnl::primitive_attr> attrs,
                                const PrimDescType& pd,
                                kernel_selector::WeightsReorderParams weights_reorder = {})
        : typed_primitive_impl<PType>(weights_reorder, pd.impl_info_str()),
          _outer(arg),
          _desc(desc),
          _attrs(attrs),
          _pd(pd),
          _prim(pd) { }

    bool is_cpu() const override { return false; }

protected:
    virtual bool optimized_out(typed_primitive_inst<PType>&) const { return false; }

    static bool has_out_scales(const std::shared_ptr<dnnl::primitive_attr>& attr) {
        int mask;
        std::vector<float> scales;
        attr->get_output_scales(mask, scales);
        const auto drfv = reinterpret_cast<const int32_t&>(DNNL_RUNTIME_F32_VAL);
        return !scales.empty() && (reinterpret_cast<const int32_t&>(scales[0]) == drfv);
    }

    static bool has_zero_points(int arg, const std::shared_ptr<dnnl::primitive_attr>& attr) {
        int mask;
        std::vector<int32_t> zp;
        attr->get_zero_points(arg, mask, zp);
        const auto drsv = reinterpret_cast<const int32_t&>(DNNL_RUNTIME_S32_VAL);
        return !zp.empty() && (reinterpret_cast<const int32_t&>(zp[0]) == drsv);
    }

    virtual std::unordered_map<int, dnnl::memory> get_arguments(typed_primitive_inst<PType>& instance) const {
        std::unordered_map<int, dnnl::memory> args;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            auto& input = instance.input_memory(i);
            args.insert({DNNL_ARG_SRC, input.get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(static_cast<int>(i)))});
        }

        {
            auto& output = instance.output_memory();
            args.insert({DNNL_ARG_DST, output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0))});
        }

        return args;
    }

    void init_kernels() override { }

    static std::shared_ptr<dnnl::primitive_attr> get_primitive_attributes(const typed_program_node<PType>& /* arg */) {
        auto attrs = std::make_shared<dnnl::primitive_attr>();
        dnnl::post_ops post_ops;
        attrs->set_post_ops(post_ops);

        return attrs;
    }

    event::ptr aggregate_events(const std::vector<event::ptr>& events, stream& stream, bool group = false, bool is_output = false) const {
        if (events.size() == 1 && !is_output)
            return events[0];

        if (group && !is_output)
            return stream.group_events(events);

        return stream.enqueue_marker(events, is_output);
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override {
        _args = get_arguments(instance);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */,
                            typed_primitive_inst<PType>& instance) override {
        auto& network = instance.get_network();
        auto& engine = network.get_engine();
        auto& stream = network.get_stream();
        auto profiling = engine.configuration().enable_profiling;
        event::ptr event;

        if (profiling) {
            stream.finish();
            event = stream.create_user_event(false);
        }

        _prim.execute(stream.get_onednn_stream(), _args);

        if (profiling) {
            stream.finish();
            event->set();
        } else {
            // Create and set user event as complete
            event = stream.create_user_event(true);
        }

        if (!event) {
            std::string error_msg = "Event was not created properly for " + instance.id();
            throw std::runtime_error(error_msg);
        }

        return event;
    }
};

}  // namespace onednn
}  // namespace cldnn
