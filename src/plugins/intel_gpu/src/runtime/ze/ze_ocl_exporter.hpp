// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_resource.hpp"
#include "ze_ocl_interop.hpp"

namespace cldnn {
namespace ze {

/// @brief Adds exported OpenCL handle to existing Level Zero resource. Does nothing if resource is already exported.
template<ze_resource_type source_type, ocl_resource_type target_type>
struct ze_ocl_exporter {
    static_assert(false, "Exporter for given resource types is not implemented");
};

template<>
struct ze_ocl_exporter<ze_resource_type::device, ocl_resource_type::device>{
public:
    static constexpr ze_resource_type source_type = ze_resource_type::device;
    static constexpr ocl_resource_type target_type = ocl_resource_type::device;
    using ocl_owner_t = ocl_owner<target_type>;
    using resource_t = ze_resource<source_type>;
    void operator()(resource_t &resource) {
        if (resource.has_ocl_handle<target_type>()) {
            return;
        }
        auto &interop = ze_ocl_interop::get_instance();
        auto handle = resource.get_ze_handle();
        auto ocl_handle = interop.find_ocl_device(handle);
        ocl_owner_t ocl_owner(ocl_handle);
        resource.attach_ocl_handle<target_type>(std::move(ocl_owner));
    }
};

template<>
struct ze_ocl_exporter<ze_resource_type::context, ocl_resource_type::context> {
public:
    static constexpr ze_resource_type source_type = ze_resource_type::context;
    static constexpr ocl_resource_type target_type = ocl_resource_type::context;
    using ocl_owner_t = ocl_owner<target_type>;
    using resource_t = ze_resource<source_type>;
    struct args_t {
        ze_device_resource device;
    };
    ze_ocl_exporter(args_t args) : _args(args) {}
    void operator()(resource_t &resource) {
        if (resource.has_ocl_handle<target_type>()) {
            return;
        }
        ze_ocl_exporter<ze_resource_type::device, ocl_resource_type::device> device_exporter;
        device_exporter(_args.device);
        ocl_context_args context_args;
        context_args.device = _args.device.get_ocl_handle<ocl_resource_type::device>();

        auto &interop = ze_ocl_interop::get_instance();
        auto handle = resource.get_ze_handle();
        auto ocl_handle = interop.create_cl_context(handle, context_args);
        ocl_owner_t ocl_owner(ocl_handle);
        resource.attach_ocl_handle<target_type>(std::move(ocl_owner));
    }
private:
    args_t _args;
};

template<>
struct ze_ocl_exporter<ze_resource_type::command_list, ocl_resource_type::command_queue> {
public:
    static constexpr ze_resource_type source_type = ze_resource_type::command_list;
    static constexpr ocl_resource_type target_type = ocl_resource_type::command_queue;
    using ocl_owner_t = ocl_owner<target_type>;
    using resource_t = ze_resource<source_type>;
    struct args_t {
        ze_device_resource device;
        ze_context_resource context;
    };
    ze_ocl_exporter(args_t args) : _args(args) {}
    void operator()(resource_t &resource) {
        if (resource.has_ocl_handle<target_type>()) {
            return;
        }
        ze_ocl_exporter<ze_resource_type::device, ocl_resource_type::device> device_exporter;
        device_exporter(_args.device);
        ze_ocl_exporter<ze_resource_type::context, ocl_resource_type::context> context_exporter({_args.device});
        context_exporter(_args.context);
        ocl_queue_args queue_args;
        queue_args.device = _args.device.get_ocl_handle<ocl_resource_type::device>();
        queue_args.context = _args.context.get_ocl_handle<ocl_resource_type::context>();

        auto &interop = ze_ocl_interop::get_instance();
        auto handle = resource.get_ze_handle();
        auto ocl_handle = interop.create_cl_queue(handle, queue_args);
        ocl_owner_t ocl_owner(ocl_handle);
        resource.attach_ocl_handle<target_type>(std::move(ocl_owner));
    }
private:
    args_t _args;
};

template<>
struct ze_ocl_exporter<ze_resource_type::usm_memory, ocl_resource_type::mem_object> {
public:
    static constexpr ze_resource_type source_type = ze_resource_type::usm_memory;
    static constexpr ocl_resource_type target_type = ocl_resource_type::mem_object;
    using ocl_owner_t = ocl_owner<target_type>;
    using resource_t = ze_resource<source_type>;
    struct args_t {
        ze_device_resource device;
        ze_context_resource context;
        cl_mem_flags flags;
        size_t size;
    };
    ze_ocl_exporter(args_t args) : _args(args) {}
    void operator()(resource_t &resource) {
        if (resource.has_ocl_handle<target_type>()) {
            return;
        }
        ze_ocl_exporter<ze_resource_type::context, ocl_resource_type::context> context_exporter({_args.device});
        context_exporter(_args.context);
        ocl_buffer_args buffer_args;
        buffer_args.context = _args.context.get_ocl_handle<ocl_resource_type::context>();
        buffer_args.flags = _args.flags;
        buffer_args.size = _args.size;

        auto &interop = ze_ocl_interop::get_instance();
        auto handle = resource.get_ze_handle();
        auto ocl_handle = interop.create_cl_buffer(handle.ptr, buffer_args);
        ocl_owner_t ocl_owner(ocl_handle);
        resource.attach_ocl_handle<target_type>(std::move(ocl_owner));
    }
private:
    args_t _args;
};

template<>
struct ze_ocl_exporter<ze_resource_type::image, ocl_resource_type::mem_object> {
public:
    static constexpr ze_resource_type source_type = ze_resource_type::image;
    static constexpr ocl_resource_type target_type = ocl_resource_type::mem_object;
    using ocl_owner_t = ocl_owner<target_type>;
    using resource_t = ze_resource<source_type>;
    struct args_t {
        ze_device_resource device;
        ze_context_resource context;
        cl_mem_flags flags;
        cl_image_format format;
        cl_image_desc desc;
    };
    ze_ocl_exporter(args_t args) : _args(args) {}
    void operator()(resource_t &resource) {
        if (resource.has_ocl_handle<target_type>()) {
            return;
        }
        ze_ocl_exporter<ze_resource_type::context, ocl_resource_type::context> context_exporter({_args.device});
        context_exporter(_args.context);
        ocl_image_args image_args;
        image_args.context = _args.context.get_ocl_handle<ocl_resource_type::context>();
        image_args.flags = _args.flags;
        image_args.format = _args.format;
        image_args.desc = _args.desc;

        auto &interop = ze_ocl_interop::get_instance();
        auto handle = resource.get_ze_handle();
        auto ocl_handle = interop.create_cl_image(handle, image_args);
        ocl_owner_t ocl_owner(ocl_handle);
        resource.attach_ocl_handle<target_type>(std::move(ocl_owner));
    }
private:
    args_t _args;
};
}
}
