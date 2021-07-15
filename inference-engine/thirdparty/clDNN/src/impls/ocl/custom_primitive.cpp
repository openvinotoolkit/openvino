// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_gpu_primitive_inst.h"
#include "cldnn/runtime/engine.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "network_impl.h"
#include "jitter.h"
#include "cldnn/runtime/error_handler.hpp"
#include "register.hpp"

#include <map>
#include <sstream>
#include <vector>
#include <memory>
#include <string>

namespace kernel_selector {
using jit_constants = kernel_selector::JitConstants;
}

namespace cldnn {
namespace ocl {

struct custom_gpu_primitive_impl : typed_primitive_impl<custom_gpu_primitive> {
    const custom_gpu_primitive_node& outer;
    std::shared_ptr<kernel_selector::cl_kernel_data> cl_kernel;
    std::vector<kernel::ptr> _kernels;
    kernel_id _kernel_id;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<custom_gpu_primitive_impl>(*this);
    }

    custom_gpu_primitive_impl(const custom_gpu_primitive_impl& other)
    : outer(other.outer)
    , cl_kernel(other.cl_kernel)
    , _kernels({})
    , _kernel_id(other._kernel_id) {
        _kernels.emplace_back(std::move(outer.get_program().get_kernel(_kernel_id)->clone()));
    }

    custom_gpu_primitive_impl(const custom_gpu_primitive_node& arg,
                             std::shared_ptr<kernel_selector::cl_kernel_data>& cl_kernel)
        : outer(arg)
        , cl_kernel(cl_kernel)
        , _kernels() {
        _kernel_id = outer.get_program().add_kernel(cl_kernel->code.kernelString);
    }

    void init_kernels() override {
        _kernels.emplace_back(std::move(outer.get_program().get_kernel(_kernel_id)));
    }

    void set_arguments_impl(custom_gpu_primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        kernel_arguments_data args;
        for (auto& dep : instance.dependencies()) {
            args.inputs.push_back(dep->output_memory_ptr());
        }
        args.output = instance.output_memory_ptr();
        stream.set_arguments(*_kernels.front(), cl_kernel.get()->params, args);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                                 custom_gpu_primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        kernel_arguments_data args;
        for (auto& dep : instance.dependencies()) {
            args.inputs.push_back(dep->output_memory_ptr());
        }
        args.output = instance.output_memory_ptr();
        return stream.enqueue_kernel(*_kernels.front(), cl_kernel.get()->params, args, events, instance.node.is_output());
    }
};

static kernel_selector::kernel_argument_element get_arg(custom_gpu_primitive::arg_desc arg) {
    kernel_selector::kernel_argument_element ret;
    switch (arg.type) {
        case custom_gpu_primitive::arg_input:
            ret.t = kernel_selector::kernel_argument_types::INPUT;
            break;
        case custom_gpu_primitive::arg_output:
            ret.t = kernel_selector::kernel_argument_types::OUTPUT;
            break;
        default:
            throw std::runtime_error("Unknown argument type");
            break;
    }

    ret.index = arg.index;

    return ret;
}

std::string value_macro(const std::string& name, const std::string& value) {
    std::ostringstream oss;
    oss << "#define " << name << " " << value << std::endl;
    return oss.str();
}

static void add_layout_to_jit(kernel_selector::jit_constants& mem_consts, const std::string& name, const layout& l) {
    // Size (in elements)
    // #define INPUT0_DIMS (uint[]) { b, f, y, x, }
    mem_consts.AddConstant(kernel_selector::MakeJitConstant(name + "_DIMS", l.size.sizes(format::bfyx)));

    // Data type
    // #define INPUT0_TYPE float
    static const std::map<data_types, std::string> dataTypeToIndex{
        {data_types::i8, "char"},
        {data_types::u8, "uchar"},
        {data_types::i32, "int"},
        {data_types::i64, "long"},
        {data_types::f16, "half"},
        {data_types::f32, "float"},
    };

    if (dataTypeToIndex.find(l.data_type) == dataTypeToIndex.end()) {
        CLDNN_ERROR_MESSAGE("add layout to jit", "Unhandled data type in layout");
    }

    mem_consts.AddConstant(kernel_selector::MakeJitConstant(name + "_TYPE", dataTypeToIndex.at(l.data_type)));

    // Format
    // #define INPUT0_FORMAT_BFYX
    mem_consts.AddConstant(
        kernel_selector::MakeJitConstant(name + "_FORMAT_" + kernel_selector::toString(to_data_layout(l.format)), ""));

    // Padding (in elements)
    // #define INPUT0_LOWER_PADDING (uint[]) { 0, 0, 0, 0 }
    // #define INPUT0_UPPER_PADDING (uint[]) { 0, 0, 0, 0 }
    mem_consts.AddConstant(
        kernel_selector::MakeJitConstant(name + "_LOWER_PADDING", l.data_padding.lower_size().sizes(format::bfyx)));
    mem_consts.AddConstant(
        kernel_selector::MakeJitConstant(name + "_UPPER_PADDING", l.data_padding.upper_size().sizes(format::bfyx)));

    // Pitches (in elements)
    // #define INPUT0_PITCHES (uint[]) { b, f, h, w, }
    auto padded_sizes = l.get_buffer_size().sizes(format::bfyx);

    std::vector<tensor::value_type> pitches(4);
    switch (l.format) {
        case format::bfyx:
            pitches[3] = 1;
            pitches[2] = padded_sizes[3];
            pitches[1] = padded_sizes[2] * pitches[2];
            pitches[0] = padded_sizes[1] * pitches[1];
            break;
        case format::byxf:
            pitches[1] = 1;
            pitches[3] = padded_sizes[1];
            pitches[2] = padded_sizes[3] * pitches[3];
            pitches[0] = padded_sizes[2] * pitches[2];
            break;
        case format::yxfb:
            pitches[0] = 1;
            pitches[1] = padded_sizes[0];
            pitches[3] = padded_sizes[1] * pitches[1];
            pitches[2] = padded_sizes[3] * pitches[3];
            break;
        case format::fyxb:
            pitches[0] = 1;
            pitches[3] = padded_sizes[0];
            pitches[2] = padded_sizes[3] * pitches[3];
            pitches[1] = padded_sizes[2] * pitches[2];
            break;
        default:
            throw std::runtime_error("Unhandled format in pitch calculation");
    }

    mem_consts.AddConstant(kernel_selector::MakeJitConstant(name + "_PITCHES", pitches));

    // Offset (in elements)
    // #define INPUT0_OFFSET 0
    int32_t offset =
        (pitches[0] * l.data_padding.lower_size().batch[0]) + (pitches[1] * l.data_padding.lower_size().feature[0]) +
        (pitches[2] * l.data_padding.lower_size().spatial[1]) + (pitches[3] * l.data_padding.lower_size().spatial[0]);
    mem_consts.AddConstant(kernel_selector::MakeJitConstant(name + "_OFFSET", std::to_string(offset)));
}

static std::string get_jit_constant(const custom_gpu_primitive_node& outer) {
    kernel_selector::jit_constants mem_consts{
        kernel_selector::MakeJitConstant("NUM_INPUTS", std::to_string(outer.get_dependencies().size()))};
    const auto primitive = outer.get_primitive().get();

    mem_consts.AddConstants({
        kernel_selector::MakeJitConstant("GLOBAL_WORKSIZE", primitive->gws),
        kernel_selector::MakeJitConstant("LOCAL_WORKSIZE", primitive->lws),
    });

    for (size_t i = 0; i < outer.get_dependencies().size(); i++) {
        add_layout_to_jit(mem_consts, "INPUT" + std::to_string(i), outer.input(i).get_output_layout());
    }

    add_layout_to_jit(mem_consts, "OUTPUT0", outer.get_output_layout());

    std::ostringstream oss;
    oss << "// Custom Layer Built-ins\n\n";
    for (auto& definition : mem_consts.GetDefinitions()) {
        oss << value_macro(definition.first, definition.second);
    }

    return oss.str();
}

static primitive_impl* create(const custom_gpu_primitive_node& arg) {
    const auto primitive = arg.get_primitive().get();

    auto cl_kernel = std::make_shared<kernel_selector::cl_kernel_data>();
    cl_kernel->code.kernelString = std::make_shared<kernel_selector::kernel_string>();
    cl_kernel->code.kernelString->entry_point = primitive->kernel_entry_point;
    cl_kernel->code.kernelString->options = primitive->build_options;
    cl_kernel->code.kernelString->jit = get_jit_constant(arg);
    for (const auto& s : primitive->kernels_code) {
        cl_kernel->code.kernelString->str += s + "\n";
    }

    cl_kernel->params.workGroups.global = primitive->gws;
    cl_kernel->params.workGroups.local = primitive->lws;

    for (const auto& p : primitive->kernel_arguments) {
        cl_kernel->params.arguments.push_back(get_arg(p));
    }

    return new custom_gpu_primitive_impl(arg, cl_kernel);
}

namespace detail {

attach_custom_gpu_primitive_impl::attach_custom_gpu_primitive_impl() {
    implementation_map<custom_gpu_primitive>::add(cldnn::impl_types::ocl, create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
