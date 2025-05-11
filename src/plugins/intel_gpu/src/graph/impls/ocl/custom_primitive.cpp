// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "custom_gpu_primitive_inst.h"
#include "jitter.h"

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
    using parent = typed_primitive_impl<custom_gpu_primitive>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::custom_gpu_primitive_impl)

    std::shared_ptr<kernel_selector::cl_kernel_data> cl_kernel;
    std::vector<kernel::ptr> _kernels;

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<custom_gpu_primitive_impl>(*this);
    }

    custom_gpu_primitive_impl()
    : _kernels() {}

    custom_gpu_primitive_impl(const custom_gpu_primitive_impl& other)
    : cl_kernel(other.cl_kernel)
    , _kernels({}) {
        for (const auto& kernel : other._kernels) {
            _kernels.emplace_back(kernel->clone(other.can_share_kernels));
        }
    }

    custom_gpu_primitive_impl(const custom_gpu_primitive_node& arg,
                             std::shared_ptr<kernel_selector::cl_kernel_data>& cl_kernel)
        : cl_kernel(cl_kernel)
        , _kernels() { }

    std::vector<std::shared_ptr<cldnn::kernel_string>> get_kernels_source() override {
        std::vector<std::shared_ptr<cldnn::kernel_string>> kernel_strings;
        kernel_strings.push_back(cl_kernel->code.kernelString);
        return kernel_strings;
    }

    void init_kernels(const kernels_cache& kernels_cache, const kernel_impl_params& params) override {
        _kernels.clear();
        auto compiled_kernels = kernels_cache.get_kernels(params);
        _kernels.insert(_kernels.begin(), compiled_kernels.begin(), compiled_kernels.end());
        this->can_share_kernels = kernels_cache.get_kernels_reuse();
    }

    void init_by_cached_kernels(const kernels_cache& kernels_cache, std::vector<std::string>& cached_kernel_ids) override {
        _kernels.emplace_back(kernels_cache.get_kernel_from_cached_kernels(cached_kernel_ids[0]));
        this->can_share_kernels = kernels_cache.get_kernels_reuse();
    }

    std::vector<std::string> get_cached_kernel_ids(const kernels_cache& kernels_cache) override {
        return {kernels_cache.get_cached_kernel_id(_kernels[0])};
    }

    void set_arguments_impl(custom_gpu_primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        kernel_arguments_data args;
        for (auto& dep : instance.dependencies()) {
            args.inputs.push_back(dep.first->output_memory_ptr());
        }
        args.outputs = { instance.output_memory_ptr() };
        stream.set_arguments(*_kernels.front(), cl_kernel.get()->params, args);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                                 custom_gpu_primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        kernel_arguments_data args;
        for (auto& dep : instance.dependencies()) {
            args.inputs.push_back(dep.first->output_memory_ptr());
        }
        args.outputs = { instance.output_memory_ptr() };
        return stream.enqueue_kernel(*_kernels.front(), cl_kernel.get()->params, args, events, instance.is_output());
    }

    std::vector<kernel::ptr> get_kernels() const override {
        return _kernels;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << *cl_kernel;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        cl_kernel = std::make_shared<kernel_selector::cl_kernel_data>();
        ib >> *cl_kernel;
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

static std::string value_macro(const std::string& name, const std::string& value) {
    std::ostringstream oss;
    oss << "#define " << name << " " << value << std::endl;
    return oss.str();
}

static void add_layout_to_jit(kernel_selector::jit_constants& mem_consts, const std::string& name, const layout& l) {
    // Size (in elements)
    // #define INPUT0_DIMS (uint[]) { b, f, y, x, }
    mem_consts.AddConstant(kernel_selector::MakeJitConstant(name + "_DIMS", l.get_tensor().sizes(format::bfyx)));

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

    OPENVINO_ASSERT(dataTypeToIndex.find(l.data_type) != dataTypeToIndex.end(), "[GPU] Add layout to jit error: unhandled data type in layout");

    mem_consts.AddConstant(kernel_selector::MakeJitConstant(name + "_TYPE", dataTypeToIndex.at(l.data_type)));

    // Format
    // #define INPUT0_FORMAT_BFYX
    mem_consts.AddConstant(
        kernel_selector::MakeJitConstant(name + "_FORMAT_" + kernel_selector::toString(to_data_layout(l.format)), ""));

    // Padding (in elements)
    // #define INPUT0_LOWER_PADDING (uint[]) { 0, 0, 0, 0 }
    // #define INPUT0_UPPER_PADDING (uint[]) { 0, 0, 0, 0 }
    mem_consts.AddConstant(
        kernel_selector::MakeJitConstant(name + "_LOWER_PADDING", layout::format_sizes(l.data_padding._lower_size, format::bfyx)));
    mem_consts.AddConstant(
        kernel_selector::MakeJitConstant(name + "_UPPER_PADDING", layout::format_sizes(l.data_padding._upper_size, format::bfyx)));

    // Pitches (in elements)
    // #define INPUT0_PITCHES (uint[]) { b, f, h, w, }
    // auto padded_sizes = l.get_buffer_size().sizes(format::bfyx);
    auto padded_sizes = l.get_padded_dims();

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
        (pitches[0] * l.data_padding._lower_size[0]) + (pitches[1] * l.data_padding._lower_size[1]) +
        (pitches[2] * l.data_padding._lower_size[3]) + (pitches[3] * l.data_padding._lower_size[2]);
    mem_consts.AddConstant(kernel_selector::MakeJitConstant(name + "_OFFSET", std::to_string(offset)));
}

static std::string get_jit_constant(const custom_gpu_primitive_node& outer, const kernel_impl_params& impl_param) {
    kernel_selector::jit_constants mem_consts{
        kernel_selector::MakeJitConstant("NUM_INPUTS", std::to_string(outer.get_dependencies().size()))};
    const auto primitive = outer.get_primitive().get();

    mem_consts.AddConstants({
        kernel_selector::MakeJitConstant("GLOBAL_WORKSIZE", primitive->gws),
        kernel_selector::MakeJitConstant("LOCAL_WORKSIZE", primitive->lws),
    });

    for (size_t i = 0; i < impl_param.input_layouts.size(); i++) {
        add_layout_to_jit(mem_consts, "INPUT" + std::to_string(i), impl_param.get_input_layout(i));
    }

    add_layout_to_jit(mem_consts, "OUTPUT0", impl_param.get_output_layout());

    std::ostringstream oss;
    oss << "// Custom Layer Built-ins\n\n";
    for (auto& definition : mem_consts.GetDefinitions()) {
        oss << value_macro(definition.first, definition.second);
    }

    return oss.str();
}

static std::unique_ptr<primitive_impl> create(const custom_gpu_primitive_node& arg, const kernel_impl_params& impl_param) {
    const auto primitive = arg.get_primitive().get();

    auto cl_kernel = std::make_shared<kernel_selector::cl_kernel_data>();
    cl_kernel->code.kernelString = std::make_shared<kernel_selector::kernel_string>();
    cl_kernel->code.kernelString->entry_point = primitive->kernel_entry_point;
    cl_kernel->code.kernelString->options = primitive->build_options;
    cl_kernel->code.kernelString->jit = get_jit_constant(arg, impl_param);
    for (const auto& s : primitive->kernels_code) {
        cl_kernel->code.kernelString->str += s + "\n";
    }

    cl_kernel->params.workGroups.global = primitive->gws;
    cl_kernel->params.workGroups.local = primitive->lws;

    for (const auto& p : primitive->kernel_arguments) {
        cl_kernel->params.arguments.push_back(get_arg(p));
    }

    return std::make_unique<custom_gpu_primitive_impl>(arg, cl_kernel);
}

namespace detail {

attach_custom_gpu_primitive_impl::attach_custom_gpu_primitive_impl() {
    implementation_map<custom_gpu_primitive>::add(cldnn::impl_types::ocl, create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::custom_gpu_primitive_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::custom_gpu_primitive)
