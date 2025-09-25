// Copyright (C) 2018-2026 Intel Corporation
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

//forward declaration
static size_t evaluate_size_expr(const std::string& size_expr, const std::vector<int64_t>& input_dims);
static void allocate_internal_buffers(custom_gpu_primitive_inst& instance,
                                      std::vector<cldnn::memory::ptr>& internal_buffers,
                                      const std::unordered_map<uint32_t, std::string>& size_expr_map);

struct custom_gpu_primitive_impl : typed_primitive_impl<custom_gpu_primitive> {
    using parent = typed_primitive_impl<custom_gpu_primitive>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::custom_gpu_primitive_impl)

    std::shared_ptr<kernel_selector::cl_kernel_data> cl_kernel;
    std::vector<kernel::ptr> _kernels;
    std::unordered_map<uint32_t, std::string> size_expr_map;
    std::vector<memory::ptr> internal_buffers; // store allocated buffers here

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<custom_gpu_primitive_impl>(*this);
    }

    custom_gpu_primitive_impl()
    : _kernels() {}

    custom_gpu_primitive_impl(const custom_gpu_primitive_impl& other)
    : cl_kernel(other.cl_kernel)
    , _kernels({}) 
    , size_expr_map(other.size_expr_map) {
        for (const auto& kernel : other._kernels) {
            _kernels.emplace_back(kernel->clone(other.can_share_kernels));
        }
    }

    custom_gpu_primitive_impl(const custom_gpu_primitive_node& arg,
                             std::shared_ptr<kernel_selector::cl_kernel_data>& cl_kernel)
        : cl_kernel(cl_kernel)
        , _kernels() { }

    custom_gpu_primitive_impl(const custom_gpu_primitive_node& arg,
                             std::shared_ptr<kernel_selector::cl_kernel_data>& cl_kernel,
                             const std::unordered_map<uint32_t, std::string>& size_expr_map)
        : cl_kernel(cl_kernel)
        , _kernels()
        , size_expr_map(size_expr_map) { }

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

    void set_kernels(cldnn::kernels_cache::compiled_kernels kernels) override {
        OPENVINO_ASSERT(kernels.size() == 1, "Only the kernels of the single primitive should be allowed.");
        auto& kernel_vec = kernels.begin()->second;
        _kernels.clear();
        _kernels.resize(kernel_vec.size());
        for (auto& k : kernel_vec) {
            auto sub_kernel_idx = k.second;
            _kernels[sub_kernel_idx] = k.first;
        }
    }

    void set_arguments_impl(custom_gpu_primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        kernel_arguments_data args;
        for (auto& dep : instance.dependencies()) {
            args.inputs.push_back(dep.first->output_memory_ptr());
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
	    }
        
	    if (internal_buffers.empty() && !size_expr_map.empty()) {
            allocate_internal_buffers(instance, internal_buffers, size_expr_map);
        }
        
	    for (auto& buf : internal_buffers) {
            args.intermediates.push_back(buf);
        }
        stream.set_arguments(*_kernels.front(), cl_kernel.get()->params, args);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                                  custom_gpu_primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();

        kernel_arguments_data args;
        for (auto& dep : instance.dependencies()) {
            args.inputs.push_back(dep.first->output_memory_ptr());
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }

        for (auto& buf : internal_buffers) {
            args.intermediates.push_back(buf);
        }
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
        case custom_gpu_primitive::arg_internal:
            ret.t = kernel_selector::kernel_argument_types::INTERNAL_BUFFER;
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

static std::string get_jit_constant(const custom_gpu_primitive_node& outer,
                                    const kernel_impl_params& impl_param,
                                    const std::vector<size_t>& gws,
                                    const std::vector<size_t>& lws) {
    kernel_selector::jit_constants mem_consts{
        kernel_selector::MakeJitConstant("NUM_INPUTS", std::to_string(outer.get_dependencies().size()))};

    mem_consts.AddConstants({
        kernel_selector::MakeJitConstant("GLOBAL_WORKSIZE", gws),
        kernel_selector::MakeJitConstant("LOCAL_WORKSIZE", lws),
    });

    for (size_t i = 0; i < impl_param.input_layouts.size(); i++) {
        add_layout_to_jit(mem_consts, "INPUT" + std::to_string(i), impl_param.get_input_layout(i));
    }

    for (size_t i = 0; i < impl_param.output_layouts.size(); i++) {
        add_layout_to_jit(mem_consts, "OUTPUT" + std::to_string(i), impl_param.get_output_layout(i));
    }

    std::ostringstream oss;
    oss << "// Custom Layer Built-ins\n\n";
    for (auto& definition : mem_consts.GetDefinitions()) {
        oss << value_macro(definition.first, definition.second);
    }

    return oss.str();
}

static void allocate_internal_buffers(custom_gpu_primitive_inst& instance,
                               std::vector<cldnn::memory::ptr>& internal_buffers,
                               const std::unordered_map<uint32_t, std::string>& size_expr_map) {
    if (!internal_buffers.empty())
        return;

    auto& engine = instance.get_network().get_engine();

    // get input layout
    const auto& input_layout = instance.dependencies().at(0).first->get_output_layout();

    // use shape (works for static + dynamic)
    auto shape = input_layout.get_shape();
    std::vector<int64_t> input_dims(shape.begin(), shape.end());

    // use runtime data type (f16/f32/etc.)
    auto data_type = input_layout.data_type;

    for (const auto& [index, size_expr] : size_expr_map) {
        // evaluate expression
        size_t element_count = evaluate_size_expr(size_expr, input_dims);

        // allocate buffer with correct dtype
        cldnn::layout internal_layout(
            data_type,
            cldnn::format::bfyx,
            cldnn::tensor(1, 1, 1, static_cast<int32_t>(element_count))
        );

        auto internal_buf = engine.allocate_memory(internal_layout);
        OPENVINO_ASSERT(internal_buf != nullptr, "Failed to allocate internal buffer");
        internal_buffers.push_back(internal_buf);
    }
}

static size_t evaluate_size_expr(const std::string& size_expr, const std::vector<int64_t>& input_dims) {
    std::string expr = size_expr;
    for (size_t i = 0; i < input_dims.size(); ++i) {
        std::string token = "INPUT0_DIMS[" + std::to_string(i) + "]";
        size_t pos = 0;
        while ((pos = expr.find(token, pos)) != std::string::npos) {
            expr.replace(pos, token.length(), std::to_string(input_dims[i]));
            pos += std::to_string(input_dims[i]).length();
        }
    }
    size_t result = 1;
    std::stringstream ss(expr);
    std::string item;
    while (std::getline(ss, item, '*')) {
        item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
        result *= std::stoul(item);
    }
    return result;
}

static std::unique_ptr<primitive_impl> create(const custom_gpu_primitive_node& arg, const kernel_impl_params& impl_param) {
    const auto primitive = arg.get_primitive().get();

    const auto& orig_output_layout = impl_param.get_output_layout();
    OPENVINO_ASSERT(orig_output_layout.is_static(), "out layouts should be static for create primitive_impl!");

    std::vector<size_t> gws, lws;
    custom_gpu_primitive::update_work_group_size(orig_output_layout.get_partial_shape(),
                                                 primitive->calcWgDimInputIdx,
                                                 orig_output_layout.get_partial_shape(),
                                                 primitive->globalSizeRules,
                                                 primitive->localSizeRules,
                                                 gws,
                                                 lws);

    if (gws.empty()) {
        gws = primitive->gws;
    }
    if (lws.empty()) {
        lws = primitive->lws;
    }

    auto cl_kernel = std::make_shared<kernel_selector::cl_kernel_data>();
    cl_kernel->code.kernelString = std::make_shared<kernel_selector::kernel_string>();
    cl_kernel->code.kernelString->entry_point = primitive->kernel_entry_point;
    cl_kernel->code.kernelString->options = primitive->build_options;
    const std::vector<size_t> const_gws = gws;
    const std::vector<size_t> const_lws = lws;
    cl_kernel->code.kernelString->jit = get_jit_constant(arg, impl_param, const_gws, const_lws);
    for (const auto& s : primitive->kernels_code) {
        cl_kernel->code.kernelString->str += s + "\n";
    }

    cl_kernel->params.workGroups.global = gws;
    cl_kernel->params.workGroups.local = lws;

    std::unordered_map<uint32_t, std::string> size_expr_map;
    for (const auto& p : primitive->kernel_arguments) {
        cl_kernel->params.arguments.push_back(get_arg(p));
        if (p.type == custom_gpu_primitive::arg_internal) {
            size_expr_map[p.index] = p.size_expr;
        }
    }
    if (!size_expr_map.empty()) {
        return std::make_unique<custom_gpu_primitive_impl>(arg, cl_kernel, size_expr_map);
    } else {
        return std::make_unique<custom_gpu_primitive_impl>(arg, cl_kernel);
    }
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
