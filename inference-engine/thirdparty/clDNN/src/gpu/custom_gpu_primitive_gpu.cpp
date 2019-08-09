/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "custom_gpu_primitive_inst.h"
#include "kernel.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "network_impl.h"
#include "engine_impl.h"
#include "jitter.h"
#include "error_handler.h"

#include <map>
#include <sstream>
#include <vector>
#include <memory>
#include <string>

using namespace cldnn;
namespace kernel_selector {
using jit_constants = kernel_selector::JitConstants;
}

namespace neural {

struct custom_gpu_primitive_gpu : typed_primitive_impl<custom_gpu_primitive> {
    const custom_gpu_primitive_node& outer;
    std::shared_ptr<kernel_selector::cl_kernel_data> cl_kernel;
    gpu::kernel _kernel;

    custom_gpu_primitive_gpu(const custom_gpu_primitive_node& arg,
                             std::shared_ptr<kernel_selector::cl_kernel_data>& cl_kernel)
        : outer(arg),
          cl_kernel(cl_kernel),
          _kernel(arg.get_program().get_engine().get_context(),
                  cl_kernel->kernelString,
                  arg.get_program().get_engine().get_context()->get_configuration().dump_custom_program) {}

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events,
                                 custom_gpu_primitive_inst& instance) override {
        uint16_t stream_id = instance.get_network().get_stream_id();
        gpu::kernel::kernel_arguments_data args;
        for (auto& dep : instance.dependencies()) {
            args.inputs.push_back((memory_impl::cptr) &(dep->output_memory()));
        }
        args.output = (memory_impl::cptr) &instance.output_memory();
        _kernel.set_output_event(stream_id, instance.node.is_output());
        return _kernel.run(stream_id, *cl_kernel.get(), events, args);
    }
};

static kernel_selector::kernel_argument_element get_arg(cldnn_arg arg) {
    kernel_selector::kernel_argument_element ret;
    switch (arg.arg_type) {
        case arg_input:
            ret.t = kernel_selector::kernel_argument_types::INPUT;
            break;
        case arg_output:
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

static void add_layout_to_jit(kernel_selector::jit_constants& mem_consts, const std::string& name, layout l) {
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
    cl_kernel->kernelString = std::make_shared<kernel_selector::kernel_string>();
    cl_kernel->kernelString->entry_point = primitive->kernel_entry_point;
    cl_kernel->kernelString->options = primitive->build_options;
    cl_kernel->kernelString->jit = get_jit_constant(arg);
    for (const auto& s : primitive->kernels_code) {
        cl_kernel->kernelString->str += s + "\n";
    }

    cl_kernel->workGroups.global = primitive->gws;
    cl_kernel->workGroups.local = primitive->lws;

    for (const auto& p : primitive->kernel_arguments) {
        cl_kernel->arguments.push_back(get_arg(p));
    }

    return new custom_gpu_primitive_gpu(arg, cl_kernel);
}

namespace {
struct attach {
    attach() { implementation_map<custom_gpu_primitive>::add({{cldnn::engine_types::ocl, create}}); }
    ~attach() {}
};
attach attach_impl;
}  // namespace
}  // namespace neural
