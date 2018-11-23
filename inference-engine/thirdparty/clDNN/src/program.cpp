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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "program_impl.h"
#include "primitive_inst.h"
#include "layout_optimizer.h"
#include "constants_propagator.h"

#include "primitive_type.h"
#include "api/CPP/activation.hpp"
#include "api/CPP/eltwise.hpp"
#include "api/CPP/input_layout.hpp"
#include "api/CPP/pooling.hpp"
#include "api/CPP/proposal.hpp"
#include "api/CPP/roi_pooling.hpp"
#include "api/CPP/reorg_yolo.hpp"

#include "activation_inst.h"
#include "batch_norm_inst.h"
#include "internal_primitive.h"
#include "internal_primitive_type_base.h"
#include "convolution_inst.h"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "deconvolution_inst.h"
#include "detection_output_inst.h"
#include "lrn_inst.h"
#include "normalize_inst.h"
#include "permute_inst.h"
#include "prior_box_inst.h"
#include "reorder_inst.h"
#include "reshape_inst.h"
#include "scale_inst.h"
#include "embed_inst.h"
#include "softmax_inst.h"
#include "split_inst.h"
#include "program_dump_graph.h"
#include "upsampling_inst.h"
#include "eltwise_inst.h"
#include "fully_connected_inst.h"
#include "mvn_inst.h"
#include "lstm_inst.h"
#include "lstm_gemm_inst.h"
#include "lstm_elt_inst.h"
#include "embed_inst.h"

#include "network_impl.h"
#include "kernel_selector_helper.h"
#include "sliding_window_utils.h"
#include "error_handler.h"

#include <fstream>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>


namespace {

    //helper function for selecting function basing on the type of the given primitive
    //this is the termination case for parameter pack recurrence, see overload below for logic
    template <class... T>
    void do_for_types(program_node&)
    {
        return;
    }

    //helper function for selecting function basing on the type of the given primitive
    //this function should be explicitly given set of types and implicitly set of functions.
    //both sets should have equal size. First function will be called if type of the given primitive
    //will match first explicitly given type, second will be called if it matches second explicitly given
    //type etc.
    //Functions given as arguments should themselves take std::shared_ptr<const T> as argument
    //where T is the type that should be match if this function should be called
    //
    //example:
    // do_for_types<
    //      convolution,
    //      pooling
    //  >(primitive,
    //      [](typed_program_node<convolution>&){ do something if 'primitive' is a convolution },
    //      [](typed_program_node<pooling>&)    { do something if 'primitive' is a pooling }
    //  );
    template <class T, class... RestOfT, class Func, class... RestOfFuncs>
    decltype(static_cast<void>(std::declval<Func>()(std::declval<typed_program_node<T>&>()))) do_for_types(
        program_node& node,
        Func const& func,
        RestOfFuncs const&... rest)
    {
        if (node.type() == T::type_id())
            func(node.as<T>());
        else
            do_for_types<RestOfT...>(node, rest...);
    }

    template <class T>
    struct single_element_container
    {
        single_element_container(T& t) : elem(&t)
        {}

        constexpr size_t size() const { return 1; }
        single_element_container<T> begin() const { return single_element_container(elem); }
        single_element_container<T> end() const { return single_element_container(nullptr); }
        single_element_container<T>& operator ++() { elem = nullptr; return *this; }
        bool operator !=(single_element_container const& sec) { return elem != sec.elem; }

        T operator *() { return *elem; }

    private:
        single_element_container(T* t) : elem(t)
        {}

        T* elem;
    };

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    //It should be used in generic code when there's a need to force vector usage
    //in foreach loop over variable which can in one context be a vector or a scalar
    //in another.
    //example:
    // T t;
    // for (auto& string : wrap_if_single(t.dump()))
    //depending on type T, t.dump() may return either std::string or std::vector<std::string>,
    //to ensure compatibility between these cases, wrap_if_single will create single-element
    //container in case t.dump() would return plain std::string.
    //
    // T& case -> returns container which holds T&
    template <class T>
    single_element_container<T> wrap_if_single(T& t)
    {
        return single_element_container<T>(t);
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // T const& case -> returns container which holds T const&
    template <class T>
    single_element_container<T const> wrap_if_single(T const& t)
    {
        return single_element_container<T const>(t);
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // T&& case -> returns container which holds new instance of T created by moving given param
    template <class T>
    single_element_container<T> wrap_if_single(T&& t)
    {
        static_assert(meta::always_false<T>::value, "Wrapping temporary object into single_element_container is an error (requires valid reference)");
        return single_element_container<T>(t);
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // std::vector case -> does not wrap, returns t as-is
    primitive::fixed_size_vector_ref const& wrap_if_single(primitive::fixed_size_vector_ref const& t)
    {
        return t;
    }

    //helper function for merging the weights/biases buffers on cpu side for depthwise separable convolution optimization
    void merge_buffers(engine_impl::ptr engine, program_node &node, layout target_layout, size_t begin_offset, size_t end_offset)
    {
        memory_impl::ptr data_to_allocate = engine->allocate_memory(target_layout);

        for (size_t i = begin_offset; i < end_offset; i++)
        {
            auto& weights = node.get_dependency(i).as<data>();
            mem_lock<char> src{ weights.get_attached_memory() };
            mem_lock<char> dst{ data_to_allocate };
            std::copy(src.begin(), src.end(), dst.begin() + (i - begin_offset)*src.size());
        }

        for (size_t i = 0; i < end_offset - begin_offset - 1; i++)
            node.remove_dependency(begin_offset + 1);

        auto& data_node = node.get_dependency(begin_offset).as<data>();
        data_node.attach_memory(*data_to_allocate, false);
    }

    //helper function for getting target layout used in depthwise sep optimization
    layout get_weights_layout(typed_program_node<cldnn::data> &data_node, int32_t split)
    {
        auto mem_layout = data_node.get_output_layout();

        return layout(mem_layout.data_type, mem_layout.format, { split * mem_layout.size.batch[0], mem_layout.size.feature[0], mem_layout.size.spatial[0], mem_layout.size.spatial[1] });
    }

    // pair.first tells whether l1 and l2 are absolutely identical
    // pair.second tells whether l1 and l2 can be reinterpreted to each other without need of reordering
    // note: layouts can only be considered identical if data size described by both layouts match (so no data are genereted nor dropped)
    // note: if layouts describe two buffers with different size, consider them not to be identical even if smaller buffer can be considered to hold subsequence of larger buffer,
    //       this behavior is required to force buffer allocation for smaller buffer which, currently, should always be performed
    std::pair<bool, bool> are_layouts_identical(layout const& l1, layout const& l2)
    {
        if (l1 == l2)
            return{ true, true };
        if (l1.data_type != l2.data_type)
            return{ false, false };
        if (l1.size != l2.size)
            return{ false, false };
        if (l1.get_linear_size() != l2.get_linear_size())
            return{ false, false };
        if ((l1.format == format::bf8_xy16 && l2.format != format::bf8_xy16) ||
            (l2.format == format::bf8_xy16 && l1.format != format::bf8_xy16))
            return{ false, false };

        auto l1_pitch = l1.get_pitches();
        auto l2_pitch = l2.get_pitches();

        //ignore pitches which will never be used (for dims with size == 1)
        for (size_t i = 0; i < CLDNN_TENSOR_DIM_MAX; ++i)
            if (l1.size.raw[i] == 1)
                l1_pitch.raw[i] = 0;
        for (size_t i = 0; i < CLDNN_TENSOR_DIM_MAX; ++i)
            if (l2.size.raw[i] == 1)
                l2_pitch.raw[i] = 0;

        auto l1_offset = l1.get_linear_offset();
        auto l2_offset = l2.get_linear_offset();
        if (l1_pitch == l2_pitch && l1_offset == l2_offset)
            return{ false, true };

        return{ false, false };
    }
}

program_impl::program_impl(engine_impl& engine_ref, topology_impl const& topology, build_options const& options, bool is_internal)
    : engine(&engine_ref), options(options), output_size_handling_enabled(true)
{
    static std::atomic<uint32_t> id_gen{ 0 };
    prog_id = ++id_gen;
    assert(prog_id != 0);

    if ((options.get<build_option_type::tuning_config>()->config.mode == tuning_mode::tuning_tune_and_cache) &&
        !engine->configuration().enable_profiling)
    {
        throw std::invalid_argument("Engine must be created with profiling enabled in tune_and_cache mode!");
    }

    init_graph(topology);
    pre_optimize_graph();
    compile_graph();
    post_optimize_graph();

    engine->compile_program(*this);
    this->dump_program("13_finished", true);

    //Makes serialization with given name.
    //Placeholder, not working yet, in progress.
    auto serialization_network_name = get_serialization_network_name(options);
    if (!serialization_network_name.empty() && !is_internal)
    {
        this->serialize(serialization_network_name);
    }

    cleanup();
}

// TODO: Remove once we will get full support for input/output padding in all primitive implementations.
void program_impl::analyze_output_size_handling_need()
{
    bool handling_needed = false;

    // Calculate output size and compare with specified.
    for (const auto& node : processing_order)
    {
        if (node->is_type<convolution>())
        {
            auto& prim_node = node->as<convolution>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range({ 0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1] }, 1);

            auto filter_size = prim_node.weights(0).get_output_layout().size;

            auto calc_output_range = calc_sliding_window_output_range<swor_mode::all>(
                prim_node.input().get_output_layout().size,
                filter_size, prim->input_offset, prim->stride, prim->dilation, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
        else if (node->is_type<deconvolution>())
        {
            auto& prim_node = node->as<deconvolution>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range({ 0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1] }, 1);

            auto filter_size = prim_node.weights(0).get_output_layout().size;

            auto calc_output_range = calc_sliding_window_needed_input_range(
                prim_node.input().get_output_layout().size,
                filter_size, prim->input_offset, prim->stride, { 1, 1, 1, 1 }, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
        else if (node->is_type<pooling>())
        {
            auto& prim_node = node->as<pooling>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range({ 0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1] }, 1);

            // TODO: Check compatibility of output size calculation (with caffe).
            auto calc_output_range = calc_sliding_window_output_range<swor_mode::exceed_once_data>(
                prim_node.input().get_output_layout().size,
                prim->size, prim->input_offset, prim->stride, { 1, 1, 1, 1 }, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
    }

    output_size_handling_enabled = handling_needed;
}

std::list<std::shared_ptr<program_node>> program_impl::get_nodes() const
{
    std::list<std::shared_ptr<program_node>> ret;

    for (auto& node : processing_order)
        ret.push_back(nodes_map.at(node->id()));
    return ret;
}

void program_impl::init_graph(topology_impl const& topology)
{
    auto const& topo_map = topology.get_primitives();
    for (auto const& prim : topo_map)
    {
        auto& n = get_or_create(prim.second);
        inputs.push_back(&n);
    }
    replace_nodes_pre();

    for (auto itr = inputs.begin(); itr != inputs.end(); )
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);
        auto deps = node->get_primitive()->dependencies();
        if (deps.empty())
            continue;

        //add pointers to node's dependencies
        for (auto& dep : deps)
        {
            try {
                auto dep_node = nodes_map.at(dep);
                node->dependencies.push_back(dep_node.get());
                dep_node->users.push_back(node);
            }
            catch (...) {
                throw std::runtime_error("Program doesn't contain primitive: " + dep +
                    " that is input to: " + node->get_primitive()->id);
            }
        }

        //primitive has dependencies so remove it from 'inputs'
        inputs.erase(node_itr);
    }

    replace_nodes_post();
    handle_lstm();
    set_outputs();
    calc_processing_order();

    dump_program("0_init", true);

    calc_prior_boxes(); dump_program("1_calculated_prior_boxes", true);
    mark_constants();
    mark_data_flow();
    dump_program("2_analyzed_graph", true);
}

void program_impl::pre_optimize_graph()
{
    trim_to_outputs(); dump_program("3_trimmed", true);
    calculate_BFS_processing_order();
    analyze_output_size_handling_need();
    for (auto& node : processing_order)
    {
        if (!node->is_type<internal_primitive>() && !node->is_type<data>())
            node->get_output_layout();
    }

    if (options.get<build_option_type::optimize_data>()->enabled())
    {
        prepare_primitive_fusing();
        layout_optimizer lo(output_size_handling_enabled);
        reorder_inputs(lo);
        // this code should move to post compilation after kernel selector will support handling reorder bias
        pre_optimize_bias(lo);
        dump_program("4_reordered_inputs", true);
    }

    handle_reshape();
    remove_redundant_reorders(); dump_program("5_removed_redundant_reorders", true);
    prepare_padding();
    prepare_depthwise_sep_opt();

    propagate_constants(); dump_program("6_propagated_constants", true);

    //try to fuse buffers (i.e. depth_concat in bfyx format) after padding calculations
    if (options.get<build_option_type::optimize_data>()->enabled())
    {
        prepare_buffer_fusing();
    }

    dump_program("7_pre_optimized", true);
}

void program_impl::compile_graph()
{
    for (auto& node : processing_order)
    {
        if (!node->is_type<internal_primitive>() && !node->is_type<data>())
        {
            node->get_output_layout();
            if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty()))
                node->selected_impl = node->type()->choose_impl(*engine, *node);
        }
    }

    dump_program("8_compiled", true);
}

void program_impl::post_optimize_graph()
{
    layout_optimizer lo;
    post_optimize_weights(lo); dump_program("9_reordered_weights", true);
    remove_redundant_reorders(); dump_program("10_removed_redundant_reorders", true); //TODO: do we need it at this place also?
    propagate_constants(); dump_program("11_propagated_constants", true);
    prep_opt_depthwise_sep_post();
    update_processing_numbers(); dump_program("12_validated_processing_order", true);
    prepare_memory_dependencies();
}

void program_impl::cleanup()
{
    for (auto& node : processing_order)
        if (!node->is_type<internal_primitive>())
            node->get_output_layout();

    //in debug build, at the end, mark all nodes as outputs so user can query for buffers of all not-optimized nodes, including internal ones etc.
    if (is_debug_build())
    {
        for (auto& node : processing_order)
        {
            if (!node->is_output())
            {
                node->set_output(true);
                outputs.push_back(node);
            }
        }
    }
}

std::string get_id_string(size_t i) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << i;
    return ss.str();
}

void program_impl::replace_nodes_pre()
{
    auto itr = nodes_map.begin();
    while (itr != nodes_map.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;

        //find split primitives and create crop primitives out of them
        if (node->is_type<split>())
        {
            auto split_prim = node->as<split>().typed_desc();
            primitive_id input_id = split_prim->input[0];
            auto split_num = split_prim->output_offsets.size();

            //create crop for each split ouptut provided
            for (decltype(split_num) i = 0; i < split_num; i++)
            {
                primitive_id output_id = node->id() + ":" + split_prim->output_ids[i];

                //create dummy crop primitive and add it to nodes map
                auto crop_prim = std::make_shared<crop>(output_id, input_id, tensor{ 1,1,1,1 }, split_prim->output_offsets[i]);
                get_or_create(crop_prim);
            }
        }
    }
}


void program_impl::replace_nodes_post()
{
    auto itr = nodes_map.begin(); //note we need to use iterators since currently processed element can be removed
    while (itr != nodes_map.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;

        //find split primitives and create crop primitives out of them
        if (node->is_type<split>())
        {
            //check if split is not used by any primitive, as it will be optimized
            if (node->get_users().size() != 0)
                throw std::logic_error("Split layer cannot be used directly! Please use split output \"" + node->id() + ":<split_output_id>\"!");

            //get_output size and validate split primitive inputs
            auto output_layout = node->get_output_layout();
            auto output_layout_size = output_layout.size;

            auto split_prim = node->as<split>().typed_desc();
            primitive_id input_id = split_prim->input[0];
            auto split_num = split_prim->output_offsets.size();

            //create crop for each split ouptut provided
            for (decltype(split_num) i = 0; i < split_num; i++)
            {
                primitive_id output_id = node->id() + ":" + split_prim->output_ids[i];

                auto node_ptr = nodes_map.find(output_id)->second;

                //calculate crop reference input size
                tensor reference_input_size;

                for (decltype(split_num) j = 0; j < i; j++)
                    reference_input_size += split_prim->output_offsets[j + 1] - split_prim->output_offsets[j];

                for (decltype(split_num) j = i; j < split_num - 1; j++)
                    reference_input_size += split_prim->output_offsets[j + 1] - split_prim->output_offsets[j];

                reference_input_size = output_layout_size - reference_input_size;

                //update crop primitive and add connections
                node_ptr->set_output_padding(output_layout.data_padding);
                auto crop_prim = node_ptr->as<crop>().typed_desc();
                crop_prim->reference_input = reference_input_size;

                add_connection(node->get_dependency(0), *node_ptr);
            }

            //remove input->split connection and remove original split node
            remove_connection(node->get_dependency(0), *node);
            optimized_out.push_back(node->id());
            nodes_map.erase(node->id());
            continue;
        }

        //find upsampling primitives with bilinear filtering and create deconvolution with proper weights instead
        if (node->is_type<upsampling>())
        {
            auto upsampling_prim = node->as<upsampling>().typed_desc();

            if (upsampling_prim->sample_type != upsampling_sample_type::bilinear)
                continue;

            //check if num_filter is not 0 (required for bilinear upsampling)
            if (upsampling_prim->num_filter == 0)
                throw std::logic_error("num_filter in upsampling cannot be 0 in bilinear filtering mode in \"" + node->id() + "\"!");

            primitive_id upsampling_id = node->id();
            auto& input_node = node->get_dependency(0);

            primitive_id input_id = upsampling_prim->input[0];
            auto num_filter = upsampling_prim->num_filter;

            //setting deconvolution parameters based on upsampling input
            auto scale = static_cast<tensor::value_type>(upsampling_prim->scale);
            tensor stride(1, 1, scale, scale);
            auto offset = static_cast<tensor::value_type>(std::ceil((scale - 1) / 2.f));
            tensor input_offset(0, 0, -offset, -offset);

            //setting weights for deconvolution
            auto kernel_size = static_cast<tensor::value_type>((2 * scale) - (scale % 2));
            layout weights_layout(data_types::f32, format::bfyx, tensor(1, 1, kernel_size, kernel_size));

            std::vector<primitive_id> weights_vec;
            for (uint32_t weights_idx = 0; weights_idx < num_filter; weights_idx++)
            {
                memory_impl::ptr data_to_allocate = engine->allocate_memory(weights_layout);
                mem_lock<float> dst{ data_to_allocate };
                float *dst_data = dst.data();
                //initialize with bilinear weights data
                auto f = static_cast<uint32_t>(std::ceil(kernel_size / 2.0f));
                float c = (2 * f - 1 - f % 2) / (2.f * f);
                float x = 0.f;
                float y = 0.f;
                for (size_t i = 0; i < weights_layout.count(); ++i) {
                    x = static_cast<float>(i % kernel_size);
                    y = static_cast<float>((i / kernel_size) % kernel_size);
                    dst_data[i] = (1 - std::abs(x / f - c)) * (1 - std::abs(y / f - c));
                }

                //create weights primitive, with dummy memory which will be replaced in firther step
                primitive_id weights_id = upsampling_id + "_deconvolution_weights" + std::to_string(weights_idx);
                layout dummy_layout(data_types::f32, format::bfyx, tensor(1, 1, 1, 1));
                float zero = 0.f;
                auto weights_prim = std::make_shared<data>(weights_id, memory::attach(dummy_layout, &zero, 1));
                get_or_create(weights_prim);

                weights_vec.push_back(weights_id);

                auto weights_node_ptr = nodes_map.find(weights_id)->second;

                //attach weights buffer
                auto& data_node = weights_node_ptr->as<data>();
                data_node.attach_memory(*data_to_allocate, false);
            }

            //remove upsampling node, rename it and move to the optimized list
            remove_connection(node->get_dependency(0), *node);
            auto rename_id = upsampling_id + "_tmp";
            rename(*node, rename_id);

            //create deconvolution primitive
            auto deconv_prim = std::make_shared<deconvolution>(upsampling_id, input_id, weights_vec, stride, input_offset);
            get_or_create(deconv_prim);

            auto deconv_node_ptr = nodes_map.find(upsampling_id)->second;

            auto upsampling_node_ptr = nodes_map.find(rename_id)->second;
            replace_all_usages(*upsampling_node_ptr, *deconv_node_ptr);
            optimized_out.push_back(rename_id);
            nodes_map.erase(rename_id);

            //add connections input->deconvolution and weights->deconvolution
            add_connection(input_node, *deconv_node_ptr);

            for (uint32_t weights_idx = 0; weights_idx < num_filter; weights_idx++)
            {
                auto weights_node_ptr = nodes_map.find(weights_vec[weights_idx])->second;
                add_connection(*weights_node_ptr, *deconv_node_ptr);
            }
            continue;
        }

        //find deconvolution primitives with stride 1 and change them to convolution with trasposed weights
        if (node->is_type<deconvolution>())
        {
            if (!options.get<build_option_type::optimize_data>()->enabled())
                continue;

            auto deconv_prim = node->as<deconvolution>().typed_desc();

            //limit optimization to stride = 1
            if (deconv_prim->stride.spatial[0] != 1 || deconv_prim->stride.spatial[1] != 1 || deconv_prim->gradient())
                continue;

            primitive_id deconv_id = node->id();
            auto& input_node = node->get_dependency(0);

            primitive_id input_id = deconv_prim->input[0];

            //setting convolution parameters based on deconvolution params
            auto stride = deconv_prim->stride;
            auto weights = deconv_prim->weights;
            std::vector<primitive_id> weights_vec;
            for (auto& weights_id : weights)
                weights_vec.push_back(weights_id);
            auto biases = deconv_prim->bias;
            std::vector<primitive_id> bias_vec;
            for (auto& bias_id : biases)
                bias_vec.push_back(bias_id);
            auto input_offset = deconv_prim->input_offset;
            auto with_activation = deconv_prim->with_activation;
            auto activation_negative_slope = deconv_prim->activation_negative_slope;
            auto output_padding = deconv_prim->output_padding;

            //remove deconvolution node and its connections to weights and biases, rename it and move to the optimized list
            tensor filter_size = { 1, 1, 1, 1 };
            remove_connection(node->get_dependency(0), *node);
            for (auto& weights_id : weights_vec)
            {
                auto weights_node_ptr = nodes_map.find(weights_id)->second;
                remove_connection(*weights_node_ptr, *node);
                //get filter spatial sizes for input offset adjustment, perform this only once as all filters shouls have same size
                if (weights_id == weights_vec[0])
                    filter_size = weights_node_ptr->get_output_layout().size;
            }

            input_offset.spatial[0] = std::abs(input_offset.spatial[0]) - (filter_size.spatial[0] - 1);
            input_offset.spatial[1] = std::abs(input_offset.spatial[1]) - (filter_size.spatial[1] - 1);

            if (!bias_vec.empty())
            {
                for (auto& bias_id : bias_vec)
                {
                    auto bias_id_node_ptr = nodes_map.find(bias_id)->second;
                    remove_connection(*bias_id_node_ptr, *node);
                }
            }
            auto rename_id = deconv_id + "_tmp";
            rename(*node, rename_id);

            //create convolution primitive
            if (biases.size() != 0)
            {
                auto conv_prim = std::make_shared<convolution>(deconv_id, input_id, weights_vec, bias_vec,
                    stride, input_offset, tensor{ 1, 1, 1, 1 }, with_activation, activation_negative_slope, output_padding);
                get_or_create(conv_prim);
            }
            else
            {
                auto conv_prim = std::make_shared<convolution>(deconv_id, input_id, weights_vec,
                    stride, input_offset, tensor{ 1, 1, 1, 1 }, with_activation, activation_negative_slope, output_padding);
                get_or_create(conv_prim);
            }

            auto conv_node_ptr = nodes_map.find(deconv_id)->second;
            auto conv_node = &conv_node_ptr->as<convolution>();
            conv_node->set_transposed(true);

            //add connections input->convolution, weights->convolution and bias->convolution
            add_connection(input_node, *conv_node_ptr);

            for (auto& weights_id : weights_vec)
            {
                auto weights_node_ptr = nodes_map.find(weights_id)->second;
                add_connection(*weights_node_ptr, *conv_node_ptr);
            }

            if (!bias_vec.empty())
            {
                for (auto& bias_id : bias_vec)
                {
                    auto bias_id_node_ptr = nodes_map.find(bias_id)->second;
                    add_connection(*bias_id_node_ptr, *conv_node_ptr);
                }
            }

            auto deconv_node_ptr = nodes_map.find(rename_id)->second;
            replace_all_usages(*deconv_node_ptr, *conv_node_ptr);
            optimized_out.push_back(rename_id);
            nodes_map.erase(rename_id);

            continue;
        }
    }
}

void program_impl::handle_lstm()
{
    bool has_lstm_children;
    auto itr = nodes_map.begin(); //note we need to use iterators since currently processed element can be removed
    while (itr != nodes_map.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;
        has_lstm_children = false;
        // replace lstm node with lstm_gemm and lstm_elt nodes
        if (node->is_type<lstm>()) {
            bool initial_hidden_term = node->as<lstm>().initial_hidden_term();
            bool initial_cell_term = node->as<lstm>().initial_cell_term();
            bool bias_term = node->as<lstm>().bias_term();
            auto lstm_prim = node->as<lstm>().typed_desc();
            primitive_id weights_id = lstm_prim->weights;
            primitive_id recurrent_id = lstm_prim->recurrent;
            primitive_id bias_id = bias_term ? lstm_prim->bias : "";
            primitive_id initial_hidden_id = initial_hidden_term ? lstm_prim->initial_hidden : "";
            primitive_id initial_cell_id = initial_cell_term ? lstm_prim->initial_cell : "";
            //removing connection with weights to get proper dependency order for next operations
            remove_connection(*nodes_map.at(weights_id), *node);
            remove_connection(*nodes_map.at(recurrent_id), *node);
            if (bias_term)
                remove_connection(*nodes_map.at(bias_id), *node);
            if (initial_hidden_term)
                remove_connection(*nodes_map.at(initial_hidden_id), *node);
            if (initial_cell_term)
                remove_connection(*nodes_map.at(initial_cell_id), *node);

            //calculating sizes
            auto input_size = node->get_dependency(0).get_output_layout().size;
            auto recurrent_size = nodes_map.at(recurrent_id)->get_output_layout().size;
            auto hidden_size = tensor(input_size.batch[0], 1, recurrent_size.spatial[0], input_size.feature[0]);
            size_t directions = recurrent_size.feature[0];
            size_t input_dependencies = node->get_dependencies().size();
            size_t sequence_len = node->as<lstm>().sequence_len();

            //if the sequence has a single element but it has multiple inputs then
            //the parent of this lstm is an lstm node. If this is a bidirectional lstm
            //then the sequence length is the number of dependencies divided by 2.
            if (sequence_len == 1 && input_dependencies > 1)
                sequence_len = (directions == 1) ? input_dependencies : input_dependencies / 2;

            //check if this lstm node has an lstm child
            for (auto& user : node->get_users())
            {
                if (user->is_type<lstm>())
                {
                    has_lstm_children = true;
                }
            }

            std::vector<program_node*> cell_list(directions * sequence_len);
            std::vector<program_node*> concat_depends(directions * sequence_len);
            std::vector<primitive_id> output_ids_offsets(directions * sequence_len);

            primitive_id hidden_fwd_id = initial_hidden_id;
            primitive_id hidden_bwd_id = initial_hidden_id;
            primitive_id cell_fwd_id = initial_cell_id;
            primitive_id cell_bwd_id = initial_cell_id;

            auto split_direction = [&](const std::string gate, bool initial_term, primitive_id& fwd_id, primitive_id& bwd_id) {
                if (initial_term) {
                    primitive_id initial_id = fwd_id;
                    fwd_id = node->id() + ":" + gate + "_fwd";
                    auto fwd_node = std::make_shared<crop>(fwd_id, initial_id, hidden_size, tensor{ 0,0,0,0 });
                    auto &n1 = get_or_create(fwd_node);
                    add_connection(*nodes_map.at(initial_id), n1);
                    bwd_id = node->id() + ":" + gate + "_bwd";
                    auto bwd_node = std::make_shared<crop>(bwd_id, initial_id, hidden_size, tensor{ 0,1,0,0 });
                    auto &n2 = get_or_create(bwd_node);
                    add_connection(*nodes_map.at(initial_id), n2);
                }
            };

            //if bidirectional lstm then initial_hidden and initial_cell terms need to be split
            if (directions > 1) {
                split_direction("hidden", initial_hidden_term, hidden_fwd_id, hidden_bwd_id);
                split_direction("cell", initial_cell_term, cell_fwd_id, cell_bwd_id);
            }

            //lstm expanding
            for (size_t dir = 0; dir < directions; ++dir) {
                auto hidden_id = dir == 0 ? hidden_fwd_id : hidden_bwd_id;
                auto cell_id = dir == 0 ? cell_fwd_id : cell_bwd_id;
                for (size_t i = 0; i < sequence_len; ++i) {
                    size_t idx = i + dir * sequence_len;
                    primitive_id lstm_gemm_id = node->id() + ":lstm_gemm" + get_id_string(idx);
                    primitive_id lstm_elt_id = node->id() + ":lstm_elt" + get_id_string(idx);
                    primitive_id crop_id = node->id() + ":crop" + get_id_string(idx);

                    size_t input_idx = i;
                    //for bidirectional lstms, if first LSTM layer then reverse input
                    //for subsequent stacked layers the input is strided on the dir dimension
                    if (directions > 0) {
                        if (input_dependencies > sequence_len) { // stacked layer
                            input_idx = dir * sequence_len + i;
                        }
                        else
                        {
                            if (dir > 0) { // first layer
                                input_idx = sequence_len - i - 1;
                            }
                        }
                    }
                    primitive_id lstm_gemm_input_id = node->get_dependency(input_idx).get_org_primitive_id();

                    auto lstm_gemm_node = std::make_shared<lstm_gemm>(lstm_gemm_id, lstm_gemm_input_id, weights_id, recurrent_id, bias_id, hidden_id, (uint32_t)dir);
                    auto &n1 = get_or_create(lstm_gemm_node);

                    auto lstm_elt_node = std::make_shared<lstm_elt>(lstm_elt_id, lstm_gemm_id, cell_id, lstm_prim->clip, lstm_prim->input_forget,
                        lstm_prim->activations, lstm_prim->activation_params, lstm_prim->offset_order);
                    auto &n2 = get_or_create(lstm_elt_node);
                    //adding lstm_elt as user
                    add_connection(n1, n2);
                    //adding dependecy to lstm_gemm node
                    //input
                    add_connection(node->get_dependency(input_idx), n1);
                    //adding weights and initial values to lstm_gemm
                    add_connection(*nodes_map.at(weights_id), n1);
                    add_connection(*nodes_map.at(recurrent_id), n1);
                    if (bias_term)
                        add_connection(*nodes_map.at(bias_id), n1);

                    //adding cell and hiddens as dependencies
                    if (i > 0)
                    {
                        add_connection(*cell_list[size_t(i - 1) * directions + dir], n2);
                        add_connection(*(concat_depends[size_t(i - 1) * directions + dir]), n1);
                    }
                    //if initial values are present
                    else
                    {
                        if (initial_hidden_term)
                            add_connection(*nodes_map.at(hidden_id), n1);
                        if (initial_cell_term)
                            add_connection(*nodes_map.at(cell_id), n2);
                    }

                    //lstm_hidden
                    hidden_id = crop_id + ":hidden";
                    auto crop_hidden = std::make_shared<crop>(hidden_id, lstm_elt_id, hidden_size, tensor{ 0,0,0,0 });
                    auto &n3 = get_or_create(crop_hidden);
                    //adding eltwise as dependency to hidden
                    add_connection(n2, n3);

                    //if parent is lstm adding hiddens as dependency
                    if (has_lstm_children)
                    {
                        for (auto& user : node->get_users())
                        {
                            add_connection(n3, *user);
                        }
                    }
                    concat_depends[i * directions + dir] = &n3;

                    //lstm_cell
                    if (i < sequence_len - 1) {
                        cell_id = crop_id + ":cell";
                        auto crop_cell = std::make_shared<crop>(cell_id, lstm_elt_id, hidden_size, tensor{ 0,1,0,0 });
                        auto &n4 = get_or_create(crop_cell);
                        add_connection(n2, n4);
                        cell_list[i * directions + dir] = &n4;
                    }
                    output_ids_offsets[i * directions + dir] = hidden_id;
                }
            }

            //if there is no next lstm, concatenation is created
            if (!has_lstm_children)
            {
                primitive_id original_id = node->id();
                primitive_id concatenation_id = original_id + ":concat";
                auto concatenation_primitive = std::make_shared<concatenation>(concatenation_id, output_ids_offsets, concatenation::along_f);
                auto &concatenation_node = get_or_create(concatenation_primitive);
                for (auto sub_dependency : concat_depends)
                {
                    add_connection(*sub_dependency, concatenation_node);
                }
                if (directions == 2) {
                    // bidirectional support requires concatenations along the direction and sequence axis
                    // instead we can concatenate along the sequence axis and reshape the tensor to the account
                    // for the direction
                    tensor output_size {input_size.batch[0], (int32_t)sequence_len, hidden_size.spatial[0], (int32_t)directions};
                    primitive_id reshape_id = original_id + ":reshape";
                    auto reshape_primitive = std::make_shared<reshape>(reshape_id, concatenation_id, output_size);
                    auto &reshape_node = get_or_create(reshape_primitive);
                    add_connection(concatenation_node, reshape_node);
                    for (auto& user : node->get_users())
                    {
                        add_connection(reshape_node, *user);
                    }
                }
            }

            //removing expanded node
            remove_all_connections(*node);
            nodes_map.erase(node->id());
            continue;
        }
    }

}

void program_impl::set_outputs()
{
    auto outputs_option = options.get<build_option_type::outputs>();
    if (!outputs_option->outputs.empty())
    {
        for (auto const& output : outputs_option->outputs)
        {
            auto o_node = nodes_map.at(output);
            o_node->set_output(true);
            outputs.push_back(o_node.get());
        }
    }
    else
    {
        for (auto& node : nodes_map)
            if (node.second->is_endpoint())
            {
                node.second->set_output(true);
                outputs.push_back(node.second.get());
            }
    }
}

void program_impl::calc_processing_order()
{
    processing_order.clear();

    //run dfs to sort nodes topologically
    for (auto input : inputs)
    {
        if (input->is_marked())
            continue;

        input->mark();
        std::list<std::pair<program_node*, std::list<program_node*>::const_iterator>> stack = { std::make_pair(input, input->users.begin()) };

        while (!stack.empty()) //imitate call stack
        {
        new_frame:
            auto& frame = stack.back();

            while (frame.second != frame.first->users.end())
            {
                auto successor = *frame.second;
                ++frame.second;

                if (!successor->is_marked())
                {
                    successor->mark();

                    //recurrence call
                    stack.push_back(std::make_pair(successor, successor->users.begin()));
                    goto new_frame;
                }
            }

            //we have finished processing one node so add it to the processing queue
            processing_order.push_front(frame.first);
            frame.first->processing_itr = processing_order.begin();

            //return from call
            stack.pop_back();
        }
    }

    uint32_t idx = 0;
    for (auto& node : processing_order)
    {
        node->processing_num = ++idx;
        node->unmark();
    }
}

void program_impl::update_processing_numbers()
{
    uint32_t idx = 0;
    for (auto& node : processing_order)
    {
        node->processing_num = ++idx;
    }

    for (auto& node : processing_order)
    {
        if (!processing_order_is_correct(node))
        {
            CLDNN_ERROR_MESSAGE(node->id(), "Incorrect processing order");
            return;
        }
    }
}

void program_impl::calc_prior_boxes()
{
    auto itr = processing_order.begin();
    while (itr != processing_order.end())
    {
        auto& node = (*itr++);
        if (!node->is_type<prior_box>())
            continue;

        auto& pb_node = node->as<prior_box>();

        pb_node.calc_result();
        remove_connection(pb_node.input(), pb_node);

        auto& result = pb_node.get_result_buffer();
        result.add_ref(); // need to inc ref count since we will be assigning this memory as cldnn_memory in next line that is not ref_count_obj
        auto cpp_mem = details::memory_c_to_cpp_converter::convert(api_cast(&result));

        auto& data_node = get_or_create(std::make_shared<data>("_cldnn_tmp_" + pb_node.id() + "_result", cpp_mem));
        replace(pb_node, data_node, false, false);
    }
}



void program_impl::mark_constants()
{
    for (auto& node : processing_order)
    {
        if (node->dependencies.empty())
            continue;
        if (node->is_type<prior_box>())
            continue;

        node->constant = true;
        for (auto& dep : node->get_dependencies())
        {
            if (!dep->constant)
            {
                node->constant = false;
                break;
            }
        }

        if (!node->constant)
            for (auto& dep : node->get_dependencies())
                if (dep->constant)
                    dep->constant_frontier = true;
    }
}

void program_impl::mark_data_flow()
{
    std::list<program_node*> stack;
    for (auto const& node : processing_order)
    {
        if ((node->is_endpoint() && !node->constant) || node->is_type<mutable_data>())
        {
            stack.push_back(node);
            node->data_flow = true;
            node->mark();
        }
    }

    while (!stack.empty())
    {
        auto node = stack.front();
        stack.pop_front();

        size_t dep_idx = 0;
        size_t inputs_count = (node->is_type<internal_primitive>() ? node->get_dependencies().size() : node->get_primitive()->input.size());
        //TODO: remove this hack after addition of constants propagation pass
        if (node->is_type<detection_output>() || node->is_type<proposal>())
            inputs_count = 2; //ignore third input as it is related to prior boxes (i.e. concat of prior-boxes)

        for (auto dep : node->get_dependencies())
        {
            bool data_flow = (dep_idx < inputs_count && !dep->constant);
            ++dep_idx;
            if (!data_flow)
                continue;

            dep->data_flow = data_flow;

            if (dep->is_marked())
                continue;

            stack.push_back(dep);
            dep->mark();
        }
    }

    for (auto& node : processing_order)
    {
        assert(!node->constant || !node->data_flow); //node which is constant cannot be marked as data flow
        node->unmark();
    }
}

void program_impl::trim_to_outputs()
{
    size_t actual_nodes = processing_order.size();
    if (!actual_nodes) //degenerated case but can happen
        return;

    if (outputs.size() == actual_nodes)
        return;

    //do backward bfs starting from all outputs
    std::list<const std::vector<program_node*>*> stack = { &outputs };
    while (!stack.empty())
    {
        auto nodes_list = stack.front();
        stack.pop_front();

        for (auto node : *nodes_list)
        {
            if (!node->is_marked())
            {
                node->mark();
                if (!node->get_dependencies().empty())
                    stack.push_back(&node->get_dependencies());
            }
        }
    }

    //all not-marked nodes should be removed
    std::list<program_node*> to_rem;
    for (auto node : processing_order)
    {
        if (node->is_type<input_layout>()) //input layout may become disconnected during prior boxes calculations so it may have not been marked at this place but we don't want to remove it
            node->mark();
        else if (!node->is_marked())
            to_rem.push_back(node);
    }

    for (auto const& node : to_rem)
    {
        if (node->is_input())
            inputs.remove(node);
        else
        {
            for (auto dep : node->dependencies)
                if (dep->is_marked())
                    dep->users.remove(node);
        }

        for (auto user : node->users)
            if (user->is_marked())
                user->dependencies.erase(std::remove(user->dependencies.begin(), user->dependencies.end(), node), user->dependencies.end());

        optimized_out.push_back(node->id());
        nodes_map.erase(node->id());
    }
}

void add_memory_dependency(program_node* node, program_node* dep)
{
    if (node->can_be_optimized() ||
        !dep->can_be_optimized())
    {
        node->add_memory_dependency(dep->id());
    }
    else
    {
        if (node->id() == dep->id())
        {
            return;
        }
        for (auto subdep : dep->get_dependencies())
        {
            add_memory_dependency(node, subdep);
            add_memory_dependency(subdep, node);
        }
    }
}

void program_impl::basic_memory_dependencies()
{
    auto itr = processing_order.begin();
    std::vector<primitive_id> past_outputs;
    while (itr != processing_order.end())
    {
        auto& node = *itr;
        itr++;

        //data primitive can't be reused
        if (node->is_type<data>())
            continue;

        // add my dependencies to restriction list (can't share input.output buffers)
        for (auto it : node->get_dependencies())
        {
            add_memory_dependency(node, it);
            add_memory_dependency(it, node);
        }

        // Note we iterate over processing order, it means if primitve has processing num greater than any of outputs, this output
        // has to land on the primitve restriction list. Otherwise memory reuse can corrupt final results.
        node->add_memory_dependency(past_outputs);
        // if current node is an output add it to the outputs list after restriction.
        if (node->is_output())
            past_outputs.push_back(node->id());
    }
}

void program_impl::skipped_branch_memory_dependencies()
{
    auto itr = processing_order.begin();
    // Primitive A can't use primitive B buffer if B->processing_num < A->processing_num and any of B users processing_num > A->processing_num
    // Otherwise it could override data that has to be used in the future.
    // TODO: improve algorithm to to O(n*log(n))
    while (itr != processing_order.end())
    {
        auto& node = *itr;
        itr++;
        auto itr2 = processing_order.begin();
        if (itr2 == itr)
            continue;
        while (itr2 != processing_order.end())
        {
            auto& node2 = *itr2;
            itr2++;
            if (node2->get_processing_num() < node->get_processing_num())
            {
                // if at least one user will be processed after 'node', node2 has to be added to forbiden list
                for (auto usr : node2->get_users())
                {
                    if (usr->get_processing_num() > node->get_processing_num())
                    {
                        add_memory_dependency(node, node2);
                        add_memory_dependency(node2, node);
                        break;
                    }
                }
            }
        }
    }
}

void program_impl::oooq_memory_dependencies()
{
    auto itr = processing_order.begin();
    // This order let us build dependencies based on syncing points.
    // Set of nodes between two syncing points will be called sync_region.
    // Major rules is: can't share resource with nodes in my sync_region

    uint32_t last_barrier = 0;
    bool needs_barrier = false;
    std::vector<cldnn::program_node*> sync_region;
    while (itr != processing_order.end())
    {
        auto& node = *itr;
        itr++;

        // if any of dep has proccess num after barrier -> needs barrier
        for (auto dep : node->get_dependencies())
        {
            if (dep->get_processing_num() >= last_barrier)
            {
                needs_barrier = true;
                break;
            }
        }

        if (needs_barrier)
        {
            last_barrier = node->get_processing_num();
            needs_barrier = false;
            // add each pair bi-direction dependency
            for (auto nd1 = sync_region.begin(); nd1 + 1 != sync_region.end(); nd1++)
            {
                for (auto nd2 = nd1 + 1; nd2 != sync_region.end(); nd2++)
                {
                    add_memory_dependency(*nd1, *nd2);
                    add_memory_dependency(*nd2, *nd1);
                }
            }

            // collect dependencies of every node in sync region
            std::vector<cldnn::program_node*> deps;
            for (auto& nd_in_region : sync_region)
                for (auto& dep : nd_in_region->get_dependencies())
                    deps.emplace_back(dep);


            for (auto& nd_in_region : sync_region)
                for (auto& dep : deps)
                {
                    add_memory_dependency(nd_in_region, dep);
                    add_memory_dependency(dep, nd_in_region);
                }

            sync_region.clear();
        }
        sync_region.push_back(node);
    }
}

void program_impl::prepare_memory_dependencies()
{
    if (!get_engine().configuration().enable_memory_pool)
        return;

    basic_memory_dependencies();
    skipped_branch_memory_dependencies();
    oooq_memory_dependencies();
}

std::string program_impl::get_memory_dependencies_string() const
{
    std::string mem_dep = "Memory dependencies/restrictions:\n";
    auto itr = processing_order.begin();
    while (itr != processing_order.end())
    {
        auto& node = *itr;
        itr++;
        mem_dep = mem_dep.append("primitive: ").append(node->id()).append(" restricted list: ");
        for (auto it : node->get_memory_dependencies())
            mem_dep == mem_dep.append(it).append(", ");
        mem_dep = mem_dep.append("\n");
    }
    return mem_dep;
}

void program_impl::remove_redundant_reorders()
{
    auto itr = processing_order.begin(); //note we need to use iterators since currently processed element can be removed
    while (itr != processing_order.end())
    {
        auto& node = (*itr++); //post-inc to avoid invalidation due to possible erase
        if (!node->is_type<reorder>()) //only care for reorders
            continue;

        program_node* current_node = node;
        std::vector<program_node*> r_nodes_to_remove;

        auto optimize = true;
        while (current_node)
        {
            auto& r_node = current_node->as<reorder>();
            current_node = nullptr;

            if (r_node.has_mean() || !r_node.get_primitive()->subtract_per_feature.empty() ||  //do not optimize if mean of subtract are present
                (r_node.is_output() && r_node.get_dependency(0).is_output())) //do not optimize when both reorder and layer before are outputs
            {
                optimize = false;
                break;
            }

            r_nodes_to_remove.push_back(&r_node);

            if (r_node.get_dependency(0).is_type<reorder>() && r_node.get_dependencies().size() == 1 && r_node.get_users().size() == 1 && r_node.get_dependency(0).get_users().size() == 1)
                current_node = &r_node.get_dependency(0);
        }
        if (!optimize)
            continue;

        assert(node->dependencies.size() == 1 && "reorder without mean should have exactly one dependecy (input)");
        auto& r_output = r_nodes_to_remove.front();
        auto& r_input = r_nodes_to_remove.back()->get_dependency(0);
        auto o_layout = r_output->get_output_layout();
        auto i_layout = r_input.get_output_layout();

        auto ident = are_layouts_identical(o_layout, i_layout);
        if (!ident.second)
            continue;

        for (auto remove_reorder_node : r_nodes_to_remove)
        {
            auto& r_node = remove_reorder_node->as<reorder>();

            if (ident.first && ident.second && r_node.is_output() && r_node.get_dependency(0).is_input()) //do not optimize when reorder is output and layer before is input
            {
                optimize = false;
                break;
            }
        }
        if (!optimize)
            continue;

        for (auto remove_reorder_node : r_nodes_to_remove)
        {
            auto& r_node = remove_reorder_node->as<reorder>();

            //mark as optimized
            r_node.can_be_optimized(true);
            r_node.requires_reinterpret(!ident.first);
            if (ident.first) //no need of reshape
                extract_and_remove(r_node); //try to remove if possible (with respect to r_node not being marked as output)
        }
    }
}

/*
    recalculate processing_order
    algorithm based on: CLRS 24.5 (critical path in DAG)
    modifications: adjust for multiple inputs
    input: any topological order in processing order
    output: BFS topological order.
*/

void program_impl::calculate_BFS_processing_order() {
    std::map<program_node*, int> distances;
    for (auto itr : processing_order)
    {
        distances[itr] = -1;
    }
    int max_distance = 0;
    for (auto itr : processing_order)
    {
        //Init
        if (distances[itr] == -1) {     // this must be an input
            distances[itr] = 0;         // initialize input
        }
        // RELAX
        for (auto& user : itr->get_users())
        {
            distances[user] = std::max(distances[user], distances[itr] + 1);
            max_distance = std::max(max_distance, distances[user]);
        }
    }

    //bucket sort nodes based on their max distance from input
    std::vector<std::vector<program_node*>> dist_lists;
    dist_lists.resize(max_distance + 1);
    for (auto itr : processing_order)
    {
        dist_lists[distances[itr]].push_back(itr);
    }

    //replace the old processing order by the new one, still topological.
    processing_order.clear();
    for (auto& dist : dist_lists)
    {
        for (auto& node : dist)
        {
            processing_order.push_back(node);
            node->processing_itr = processing_order.end();
            node->processing_itr--;
        }
    }
    update_processing_numbers();
    return;
}

void program_impl::reorder_inputs(layout_optimizer& lo)
{
    //first pass to set layout optimization_attributes for topology
    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;
        if (prim.type() == cldnn::convolution::type_id())
        {
            if (prim.as<convolution>().get_primitive()->split() > 1)
                lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::splitted_convolution, 1);
        }

        //list of layers that do not support yxfb or perform worse than bfyx
        if (prim.type() == cldnn::detection_output::type_id() || prim.type() == cldnn::proposal::type_id() ||
            prim.type() == cldnn::roi_pooling::type_id() || prim.type() == cldnn::deconvolution::type_id() ||
            prim.type() == cldnn::upsampling::type_id() || prim.type() == cldnn::reorg_yolo::type_id())
            lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::bfyx_only_layer, 1);
    }

    const auto reorder_input = [this, &lo](typed_program_node<convolution>& conv_node)
    {
        auto conv_prim = conv_node.get_primitive();
        auto& input_node = conv_node.get_dependency(0);
        auto&& weights_layout = conv_node.weights(0).get_output_layout();
        auto&& input_layout = input_node.get_output_layout();

        std::shared_ptr<reorder> new_input = nullptr;

        if (input_node.type() == reorder::type_id()) //convolution's input is a reorder
        {
            auto reorder_prim = input_node.as<reorder>().typed_desc();
            auto& reorder_input = input_node.get_dependency(0);
            auto reorder_layout = input_node.get_output_layout();
            reorder_layout.data_type = reorder_prim->output_data_type;
            new_input = lo.get_reorder(
                reorder_layout,
                reorder_prim->id,
                layout_optimizer::data_type::input,
                conv_node,
                weights_layout).first;

            auto reorder_removed = false;
            if (new_input && new_input->output_format != format::winograd_2x3_s1_data && new_input->output_format != format::bf8_xy16 && new_input->output_format != format::byxf) //output format is not optimal
            {
                auto reorder_input_layout = reorder_input.get_output_layout();

                auto opt_layout = layout(new_input->output_data_type, new_input->output_format, reorder_input_layout.size);
                if (reorder_input_layout == opt_layout) //reorder 'breaks' optimal format
                {
                    if (reorder_prim->subtract_per_feature.empty() &&
                        reorder_prim->mean.empty() &&
                        !reorder_prim->output_padding) //just plain reorder
                    {
                        conv_node.replace_dependency(0, reorder_input);
                        if (input_node.get_users().size() == 0 && !input_node.is_output())
                        {
                            reorder_removed = extract_and_remove(input_node);
                        }
                        new_input = nullptr;
                    }
                    else //change reorder's output layout
                    {
                        reorder_prim->output_format = opt_layout.format;
                        reorder_prim->output_data_type = opt_layout.data_type;
                        new_input = nullptr;
                    }
                }
                else //current reorder gives bad output, simply change it
                {
                    reorder_prim->output_format = opt_layout.format;
                    reorder_prim->output_data_type = opt_layout.data_type;
                    new_input = nullptr;
                }
            }

            if (!reorder_removed)
                input_node.recalc_output_layout();
            else
                conv_node.recalc_output_layout();
        }
        else
        {
            new_input = lo.get_reorder(
                input_node.get_output_layout(),
                input_node.id(),
                layout_optimizer::data_type::input,
                conv_node,
                weights_layout).first;
        }

        if (new_input && new_input->output_format == format::winograd_2x3_s1_data)
        {
            auto lower_size = (conv_prim->input_offset.negate() + input_layout.size);

            tensor upper_input_padding = tensor{ 0 };
            upper_input_padding.spatial[0] = (2 - (lower_size.spatial[0] % 2)) % 2;          //winograd conv requires input's x to be in form 4 + 2n, with restriction that x >= 3, we can shortage it to x % 2 == 0
            upper_input_padding.spatial[1] = (8 - ((lower_size.spatial[1] - 2) % 8)) % 8;    //for y, y - 2 % 8 == 0 must hold

            apply_needed_padding(conv_node, input_node, padding{ conv_prim->input_offset.negate().sizes(), upper_input_padding.sizes() });

            auto winograd_output = std::make_shared<reorder>("_winograd_" + conv_node.id(), conv_node.id(), input_layout.format,
                input_layout.data_type, std::vector<float>{}, cldnn_reorder_mean_mode::mean_subtract, conv_node.output_layout.data_padding);
            conv_node.output_layout.data_padding = padding{};
            auto& back_node = get_or_create(winograd_output);
            back_node.processing_itr = processing_order.insert(std::next(conv_node.processing_itr), &back_node);

            auto bias_term = conv_node.bias_term();
            //create additional eltwise node after reorder to compute bias
            if (bias_term)
            {
                auto& bias_node = conv_node.get_dependency(2);
                std::vector<primitive_id> inputs = { back_node.id(), bias_node.id() };
                auto winograd_output_biases = std::make_shared<eltwise>(back_node.id() + "_bias", inputs,
                    cldnn::eltwise_mode::sum, conv_prim->with_activation, conv_prim->activation_negative_slope,
                    back_node.output_layout.data_padding);
                back_node.output_layout.data_padding = padding{};
                auto& back_bias_node = get_or_create(winograd_output_biases);
                back_bias_node.processing_itr = processing_order.insert(std::next(back_node.processing_itr), &back_bias_node);
                replace_all_usages(back_node, back_bias_node);
                add_connection(back_node, back_bias_node);
                add_connection(bias_node, back_bias_node);
                conv_node.invalidate_users();
                replace_all_usages(conv_node, back_bias_node);
            }

            if (conv_prim->with_activation)
            {
                conv_node.typed_desc()->with_activation = false;
                if (!bias_term)
                    back_node.set_fused_activation(activation_relu_negative_slope, cldnn_activation_additional_params_t{ conv_prim->activation_negative_slope, 0.0f });
            }

            if (!bias_term)
            {
                conv_node.invalidate_users();
                replace_all_usages(conv_node, back_node);
            }
            add_connection(conv_node, back_node);

            auto& r_node = get_or_create(new_input);
            r_node.as<reorder>().set_input_offset(conv_prim->input_offset);

            if (!bias_term)
            {
                swap_names(conv_node, back_node);
                if (conv_node.is_output())
                {
                    conv_node.set_output(false);
                    back_node.set_output(true);
                    for (auto& output : outputs)
                    {
                        if (output == &conv_node)
                        {
                            output = &back_node;
                            break;
                        }
                    }
                }
            }
            else
            {
                conv_node.remove_dependency(2);
                auto& back_bias_node = *nodes_map.find(back_node.id() + "_bias")->second;
                swap_names(conv_node, back_bias_node);
                if (conv_node.is_output())
                {
                    conv_node.set_output(false);
                    back_bias_node.set_output(true);
                    for (auto& output : outputs)
                    {
                        if (output == &conv_node)
                        {
                            output = &back_bias_node;
                            break;
                        }
                    }
                }
            }
        }

        if (new_input && (new_input->output_format == format::bf8_xy16 || new_input->output_format == format::byxf))
        {
            auto conv1x1_output = std::make_shared<reorder>("_conv1x1_reorder_back_" + conv_node.id(), conv_node.id(), input_layout.format, input_layout.data_type);
            auto& back_node = get_or_create(conv1x1_output);
            back_node.processing_itr = processing_order.insert(std::next(conv_node.processing_itr), &back_node);

            conv_node.invalidate_users();
            replace_all_usages(conv_node, back_node);
            add_connection(conv_node, back_node);
        }

        if (new_input)
        {
            auto& r_node = get_or_create(new_input);
            add_intermediate(r_node, conv_node, 0, r_node.dependencies.empty());
            conv_node.recalc_output_layout();
        }
    };

    const auto reorder_input_detection_output = [this, &lo](typed_program_node<detection_output>& detection_output_node)
    {
        auto detection_output_prim = detection_output_node.get_primitive();

        for (size_t i = 0; i < detection_output_node.get_dependencies().size(); i++)
        {
            auto& input = detection_output_node.get_dependency(i);
            std::shared_ptr<reorder> new_input = lo.get_reorder(
                input.get_output_layout(),
                input.id(),
                layout_optimizer::data_type::input,
                detection_output_node,
                layout{ data_types::f32, format::bfyx, tensor{} }).first;

            if (new_input)
            {
                add_intermediate(new_input, detection_output_node, i);
            }
        }
    };

    for (auto& prim : processing_order)
    {
        //there's an assumption that only convolution will take data/input_layout as input
        //exception to that rule would be a convolution which takes a reorder as input - see reoder_input above
        do_for_types<convolution, detection_output>(*prim,
            reorder_input,                  //case for convolution
            reorder_input_detection_output  //case for detection-output
            );
    }
}

//function which prepares given primitive for weights optimization
template <typename T>
void program_impl::optimize_bias(T& node, layout_optimizer& lo)
{
    layout output_layout = node.get_output_layout();

    size_t weights_offset = node.get_primitive()->input.size();
    size_t bias_offset = weights_offset + wrap_if_single(node.get_primitive()->weights).size();
    for (size_t i = bias_offset; i < node.get_dependencies().size(); ++i)
    {
        //find weights primitive with given pimitive_id and add it to weights_optimizer
        const program_node& bias = node.get_dependency(i);
        const auto bias_type = layout_optimizer::data_type::bias;
        auto reorder = lo.get_reorder(
                bias.get_output_layout(),
                bias.id(),
                bias_type,
                node,
                output_layout);

        if (reorder.first)
            this->add_intermediate(reorder.first, node, i, !reorder.second);
    }
}
template void program_impl::optimize_bias<convolution_node>(convolution_node& node, layout_optimizer& lo);
template void program_impl::optimize_bias<deconvolution_node>(deconvolution_node& node, layout_optimizer& lo);
template void program_impl::optimize_bias<fully_connected_node>(fully_connected_node& node, layout_optimizer& lo);
template void program_impl::optimize_bias<embed_node>(embed_node& node, layout_optimizer& lo);

void program_impl::pre_optimize_bias(layout_optimizer& lo)
{
    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;
        if (prim.type() == convolution::type_id())
        {
            if (!prim.as<convolution>().weights_quantization_term())
                optimize_bias(prim.as<convolution>(), lo);
        }
        else if (prim.type() == deconvolution::type_id())
        {
            optimize_bias(prim.as<deconvolution>(), lo);
        }
        else if (prim.type() == fully_connected::type_id())
        {
            if (!prim.as<fully_connected>().weights_quantization_term())
                optimize_bias(prim.as<fully_connected>(), lo);
        }
        else if (prim.type() == embed::type_id())
        {
            optimize_bias(prim.as<embed>(), lo);
        }
    }
}

template <typename T>
void program_impl::optimize_depthwise_sep_pre(T& node)
{
    //enable optimization only when IFM / split <= 8 (otherwise scheduling multiple opt kernels is better) and split >= 16
    if (!(node.get_dependency(0).get_output_layout().size.feature[0] / node.get_primitive()->split() <= 8) ||
        !(node.get_primitive()->split() >= 16))
        return;

    //make sure the weights and biases are data type and
    //are not reused in other primitives as they will be overriden with concatenated ones
    for (size_t i = 1; i < node.get_dependencies().size(); i++)
    {
        auto& weights_or_biases = node.get_dependency(i);
        if (weights_or_biases.get_users().size() > 1 || weights_or_biases.type() != data::type_id())
            return;
    }

    node.set_depthwise_sep_opt(true);
}
template void program_impl::optimize_depthwise_sep_pre<convolution_node>(convolution_node& node);
template void program_impl::optimize_depthwise_sep_pre<deconvolution_node>(deconvolution_node& node);

void program_impl::prepare_depthwise_sep_opt()
{
    //depthiwise separated convolution/deconvolution optimization
    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;
        if (prim.type() == convolution::type_id())
        {
            optimize_depthwise_sep_pre(prim.as<convolution>());
        }
        else if (prim.type() == deconvolution::type_id())
        {
            optimize_depthwise_sep_pre(prim.as<deconvolution>());
        }
    }
}

void program_impl::handle_reshape()
{
    //reshape primitive by definition does not change underlying data, only shape description
    //however during graph initialization and data optimization the layouts can be changed without user's knowledge,
    //when reshape is followed by reorder, it is likely that reorder's output will not be as expected (for example reshape with flattened shape)
    //this pass resolved the issue by changing graph in the following way
    //- in case reshape has multiple users with reshape->reorder sequence, it will be splitted to multiple reshape primitives with single user
    //- in case of reshape->reorder sequence, the additional reorder before reshape will be added,
    //  if last reorder does not contain padding or mean subtract, it will be removed later in the graph

    for (const auto& node : processing_order)
    {
        if (node->is_type<reshape>())
        {
            auto& input_node = node->get_dependency(0);

            if (input_node.is_type<reorder>())
                continue;

            //vector for storing nodes that are reorder type, for which splitted primitives are needed (except for the first one where orginal reshape will be used)
            std::vector<program_node*> reorder_node_to_split;

            //find the users of reshape that are reorder type, if none present then skip the current node
            for (const auto& user : node->get_users())
            {
                if (user->is_type<reorder>())
                    reorder_node_to_split.push_back(user);
            }

            if (!reorder_node_to_split.empty())
            {
                auto& prim_node = node->as<reshape>();
                const auto& prim = prim_node.get_primitive();
                auto output_shape = prim->output_shape;

                //vector for storing reshape nodes to connect to new reorder nodes (if needed)
                std::vector<program_node*> reorder_reshape_nodes;

                bool skip_first_user = false;
                auto reshape_users = node->get_users();
                for (const auto& user : reshape_users)
                {
                    //reshape node for first user will be the orginal reshape from the graph
                    if (!skip_first_user)
                    {
                        if (std::find(reorder_node_to_split.begin(), reorder_node_to_split.end(), user) != reorder_node_to_split.end())
                            reorder_reshape_nodes.push_back(node);
                        skip_first_user = true;
                        continue;
                    }

                    //other reshapes will be clones of the orginal one connected to reshape->reorder sequences
                    if (std::find(reorder_node_to_split.begin(), reorder_node_to_split.end(), user) != reorder_node_to_split.end())
                    {
                        auto new_reshape = std::make_shared<reshape>("_reshape_split_" + user->id() + "_" + node->id(), input_node.id(), output_shape);
                        auto& new_reshape_node = get_or_create(new_reshape);
                        add_connection(input_node, new_reshape_node);
                        user->replace_dependency(0, new_reshape_node);
                        new_reshape_node.processing_itr = processing_order.insert(std::next(input_node.processing_itr), &new_reshape_node);
                        reorder_reshape_nodes.push_back(&new_reshape_node);
                    }
                }

                //add new reorder nodes to proper reshape node
                auto reshape_reorder_id = 0;
                for (const auto& reorder_node : reorder_node_to_split)
                {
                    auto& reorder_reshape_node = reorder_reshape_nodes[reshape_reorder_id];
                    auto reshape_in_layout = reorder_node->get_output_layout();
                    auto reshape_input = std::make_shared<reorder>("_reshape_input_" + reorder_node->id() + "_" + reorder_reshape_node->id(), input_node.id(), reshape_in_layout.format, reshape_in_layout.data_type);
                    auto& reshape_input_node = get_or_create(reshape_input);
                    add_intermediate(reshape_input_node, *reorder_reshape_node, 0, reshape_input_node.dependencies.empty());
                    reshape_reorder_id++;
                }
            }

            auto reshape_layout = node->get_output_layout();
            if (!(node->is_output()) && (reshape_layout.format != cldnn::format::bfyx))
            {
                auto bfyx_layout = layout({ reshape_layout.data_type, cldnn::format::bfyx, reshape_layout.size });
                //when some primitive does an implicit reorder to some other format then we lose the info about pitches in reshape stage
                //we assume user provides the input vector in bfyx
                if (!are_layouts_identical(reshape_layout, bfyx_layout).second)
                {
                    auto reshape_input = std::make_shared<reorder>("_reshape_input_" + node->id(), input_node.id(), cldnn::format::bfyx, reshape_layout.data_type);
                    auto& reshape_input_node = get_or_create(reshape_input);
                    add_intermediate(reshape_input_node, *node, 0, reshape_input_node.dependencies.empty());

                    auto reshape_users = node->get_users();
                    for (const auto& user : reshape_users)
                    {
                        size_t idx = 0;
                        for (size_t i = 0; i < user->get_dependencies().size(); i++)
                        {
                            auto& input = user->get_dependency(i);
                            if (input.id() == node->id()) {
                                idx = i;
                                break;
                            }
                        }
                        auto reshape_output = std::make_shared<reorder>("_reshape_output_" + node->id(), user->id(), reshape_layout.format, reshape_layout.data_type);
                        auto& reshape_output_node = get_or_create(reshape_output);
                        add_intermediate(reshape_output_node, *user, idx, reshape_output_node.dependencies.empty());
                    }
                }
            }
        }
    }
}

//function which prepares given primitive for weights optimization
template <typename T>
void program_impl::optimize_weights(T& node, layout_optimizer& lo)
{
    auto weights_offset = node.get_primitive()->input.size();
    auto bias_offset = weights_offset + wrap_if_single(node.get_primitive()->weights).size();
    for (auto i = weights_offset; i < bias_offset; i++)
    {
        auto& weights = node.get_dependency(i);
        auto* impl = node.get_selected_impl().get();
        auto output_layout = node.get_output_layout();
        auto& weights_node = node.get_dependency(1);
        auto weights_layout = weights_node.get_output_layout();
        const auto weights_type = layout_optimizer::data_type::weights;

        auto reorders = lo.get_generic_layer(
                impl->_weights_reorder_params,
                weights.id(),
                weights_layout,
                weights_type);

        for (auto& reorder : reorders)
        {
            //insert new generic_layer node to topology
            this->add_intermediate(reorder.first, node, i, !reorder.second);
            //set generic_layer's node output layout and implementation
            auto& g_node = node.get_dependency(i);
            g_node.get_output_layout(false);
            g_node.selected_impl = g_node.type()->choose_impl(*engine, g_node);
        }
        //set the old output layout and do not invalidate users as change of weights will not affect output layout
        node.set_output_layout(output_layout, false);
    }
}
template void program_impl::optimize_weights<convolution_node>(convolution_node& node, layout_optimizer& lo);
template void program_impl::optimize_weights<deconvolution_node>(deconvolution_node& node, layout_optimizer& lo);
template void program_impl::optimize_weights<fully_connected_node>(fully_connected_node& node, layout_optimizer& lo);

void program_impl::post_optimize_weights(layout_optimizer& lo)
{
    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;
        if (prim.type() == convolution::type_id())
        {
            optimize_weights(prim.as<convolution>(), lo);
        }
        else if (prim.type() == deconvolution::type_id())
        {
            optimize_weights(prim.as<deconvolution>(), lo);
        }
        else if (prim.type() == fully_connected::type_id())
        {
            optimize_weights(prim.as<fully_connected>(), lo);
        }
        //else if (prim.type() == lstm_gemm::type_id())//TODO: Enable postoptimize weights for lstm
        //{
        //    prep_opt(prim.as<lstm_gemm>()); //we should take care of weights and reccurent
        //}
    }
}

template <typename T>
void program_impl::optimize_depthwise_sep_post(T& node)
{
    if (!node.get_depthwise_sep_opt())
        return;

    const auto& split = node.get_primitive()->split();

    auto dependency_offset = node.get_primitive()->input.size();
    //concatenate weights
    {
        //if weights were optimized it is needed to use the sizes after optimization
        auto target_layout = get_weights_layout(node.get_dependency(dependency_offset), split);
        merge_buffers(engine, node, target_layout, dependency_offset, dependency_offset + split);
        dependency_offset++;
    }

    //concatenate biases
    if (node.get_primitive()->bias.size() != 0)
    {
        const auto& bias_layout = node.get_dependency(dependency_offset).get_output_layout();
        auto target_layout = layout(bias_layout.data_type, cldnn::format::bfyx, { 1, 1, bias_layout.size.spatial[0] * split, 1 });
        merge_buffers(engine, node, target_layout, dependency_offset, dependency_offset + split);
        dependency_offset++;
    }

    if (node.template is_type<convolution>())
    {
        auto& prim_node = node.template as<convolution>();
        const auto& prim = prim_node.get_primitive();

        // concatenate weights quantization factors
        if (prim->weights_quantization_factors.size() != 0)
        {
            const auto& weights_quantization_layout = node.get_dependency(dependency_offset).get_output_layout();
            auto target_layout = layout(weights_quantization_layout.data_type, cldnn::format::bfyx, { 1, 1, weights_quantization_layout.size.batch[0] * split, 1 });
            merge_buffers(engine, node, target_layout, dependency_offset, dependency_offset + split);
            dependency_offset++;
        }
        // concatenate output callibration factors
        if (prim->output_calibration_factors.size() != 0)
        {
            const auto& output_callibration_layout = node.get_dependency(dependency_offset).get_output_layout();
            auto target_layout = layout(output_callibration_layout.data_type, cldnn::format::bfyx, { 1, 1, output_callibration_layout.size.batch[0] * split, 1 });
            merge_buffers(engine, node, target_layout, dependency_offset, dependency_offset + split);
            dependency_offset++;
        }
    }

    if (node.get_primitive())
        //override node split, as only one kernel will be executed
        node.set_split(1);
}
template void program_impl::optimize_depthwise_sep_post<convolution_node>(convolution_node& node);
template void program_impl::optimize_depthwise_sep_post<deconvolution_node>(deconvolution_node& node);

void program_impl::prep_opt_depthwise_sep_post()
{
    //depthiwise separated convolution/deconvolution optimization
    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;
        if (prim.type() == convolution::type_id())
        {
            optimize_depthwise_sep_post(prim.as<convolution>());
        }
        else if (prim.type() == deconvolution::type_id())
        {
            optimize_depthwise_sep_post(prim.as<deconvolution>());
        }
    }
}

void program_impl::apply_needed_padding(program_node& node, program_node& prev_node,
    const padding& needed_padding)
{
    auto target_layout = prev_node.get_output_layout();

    // Short circuit if padding did not change.
    if (target_layout.data_padding == needed_padding)
        return;

    // Special handling for input nodes.
    if (prev_node.is_type<input_layout>() || prev_node.is_type<mutable_data>())
    {
        target_layout.data_padding = needed_padding;

        auto r_prim = std::make_shared<reorder>("reorder_input_" + node.id(), prev_node.id(), target_layout);
        add_intermediate(r_prim, node, 0);
        return;
    }

    prev_node.merge_output_padding(needed_padding);
}

void program_impl::prepare_padding()
{
    if (output_size_handling_enabled)
    {
        // Prepare upper padding for primitives that support output_size parameter.
        for (const auto& node : processing_order)
        {
            if (node->is_type<convolution>())
            {
                auto& prim_node = node->as<convolution>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                auto filter_size = prim_node.weights(0).get_output_layout().size;

                auto needed_padding = calc_sliding_window_needed_input_padding(
                    prim_node.input().get_output_layout(),
                    prim->output_size, filter_size, prim->input_offset, prim->stride, prim->dilation, false, 1);
                apply_needed_padding(prim_node, prim_node.input(), needed_padding);
            }
            else if (node->is_type<deconvolution>())
            {
                auto& prim_node = node->as<deconvolution>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                auto filter_size = prim_node.weights(0).get_output_layout().size;

                auto needed_padding = calc_sliding_window_needed_input_padding(
                    prim_node.input().get_output_layout(),
                    prim->output_size, filter_size, prim->input_offset, prim->stride, { 1, 1, 1, 1 }, true, 1);

                apply_needed_padding(prim_node, prim_node.input(), needed_padding);
            }
            else if (node->is_type<pooling>())
            {
                auto& prim_node = node->as<pooling>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                // NOTE: Currently there is no pooling implementation/pooling mode which does not check input data range.
                // There is no need to add padding requirements on pooling inputs.
                //auto needed_padding = calc_sliding_window_needed_input_padding(
                //    prim_node.input().get_output_layout(),
                //    prim->output_size, prim->size, prim->input_offset, prim->stride, {1, 1, 1, 1}, false, 1);
                auto needed_padding = prim_node.input().get_output_layout().data_padding;

                apply_needed_padding(prim_node, prim_node.input(), needed_padding);
            }
        }
    }

    // Prepare optimized padding for bfyx convolution.
    for (auto& pair : nodes_map)
    {
        if (pair.second->type() != convolution::type_id())
            continue;

        auto& node = pair.second->as<convolution>();
        if (node.get_dependencies().empty())
            continue;

        auto conv = node.get_primitive();
        auto& conv_input_node = node.get_dependency(0);
        auto conv_layout = node.get_output_layout();

        // right now output padding optimization is only available for bfyx format and data type = float32
        if (conv_layout.format != cldnn::format::bfyx
            && conv_layout.format != cldnn::format::bf8_xy16
            && conv_layout.format != cldnn::format::byxf_af32
            && conv_layout.format != cldnn::format::fs_bs_yx_bsv4_fsv32)
        {
            continue;
        }

        // Calculating input padding needed for convolution
        auto& filter_node = node.as<convolution>().weights(0);
        auto filter_prim = filter_node.get_primitive();

        layout filter_layout = filter_node.get_output_layout();

        // convolution have only one input primitive
        auto prev_prim_output_layout = conv_input_node.get_output_layout();

        // Compute initial required paddings for primitive used as input for convolution.
        auto input_offset = conv->input_offset;
        auto stride = conv->stride;
        auto dilation = conv->dilation;

        auto input_limit_x = input_offset.spatial[0] + (conv_layout.size.spatial[0] - 1) * stride.spatial[0] + (filter_layout.size.spatial[0] - 1) * dilation.spatial[0] + 1;
        auto input_limit_y = input_offset.spatial[1] + (conv_layout.size.spatial[1] - 1) * stride.spatial[1] + (filter_layout.size.spatial[1] - 1) * dilation.spatial[1] + 1;

        auto left_padding = std::max(-input_offset.spatial[0], 0);
        auto top_padding = std::max(-input_offset.spatial[1], 0);
        auto right_padding = std::max(input_limit_x - prev_prim_output_layout.size.spatial[0], 0);
        auto bottom_padding = std::max(input_limit_y - prev_prim_output_layout.size.spatial[1], 0);

        // Adjust right padding, so entire buffer size in X dimension is properly aligned.
        // TODO: NOTE: Will be reenabled with next check-in once heuristic for line-aligned algorithm will be added.
        //auto needed_buffer_size_x = static_cast<cldnn::tensor::value_type>(
        //    round_up_to(left_padding + prev_prim_output_layout.size.spatial[0] + right_padding, 16));
        //right_padding = needed_buffer_size_x - left_padding - prev_prim_output_layout.size.spatial[0];

        cldnn::padding needed_padding({ 0, 0, left_padding, top_padding }, { 0, 0, right_padding, bottom_padding }, 0);
        needed_padding = padding::max(prev_prim_output_layout.data_padding, needed_padding);

        apply_needed_padding(node, conv_input_node, needed_padding);
    }
}

void program_impl::propagate_constants()
{
    constants_propagator prop(this);

    for (auto& node : processing_order)
        prop.visit_node(*node);

    auto&& to_replace = prop.calculate();

    //remove all nodes which are no longer relevant, i.e. nodes which:
    // 1. are constants, and
    // 2. do not have non-const user (so their data are not used during inference), and
    // 3. are not marked as outputs.
    // in case if node has either non-const user or is marked as output, it should be replace with cldnn::data rather than removed (see next loop)
    auto proc_itr = processing_order.begin();
    while (proc_itr != processing_order.end())
    {
        auto& node = (*proc_itr++);
        if (!node->is_constant())
            continue;
        if (node->has_non_const_user() || (node->is_output() && !node->is_type<data>()))
            continue;

        auto& users = node->users;
        auto& deps = node->dependencies;

        for (size_t idx = 0; idx < deps.size(); idx++)
        {
            deps.at(idx)->users.remove(node);
        }
        deps.clear();

        for (auto& usr : users) {
            auto& usr_deps = usr->dependencies;
            usr_deps.erase(std::remove(usr_deps.begin(), usr_deps.end(), node), usr_deps.end());
        }
        users.clear();

        if (!node->is_output())
        {
            auto rem = remove_if_dangling(*node);
            assert(rem && "Non-output constant node which has only constant users should have been removed during constants propagation pass");
            (void)rem;
        }
    }

    //replace all constant nodes which are relevant for inference (either used by non-const user or marked as output) with recomputed cldnn::data
    for (auto& cout : to_replace)
    {
        auto& id_to_replace = cout.first;

        //TODO: do not use API primitives internally and get rid of this last 'cldnn::memory' internal usage
        memory api_memory = details::memory_c_to_cpp_converter::convert(api_cast(cout.second.get()));
        //c-cpp converter does not retain since normally it is done inside API-impl layer (cldnn.cpp) so we need to do it manually
        cout.second->add_ref();

        auto const_data = std::make_shared<data>("_cldnn_const_prop_" + id_to_replace, api_memory /* <<< REMOVE ME WHEN POSSIBLE */);
        auto& new_node = get_or_create(const_data);
        auto& curr_node = *nodes_map.at(id_to_replace);

        if (!curr_node.is_type<generic_layer>())
        {
            auto curr_node_deps = curr_node.get_dependencies();
            for (auto& dep : curr_node_deps)
            {
                auto dep_users = dep->get_users();
                for (auto& dep_user : dep_users)
                {
                    if (dep_user == &curr_node)
                        remove_connection(*dep, curr_node);
                }
            }
        }

        curr_node.dependencies.clear();
        //remove all constant users (as they will be either removed or replaced by cldnn::data which does not have any dependencies)
        curr_node.users.erase(
            std::remove_if(curr_node.users.begin(), curr_node.users.end(), [](program_node* node) { return node->is_constant(); }),
            curr_node.users.end()
        );
        replace(curr_node, new_node, false, false);
    }
}

void program_impl::prepare_buffer_fusing()
{
    bool is_debug = options.get<build_option_type::debug>()->enabled();
    auto itr = processing_order.begin();
    while (itr != processing_order.end())
    {
        auto& node = (*itr++);

        // TODO: Move fused activation to previous layer when possible
        if (node->fused_activation.activation_func != cldnn_activation_func_t::activation_none)
            continue;

        do_for_types<concatenation>(*node, [this, is_debug](concatenation_node& node)
        {
            // buffer fusing should not be performed if one of inputs produces padded output since
            // it could break desired memory alignment. On the other hand, if this node uses all inputs
            // exclusively (see check above) they should not have output padding set since concatenation
            // does not ask for any.
            if (node.has_padded_dependency())
                return;

            auto concat_axis = node.get_primitive()->axis;
            auto padd = node.get_output_layout().data_padding;

            tensor lower_padd = padd.lower_size();
            tensor upper_padd = padd.upper_size();

            auto upper_padd_val = node.get_output_layout().get_buffer_size().raw[concat_axis] - lower_padd.raw[concat_axis];
            tensor lower_padd_offset = lower_padd;

            std::list<std::pair<const std::vector<program_node*>, tensor>> stack = { std::make_pair(node.get_dependencies(), tensor{ 0, 0, 0, 0 }) };
            while (!stack.empty())
            {
                auto nodes_list = stack.front();
                stack.pop_front();

                auto cascade_adjustment = nodes_list.second;
                upper_padd.raw[concat_axis] = upper_padd_val;
                lower_padd = lower_padd_offset;

                //check if concatenation in place can be applied for inputs set
                for (auto input : nodes_list.first)
                {
                    //if any of this node's inputs is used by more than one primitive and is not optimized concatenation then do not fuse buffers,
                    //also, if an input is marked as network output, prevent optimizations which would affect a form of its output (unless debug flag is set)
                    // todo: in future, if this case is problem, it can be optimized further to enable buffer fusing
                    //       per single input rather than all/none
                    // + restrict input types to pooling, convolution and activation only due to problems with output padding on b and f
                    if ((!input->is_type<pooling>() && !input->is_type<convolution>() && !input->is_type<activation>() && !input->is_type<concatenation>() && !input->is_type<crop>() && !input->is_type<scale>()) ||
                        (input->is_output() && !is_debug) ||
                        input->get_users().size() > 2)
                        return;

                    if (input->get_users().size() > 1)
                    {
                        auto user_count = input->get_users().size();
                        for (auto& user : input->get_users())
                            if (user->is_type<concatenation>())
                                user_count--;
                        if (user_count > 1)
                            return;
                    }

                    //check only for spatial paddings. Accept feature and batch
                    if (input->get_output_layout().data_padding.lower_size().spatial[0] != 0 ||
                        input->get_output_layout().data_padding.upper_size().spatial[0] != 0 ||
                        input->get_output_layout().data_padding.lower_size().spatial[1] != 0 ||
                        input->get_output_layout().data_padding.upper_size().spatial[1] != 0)
                        return;
                }

                //apply concatenation in place optimization
                for (auto input : nodes_list.first)
                {
                    auto input_lenght = input->get_output_layout().size.raw[concat_axis];

                    // shrink upper pad so it points at the end of the input's buffer
                    //
                    //   |--- lower padd ---|                    |---------- upper padd -----------|
                    //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
                    upper_padd.raw[concat_axis] -= input_lenght;

                    //adjust padding sizes for cascade concatenations
                    auto lower_padd_tmp = lower_padd;
                    lower_padd_tmp.raw[concat_axis] += cascade_adjustment.raw[concat_axis];
                    auto upper_padd_tmp = upper_padd;
                    upper_padd_tmp.raw[concat_axis] -= cascade_adjustment.raw[concat_axis];

                    // set new padding for input
                    input->set_output_padding(padding(lower_padd_tmp.sizes(), upper_padd_tmp.sizes()));

                    // move lower padd further
                    //
                    //   |-------------- lower padd -------------|---------- upper padd -----------|
                    //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|

                    lower_padd.raw[concat_axis] += input_lenght;

                    if (input->type() == concatenation::type_id() && input->can_be_optimized())
                    {
                        if (input->as<concatenation>().get_primitive()->axis != node.get_primitive()->axis)
                            return;

                        if (!input->get_dependencies().empty())
                            stack.push_back(std::make_pair(input->get_dependencies(), input->get_output_layout().data_padding.lower_size()));
                    }
                }
            }

            node.can_be_optimized(true);
        });

        // zero copy
        do_for_types<crop>(*node, [this, is_debug](crop_node& node)
        {
            //if the node is marked as network output, prevent optimizations which would affect a form of its output, unless debug flag is set
            if (node.is_output() && !is_debug)
                return;

            //do not optimize when next node is concatenation which is not output
            if (node.get_users().size() == 1 && node.get_users().front()->is_type<concatenation>() && !node.get_users().front()->is_output())
                return;

            if (node.get_dependencies().size() == 1 &&
                node.get_users().size() > 0)
            {
                // optimization is avaiable for croping across depth(features) only
                // if output padding has defined padding accross featuers already it wouldn't
                // work because it expect to have zeros in the padded area.
                auto format = node.get_output_layout().format;
                auto crop_prim = node.get_primitive();
                auto input_layout = node.get_dependency(0).get_output_layout();
                auto out_padd = node.get_output_layout().data_padding;
                if (format == format::bfyx &&
                    crop_prim->reference_input.batch[0] == input_layout.size.batch[0] &&
                    crop_prim->reference_input.spatial[0] == input_layout.size.spatial[0] &&
                    crop_prim->reference_input.spatial[1] == input_layout.size.spatial[1] &&
                    out_padd.lower_size().feature[0] == 0 &&
                    out_padd.upper_size().feature[0] == 0 &&
                    out_padd.lower_size().batch[0] == 0 &&
                    out_padd.upper_size().batch[0] == 0 &&
                    out_padd.lower_size().spatial[0] == 0 &&
                    out_padd.lower_size().spatial[1] == 0 &&
                    out_padd.upper_size().spatial[0] == 0 &&
                    out_padd.upper_size().spatial[1] == 0)
                {
                    //  Regular crop
                    //  crop input buffer
                    //  |___________data____________|
                    //
                    //  crop output buffer
                    //  |-------->| offsets[f]  |<--|
                    //            |_____data____|
                    //             <------------>
                    //           reference size
                    //
                    //  Inplace crop
                    //  crop output buffer
                    //  |_low_pad_|__data_size__|___|<-upper pad

                    node.set_output_padding(padding(
                    { out_padd.lower_size().batch[0], crop_prim->offsets.feature[0], out_padd.lower_size().spatial[0], out_padd.lower_size().spatial[1] },
                    { out_padd.upper_size().batch[0], input_layout.size.feature[0] - crop_prim->offsets.feature[0] - crop_prim->reference_input.feature[0],
                        out_padd.upper_size().spatial[0], out_padd.upper_size().spatial[1] }));
                    node.can_be_optimized(true);
                }
            }
        });

        do_for_types<reshape>(*node, [this](reshape_node& node)
        {
            node.get_output_layout();
            if (node.is_in_place() && node.get_fused_activation_func() == activation_none)
                node.can_be_optimized(true);
        });
        do_for_types<reorder>(*node, [this](reorder_node& node)
        {
            auto& input = node.input();
            auto output_layout = node.get_output_layout();
            //This is WA for topologies that due to additional reorders added perform worse with conv1x1 optimization
            auto remove_bf8_xy_opt = ((input.is_type<pooling>() || input.is_type<concatenation>()) &&
                output_layout.format == format::bf8_xy16 && input.get_users().size() == 1);
            //Remove reorder from convolution 1x1 to bfyx in some conditions
            auto remove_byxf_opt = (input.is_type<convolution>() &&
                input.get_users().size() == 1 &&
                input.get_output_layout().format == format::byxf);
            //check if all inputs user have the same format
            auto all_users_same_format = true;
            auto input_user_layout_format = input.get_users().front()->get_output_layout().format;
            for (auto const& user : input.get_users())
            {
                if (user->get_output_layout().format != input_user_layout_format)
                {
                    all_users_same_format = false;
                    break;
                }
            }
            auto same_data_type = input.get_output_layout().data_type == output_layout.data_type;
            //Optimization only available in case of layers that support different input and output formats.
            //todo: new api needs to be created to read such caps
            if (!(input.is_type<pooling>() && (output_layout.format == format::bfyx || output_layout.format == format::yxfb || output_layout.format == format::byxf) && all_users_same_format && same_data_type) &&
                !remove_bf8_xy_opt &&
                !(input.is_type<convolution>() && input.get_output_layout().format == format::bf8_xy16) &&
                !(input.is_type<eltwise>() && (output_layout.format == format::bfyx || output_layout.format == format::yxfb || output_layout.format == format::byxf) && all_users_same_format && same_data_type) &&
                !(remove_byxf_opt && (node.get_users().front()->is_type<eltwise>() || node.get_users().front()->is_type<pooling>())))
                return;

            if (remove_bf8_xy_opt)
            {
                auto users_user_layout = node.get_users().front()->get_users().front()->get_output_layout();
                auto input_layout = input.get_output_layout();
                auto target_layout = layout(input_layout.data_type, users_user_layout.format, input_layout.size, input_layout.data_padding);
                input.set_output_layout(target_layout, false);
            }
            else if (remove_byxf_opt)
            {
                auto user = node.get_users().front();
                auto users_users = node.get_users().front()->get_users();

                for (auto const& users_user : users_users)
                {
                    if (users_user->get_output_layout().format != format::byxf && !users_user->is_type<eltwise>())
                    {
                        remove_byxf_opt = false;
                        break;
                    }
                }

                if (remove_byxf_opt)
                {
                    auto input_layout = input.get_output_layout();
                    user->set_output_layout(input_layout, false);
                }
            }
            else
                input.set_output_layout(output_layout, false);

            node.can_be_optimized(true);
            extract_and_remove(node); //try to remove redundant reorders
        });
    }
}

void program_impl::fuse_skip_layers(program_node* node)
{
    do_for_types<eltwise>(*node, [this](eltwise_node& node)
    {
        bool skippable = false;
        int index = 0;
        if (node.get_primitive()->mode != eltwise_mode::sum || node.inputs_count() != 2)
            return;

        if (node.input(0).is_type<deconvolution>())
        {
            skippable = true;
        }
        else if (node.input(1).is_type<deconvolution>())
        {
            skippable = true;
            index = 1;
        }

        if (!skippable)
            return;

        auto& to_fuse_with = node.input(index).as<deconvolution>();
        int to_fuse_index = index == 0 ? 1 : 0;

        // check that node doesn't have fused eltwise already
        if (to_fuse_with.has_fused_sum())
            return;

        //remove dependencies and users of elwtise that is going to be extracted
        add_connection(node.input(to_fuse_index), to_fuse_with);
        remove_connection(node.input(to_fuse_index), node);

        //replace processing_num of the node where fusing take place and eltwise
        auto new_processing_num = node.processing_num;
        processing_order.erase(to_fuse_with.processing_itr);
        to_fuse_with.processing_itr = processing_order.insert(node.processing_itr, &to_fuse_with);
        to_fuse_with.processing_num = new_processing_num;

        //make sure that new fused node's users have higher processing_num than fused node
        for (auto user : to_fuse_with.get_users())
        {
            if (user->processing_num < new_processing_num)
            {
                processing_order.erase(user->processing_itr);
                user->processing_itr = processing_order.insert(std::next(to_fuse_with.processing_itr), user);
                user->processing_num = new_processing_num + 1;
            }
        }

        if (node.get_fused_activation_func() != activation_none)
            to_fuse_with.set_fused_activation(node.get_fused_activation_func(), node.get_fused_activation_params());
        to_fuse_with.set_output_padding(node.get_output_layout().data_padding);

        extract_and_remove(node);
    });
}

void program_impl::prepare_primitive_fusing()
{
    bool is_debug = options.get<build_option_type::debug>()->enabled();

    auto itr = processing_order.begin(); //note we need to use iterators since currently processed element can be removed
    while (itr != processing_order.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        do_for_types<activation>(*node, [this, is_debug](activation_node& node)
        {

            auto& input = node.input();

            //Restrictions:
            // - inputs cannot be padded
            // - primitives input cannot be output
            // - no activation additional input
            // - input was optimized
            if (node.has_padded_dependency() || (input.is_output() && !is_debug) || node.is_output() ||
                node.get_dependencies().size() != 1 || input.can_be_optimized())
                return;

            // - check if there is no activation fused already
            // - limit to primitives which implementations support activation fusing
            if (input.get_users().size() != 1 || input.get_fused_activation_func() != activation_none ||
                //TODO: new api needs to be created to read such caps
                //right now use whitelist so no new primitives will be affected in case of lack of fused activation support
                (!input.is_type<batch_norm>() && !input.is_type<concatenation>() && !input.is_type<convolution>() &&
                    !input.is_type<crop>() && !input.is_type<deconvolution>() && !input.is_type<eltwise>() &&
                    !input.is_type<fully_connected>() && !input.is_type<lrn>() && !input.is_type<normalize>() &&
                    !input.is_type<permute>() && !input.is_type<pooling>() && !input.is_type<reorder>() &&
                    !input.is_type<roi_pooling>() && !input.is_type<scale>() &&
                    !input.is_type<softmax>() && !input.is_type<upsampling>() && !input.is_type<mvn>()))
                return;

            input.set_fused_activation(node.get_primitive()->activation_func, node.get_primitive()->additional_params);
            input.set_output_padding(node.get_output_layout().data_padding);

            extract_and_remove(node);
        });
    }

    //Second loop tries fusing several reorders one by one (if present) into one reorder
    itr = processing_order.begin();
    while (itr != processing_order.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        do_for_types<reorder>(*node, [this, is_debug](reorder_node& node)
        {
            auto& input = node.input();

            //Restrictions:
            // - inputs cannot be padded
            // - primitives input cannot be output
            // - input was optimized
            if (node.has_padded_dependency() || (input.is_output() && !is_debug) || node.get_dependencies().size() != 1 ||
                input.can_be_optimized())
                return;

            // - check if previous node is reorder with 1 user
            // - do not fuse if current node has mean subtract
            if (input.get_users().size() != 1 || !input.is_type<reorder>() ||
                node.has_mean() || !node.get_primitive()->subtract_per_feature.empty())
                return;

            input.set_output_layout(node.get_output_layout(), false);
            extract_and_remove(node);
        });
    }
    //Third loop tries fusing eltwise (sum) with deconvolution
    itr = processing_order.begin();
    while (itr != processing_order.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        fuse_skip_layers(node);
    }
}

program_node& program_impl::get_or_create(std::shared_ptr<primitive> prim)
{
    auto itr = nodes_map.lower_bound(prim->id);
    if (itr != nodes_map.end() && itr->first == prim->id)
        return *itr->second;

    auto new_node = prim->type->create_node(*this, prim);
    new_node->set_org_primitive_id(new_node->id());
    nodes_map.insert(itr, { prim->id, new_node });
    return *new_node;
}

void program_impl::add_intermediate(program_node& node, program_node& next, size_t prev_idx, bool connect_int_node_with_old_dep)
{
    if (connect_int_node_with_old_dep && !node.dependencies.empty())
        throw std::invalid_argument("Node which is about to be added inbetween two other nodes should not have any existing dependencies");

    auto& prev = next.get_dependency(prev_idx);
    //firstly add connection, later replace dependency, so 'prev' won't become dangling and therefore removed
    if (connect_int_node_with_old_dep)
    {
        add_connection(prev, node);
        if (node.processing_itr != processing_order.end())
            processing_order.erase(node.processing_itr);

        auto itr = prev.processing_itr;
        node.processing_itr = processing_order.insert(++itr, &node);
        node.processing_num = prev.processing_num;
    }

    next.replace_dependency(prev_idx, node);
    node.constant = prev.constant;
    node.data_flow = prev.data_flow;
    if (prev.constant_frontier)
    {
        node.constant_frontier = true;
        prev.constant_frontier = false;
    }
}

void program_impl::rename(program_node & node, primitive_id const & new_id)
{
    if (nodes_map.count(new_id))
        throw std::runtime_error("Trying to rename program_node but node with id " + new_id + " already exists");
    if (node.is_output())
        throw std::invalid_argument("Trying to rename an output node. If you intend to do that, please clear 'output' flag manually.");

    auto node_ptr = nodes_map.find(node.id())->second;
    nodes_map.emplace(new_id, node_ptr);
    nodes_map.erase(node.id());

    if (!node.is_type<internal_primitive>())
        const_cast<primitive_id&>(node.desc->id) = new_id;
    else
        reinterpret_cast<details::internal_program_node_base&>(node).internal_id = new_id;
}

void program_impl::swap_names(program_node& node1, program_node& node2)
{
    const auto _extract_id = [](program_node& node) -> primitive_id&
    {
        if (!node.is_type<internal_primitive>())
            return const_cast<primitive_id&>(node.desc->id);
        else
            return reinterpret_cast<details::internal_program_node_base&>(node).internal_id;
    };

    nodes_map.at(node1.id()).swap(nodes_map.at(node2.id()));
    std::swap(_extract_id(node1), _extract_id(node2));
}

void program_impl::replace_all_usages(program_node & old_node, program_node & new_node)
{
    auto itr = old_node.users.begin();
    bool end = (itr == old_node.users.end());
    while (!end)
    {
        auto& usage = (*itr++);
        end = (itr == old_node.users.end());
        usage->replace_dependency(old_node, new_node);
    }
}

void program_impl::replace(program_node& old_node, program_node& new_node, bool replace_whole_branch, bool check_output_layouts_integrity)
{
    if ((!new_node.dependencies.empty() && !replace_whole_branch) || !new_node.users.empty())
        throw std::invalid_argument("Node which is about to replace other node should be detached");

    if (new_node.is_output())
        throw std::invalid_argument("Replacement node shouldn't be marked as an output since it's impossible to rename such node.");

    auto id = old_node.id();
    new_node.output_layout = old_node.get_output_layout();
    new_node.valid_output_layout = old_node.valid_output_layout;

    if (!replace_whole_branch)
    {
        //copy old's dependencies
        while (!old_node.dependencies.empty())
        {
            auto& dep = old_node.dependencies.back();
            add_connection(*dep, new_node);
            remove_connection(*dep, old_node);
        }
    }

    //append users
    for (auto& user : old_node.users)
    {
        new_node.users.push_back(user);
        for (auto& users_dep : user->dependencies)
        {
            if (users_dep == &old_node)
            {
                users_dep = &new_node;
                break;
            }
        }
    }

    old_node.users.clear();

    if (check_output_layouts_integrity && new_node.valid_output_layout)
        new_node.recalc_output_layout();

    bool old_was_output = false;
    //copy node's state
    if (old_node.is_output())
    {
        old_was_output = true;
        old_node.set_output(false);
        outputs.erase(std::remove(outputs.begin(), outputs.end(), &old_node), outputs.end());
    }
    if (new_node.is_input())
        inputs.push_back(&new_node);
    if (old_node.is_input())
        inputs.remove(&old_node);

    new_node.constant = old_node.constant;
    new_node.constant_frontier = old_node.constant_frontier;
    new_node.user_mark = old_node.user_mark;

    auto old_news_pos = new_node.processing_itr;
    new_node.processing_itr = processing_order.insert(old_node.processing_itr, &new_node);
    new_node.processing_num = old_node.processing_num;
    if (old_news_pos != processing_order.end())
        processing_order.erase(old_news_pos);
    if (old_node.processing_itr != processing_order.end())
        processing_order.erase(old_node.processing_itr);

    nodes_map.erase(id);
    rename(new_node, id);

    //mark new node as an output after renaming
    if (old_was_output)
    {
        new_node.set_output(true);
        outputs.push_back(&new_node);
    }
}

bool program_impl::remove_if_dangling(program_node& node, bool detach_whole_branch)
{
    if (!node.users.empty())
        return false;
    if (!detach_whole_branch && !node.dependencies.empty())
        return false;

    std::list<program_node*> to_remove;
    std::list<program_node*> marked;
    if (detach_whole_branch)
    {
        node.mark();
        std::list<program_node*> queue = { &node };
        while (!queue.empty())
        {
            auto curr = queue.front();
            queue.pop_front();
            marked.push_back(curr);

            //remove only if all users also has been marked
            bool rem = !std::any_of(curr->get_users().begin(), curr->get_users().end(), [](program_node* node) { return !node->is_marked(); });
            if (rem)
                to_remove.push_back(curr);

            for (auto dep : curr->get_dependencies())
            {
                if (!dep->is_marked())
                {
                    dep->mark();
                    queue.push_back(dep);
                }
            }
        }
    }
    else
        to_remove.push_back(&node);

    for (auto n : marked)
        n->unmark();

    for (auto rem : to_remove)
    {
        if (!rem->is_output() || is_debug_build())
        {
            if (detach_whole_branch)
            {
                for (auto& user : rem->get_users())
                    user->remove_dependency(*rem);
            }
            if (rem->is_input())
                inputs.remove(rem);

            if (std::find(processing_order.begin(), processing_order.end(), rem) != processing_order.end())
                processing_order.erase(rem->processing_itr);
            optimized_out.push_back(rem->id());
            nodes_map.erase(rem->id());
        }
    }

    return true;
}

bool program_impl::extract_and_remove(program_node& node)
{
    if (node.get_dependencies().size() != 1)
        return false;

    if (node.is_output() && node.get_dependency(0).is_output() && !is_debug_build()) //TODO: add a mechanism to support removal of nodes which are marked as outputs
        return false;

    if (node.is_output() && !is_debug_build())
    {
        auto& prev = node.get_dependency(0);
        auto node_id = node.id();

        node.set_output(false);
        outputs.erase(std::remove(outputs.begin(), outputs.end(), &node), outputs.end());

        rename(node, "_cldnn_tmp_" + node_id);
        rename(prev, node_id);

        prev.set_output(true);
        outputs.push_back(&prev);
    }

    auto& input = node.get_dependency(0);
    node.dependencies.clear();
    input.users.remove(&node);

    if (node.constant_frontier)
    {
        assert(node.constant && "Constant frontier should also, by definition, be constant");
        assert(input.constant && "Input for constant forontier should, by definition, be constant");
        input.constant_frontier = true;
    }

    if (!node.is_endpoint())
        replace_all_usages(node, input);
    else
        remove_if_dangling(node);

    return true;
}

void program_impl::replace_data_with_optimized(std::map<primitive_id, memory_impl::ptr> const & replace_map)
{
    for (auto& result : replace_map)
    {
        auto& node = *nodes_map.at(result.first);
        assert(node.is_type<data>() && "Optimized primitive is not a cldnn::data");
        assert(result.second != nullptr && "Memory which handles result of optimization should not be nullptr");
        node.as<data>().attach_memory(*result.second, false);
    }
}

void program_impl::dump_memory_pool() const
{
    if (!get_engine().configuration().enable_memory_pool)
        return;
    auto path = get_dir_path(options);
    if (path.empty())
    {
        return;
    }

    path += "cldnn_memory_pool.log";
    auto dep = get_memory_dependencies_string();
    get_engine().dump_memory_pool(*this, path, dep);
    dump_program("14_memory_pool", true);
}

//TODO: break this function into number of smaller ones + add per-primitive fields (possibly use primitive_inst::to_string?)
void program_impl::dump_program(const char* stage, bool with_full_info, std::function<bool(program_node const&)> const& filter) const
{
    auto path = get_dir_path(options);
    if (path.empty())
    {
        return;
    }

    std::ofstream graph(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".graph");
    dump_graph_init(graph, *this, filter);

    if (!with_full_info)
    {
        return;
    }

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".info");
    dump_graph_info(graph, *this, filter);

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".order");
    dump_graph_processing_order(graph, *this);

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".optimized");
    dump_graph_optimized(graph, *this);
}

//Dumps weights and biasses in serialization process, not working yet, in progress.
void program_impl::dump_weights_and_biasses(std::vector<unsigned long long>& offsets, std::vector<std::string>& data_names, std::ofstream& file_stream) const
{
    for (auto const& n : nodes_map)
    {
        auto dependency_count = (unsigned int)n.second.get()->get_dependencies().size();
        for (unsigned int dp = 0; dp < dependency_count; dp++)
        {
            auto& dependency = n.second.get()->get_dependency(dp);
            if (dependency.is_type<data>())
            {
                offsets.push_back(offsets.empty() ? 0ull : offsets.back());
                auto& mem = dependency.as<data>().get_attached_memory();
                if (mem.get_layout().data_type == data_types::f32)
                    dump_data(mem, file_stream, offsets.back(), sizeof(float));
                else
                    dump_data(mem, file_stream, offsets.back(), sizeof(short));
                data_names.push_back(dependency.as<data>().id());
            }
        }
    }
    file_stream.close();
}

//Makes serialization with given name.
//Placeholder, not working yet, in progress.
void program_impl::serialize(std::string network_name, std::function<bool(program_node const&)> const& filter) const
{
    std::vector<unsigned long long> offsets;
    std::vector<std::string> data_names;

    std::ofstream file_stream(network_name + "_" + "serialization" + ".bin", std::ios::binary);
    dump_kernels(engine->get_context().get()->get_kernels_cache().get_context().get_binaries(), offsets, data_names, file_stream);
    dump_weights_and_biasses(offsets, data_names, file_stream);

    std::ofstream graph(network_name + "_" + "serialization" + ".xml");
    dump_to_xml(graph, *this, filter, offsets, data_names);
}
