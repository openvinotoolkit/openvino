/*
// Copyright (c) 2018 Intel Corporation
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

#include "layout_optimizer.h"
#include "topology_impl.h"
#include "network_impl.h"
#include "primitive_inst.h"
#include "error_handler.h"

#include "data_inst.h"
#include "reorder_inst.h"
#include "generic_layer.hpp"
#include <boost/filesystem.hpp>
#include <sstream>

#include "eltwise_inst.h"
#include "pooling_inst.h"

using namespace cldnn;

namespace {
    bool should_use_winograd_2x3_s1(std::shared_ptr<const convolution> const& prim, layout const& input_layout, layout const& weights_layout, bool output_size_handling_enabled)
    {
        //cases when NOT to use winograd
        if (input_layout.size.feature[0] % 64 != 0       //current algorithm is effective for ifm to be multiply of 64
            || weights_layout.size.spatial[0] != 3          //weights have to be 3x3 by definiton
            || weights_layout.size.spatial[1] != 3          //weights have to be 3x3 by definition
            || weights_layout.size.batch[0] % 64 != 0       //current algorithm is effective for ofm to be multiply of 64
            || prim->stride != tensor{ 1 }                  //stride has to be 1x1 by definition
            || prim->dilation != tensor{ 1 }                //no support for dilation
            || prim->split() != 1                           //no support for splitted convolutions
            || (output_size_handling_enabled && prim->with_output_size) //no support for convolutions with user-specified output size
            || (input_layout.count() > 3000000)             //limit max input size as winograd consumes more memory
            || (input_layout.count() < 50000)               //limit min input size as winograd is not effective for small input
            || (input_layout.size.spatial[0] < 8 && input_layout.size.spatial[0] < 8)) //disable winograd for small spatials as perf is poor
        {
            return false;
        }

        return true;
    }
}

layout_optimizer::layout_optimizer(bool output_size_handling_enabled)
    : _optimization_attributes()
    , _output_size_handling_enabled(output_size_handling_enabled)
{
}

bool layout_optimizer::convolution_bfyx_opt(layout const& output_layout, const layout& weights_layout, std::shared_ptr<const convolution> conv)
{
    //A set of rules that define when bfyx mem format has better performance than yxfb
    if (output_layout.size.batch[0] == 16 || output_layout.size.batch[0] % 16 != 0 || 
        output_layout.data_type != data_types::f16 || weights_layout.size.batch[0] % 16 != 0 ||
        !((weights_layout.size.spatial[0] == 1 && weights_layout.size.spatial[1] == 1) ||
        (weights_layout.size.spatial[0] >= 5 && weights_layout.size.spatial[1] >= 5) ||
            (conv->stride.spatial[0] > 1 && conv->stride.spatial[1] > 1) ||
            (weights_layout.size.feature[0] <= 32 && output_layout.size.spatial[0] < 224 && output_layout.size.spatial[1] < 224) ||
            (weights_layout.size.feature[0] <= 64 && output_layout.size.spatial[0] < 112 && output_layout.size.spatial[1] < 112) ||
            (weights_layout.size.feature[0] <= 128 && output_layout.size.spatial[0] < 56 && output_layout.size.spatial[1] < 56) ||
            (weights_layout.size.feature[0] <= 256 && output_layout.size.spatial[0] < 28 && output_layout.size.spatial[1] < 28) ||
            (weights_layout.size.feature[0] <= 512 && output_layout.size.spatial[0] < 14 && output_layout.size.spatial[1] < 14) ||
            (weights_layout.size.feature[0] <= 1024 && output_layout.size.spatial[0] <= 7 && output_layout.size.spatial[1] <= 7)) ||
        //WA for AgeGender, which has one convolution that is better on yxfb, but due to additonal reorder overall performance is worse than bfyx
        (output_layout.size.spatial[0] == 82 && output_layout.size.spatial[1] == 82) ||
        (_optimization_attributes.splitted_convolution && output_layout.size.batch[0] == 16) ||
        (!_optimization_attributes.splitted_convolution && output_layout.size.batch[0] >= 128) ||
        _optimization_attributes.bfyx_only_layer)
        return true;

    return false;
}

bool layout_optimizer::convolution_byxf_opt(layout const& output_layout, const layout& weights_layout, std::shared_ptr<const convolution> conv)
{
    //A set of rules that define when byxf mem format has better performance
    if ((output_layout.data_type == data_types::f16 &&
        weights_layout.size.spatial[0] == 1 && weights_layout.size.spatial[1] == 1 &&
        output_layout.size.feature[0] % 64 == 0 && weights_layout.size.batch[0] % 64 == 0 &&
        conv->stride.spatial[0] == 1 && conv->stride.spatial[1] == 1 &&
        conv->input_offset.spatial[0] == 0 && conv->input_offset.spatial[1] == 0) ||
        //Winograd
        should_use_winograd_2x3_s1(conv, output_layout, weights_layout, _output_size_handling_enabled))
        return true;

    return false;
}

bool layout_optimizer::users_for_convolution_byxf_opt(program_node const& node, uint32_t depth)
{
    //This function checks if byxf optimization can be applied to the required depth of node's users.
    //Setting depth to 1 will check only node's users, depth = 2 are user's users etc.
    if (depth == 0)
        return true;

    bool use_byxf = false;
    for (auto& user : node.get_users())
    {
        //primitives that support transitions byxf->other format and other format->byxf are valid for byxf opt
        if (user->type() == cldnn::eltwise::type_id() || user->type() == cldnn::pooling::type_id())
            use_byxf = users_for_convolution_byxf_opt(*user, depth - 1);
        //convolution that is capable to use byxf and is performant is also valid for byxf opt
        else if (user->type() == cldnn::convolution::type_id())
        {
            auto conv_prim = user->as<convolution>().get_primitive();
            if (convolution_byxf_opt(user->calc_output_layout(), user->get_dependency(1).get_output_layout(), conv_prim))
                use_byxf = users_for_convolution_byxf_opt(*user, depth - 1);
            else
            {
                use_byxf = false;
                break;
            }
        }
        else
        {
            use_byxf = false;
            break;
        }
    }
    return use_byxf;
}

bool layout_optimizer::deps_depth_in_same_format(program_node const& node, const cldnn::format format, uint32_t depth)
{
    //This function checks if requested format is the same for node's users in the required depth.
    //Setting depth to 1 will check only node's dependencies, depth = 2 are dep's dependencies etc.
    if (depth == 0)
        return true;

    bool same_format = false;
    for (auto& dep : node.get_dependencies())
    {
        //skip data and generic_layers
        if (dep->type() == cldnn::data::type_id() || dep->type() == cldnn::generic_layer::type_id())
            continue;

        //if dependency is of type reorder and format is different then skip it and move to its dependency
        //further in graph such reorders could be optimized out
        if(dep->type() == cldnn::reorder::type_id() && dep->get_dependencies().size() == 1 && dep->get_output_layout().format != format)
            same_format = deps_depth_in_same_format(dep->get_dependency(0), format, depth);
        else if (dep->get_output_layout().format == format)
            //if dependency is of type reorder and format is the same, check if its users are primitives with support for different input and output formats
            //if that is true then graph optimizer will optimize such reorder and layout for its dependency will be changed
            if (dep->type() == cldnn::reorder::type_id() &&
                (dep->get_dependency(0).type() == cldnn::eltwise::type_id() || dep->get_dependency(0).type() == cldnn::pooling::type_id()) &&
                dep->get_dependencies().size() == 1)
                same_format = deps_depth_in_same_format(dep->get_dependency(0), format, depth - 1);
            else
                same_format = deps_depth_in_same_format(*dep, format, depth - 1);
        else
        {
            same_format = false;
            break;
        }
    }
    return same_format;
}

layout layout_optimizer::get_expected_layout(layout const& current_layout, data_type type, convolution_node const& node, layout const& output_or_weights_layout)
{
    auto prim = node.get_primitive();
    auto expected_tensor = current_layout.size;
    auto expected_data_type = current_layout.data_type;
    auto expected_format = current_layout.format;

    if (type == data_type::weights || type == data_type::bias)
    {
        expected_data_type = output_or_weights_layout.data_type;
    }

    switch (type)
    {
    case data_type::bias: //convolution bias
        expected_tensor = cldnn::tensor(1, 1, static_cast<tensor::value_type>(current_layout.count()), 1);
        expected_format = cldnn::format::bfyx;
        break;

    case data_type::input: //convolution input

        if (current_layout.data_type == data_types::f16 &&
            layout_optimizer::convolution_byxf_opt(current_layout, output_or_weights_layout, prim) &&
            (users_for_convolution_byxf_opt(node, 2) || deps_depth_in_same_format(node, cldnn::format::byxf, 2)) &&
            //TODO: remove this condition when yxfb optimizations will be disabled
            current_layout.format != cldnn::format::yxfb &&
            current_layout.size.batch[0] == 1 &&
            !node.get_transposed())
        {
            expected_tensor = current_layout.size;
            expected_format = cldnn::format::byxf;
        }
        // MMAD case
        else if (current_layout.data_type == data_types::i8)
        {
            expected_tensor = current_layout.size;
            expected_format = current_layout.format;//cldnn::format::byxf_af32;
        }
        else if (layout_optimizer::convolution_bfyx_opt(current_layout, output_or_weights_layout, prim)
            || (_output_size_handling_enabled && prim->with_output_size) ||
            node.get_transposed())
        {
            if (current_layout.data_type == data_types::f32 &&
                current_layout.size.batch[0] % 16 == 0 &&
                current_layout.format == format::bfyx &&
                output_or_weights_layout.size.spatial[0] == 1 && output_or_weights_layout.size.spatial[1] == 1 &&
                prim->stride.spatial[0] == 1 && prim->stride.spatial[1] == 1 &&
                prim->input_offset.spatial[0] == 0 && prim->input_offset.spatial[1] == 0 &&
                !node.get_transposed())
            {
                if (!((current_layout.size.feature[0] % 8) == 0 && (current_layout.size.spatial[0] * current_layout.size.spatial[1]) == 16 &&
                    current_layout.data_padding == padding{ { 0,0,0,0 }, 0 }))
                {
                    expected_tensor = current_layout.size.transform(cldnn::format::bf8_xy16, 1);
                    expected_format = cldnn::format::bf8_xy16;
                }
            }
            else
            {
                expected_tensor = current_layout.size;
                expected_format = cldnn::format::bfyx;
            }
        }
        else
        {
            expected_tensor = current_layout.size;
            expected_format = cldnn::format::yxfb;
        }

        break;

    default:
        throw std::runtime_error("Unsupported data type in layout_optimizer::get_expected_layout for convolution primitive");
    }

    return layout(expected_data_type, expected_format, expected_tensor);
}

layout layout_optimizer::get_expected_layout(layout const& current_layout, data_type type, fully_connected_node const& node, layout const& output_or_weights_layout)
{
    auto prim = node.get_primitive();
    auto expected_tensor = current_layout.size;
    auto expected_data_type = current_layout.data_type;
    auto expected_format = current_layout.format;

    if (type == data_type::weights || type == data_type::bias)
    {
        expected_data_type = output_or_weights_layout.data_type;
    }

    switch (type)
    {
    case data_type::bias: //fc bias
        expected_tensor = cldnn::tensor(1, 1, static_cast<tensor::value_type>(current_layout.count()), 1);
        expected_format = cldnn::format::bfyx;
        break;

    default:
        throw std::runtime_error("Unsupported data type in layout_optimizer::get_expected_layout for fully-connected primitive");
    }

    return layout(expected_data_type, expected_format, expected_tensor);
}

layout layout_optimizer::get_expected_layout(layout const& current_layout, data_type type, lstm_gemm_node const& node, layout const& output_or_weights_layout)
{
    auto prim = node.get_primitive();
    auto expected_tensor = current_layout.size;
    auto expected_data_type = current_layout.data_type;
    auto expected_format = current_layout.format;

    if (type == data_type::weights || type == data_type::bias)
    {
        expected_data_type = output_or_weights_layout.data_type;
    }

    switch (type)
    {
    case data_type::bias:
        expected_tensor = cldnn::tensor(1, 1, static_cast<tensor::value_type>(current_layout.count()), 1);
        expected_format = cldnn::format::bfyx;
        break;

    default:
        throw std::runtime_error("Unsupported data type in layout_optimizer::get_expected_layout for fully-connected primitive");
    }

    return layout(expected_data_type, expected_format, expected_tensor);
}

layout layout_optimizer::get_expected_layout(layout const& current_layout, data_type type, deconvolution_node const& node, layout const& output_or_weights_layout)
{
    auto prim = node.get_primitive();
    auto expected_tensor = current_layout.size;
    auto expected_data_type = current_layout.data_type;
    auto expected_format = current_layout.format;

    if (type == data_type::weights || type == data_type::bias)
    {
        expected_data_type = output_or_weights_layout.data_type;
    }

    switch (type)
    {
    case data_type::bias: //convolution bias
        expected_tensor = cldnn::tensor(1, 1, static_cast<tensor::value_type>(current_layout.count()), 1);
        expected_format = cldnn::format::bfyx;
        break;

    default:
        throw std::runtime_error("Unsupported data type in layout_optimizer::get_expected_layout for deconvolution primitive");
    }

    return layout(expected_data_type, expected_format, expected_tensor);
}

layout layout_optimizer::get_expected_layout(layout const& current_layout, data_type type, detection_output_node const& node, layout const& output_or_weights_layout)
{
    auto prim = node.get_primitive();
    auto expected_tensor = current_layout.size;
    auto expected_data_type = data_types::f32;
    auto expected_format = output_or_weights_layout.format;

    if (type != data_type::input)
        CLDNN_ERROR_MESSAGE(prim->id, "detection_output only supports optimization of its output (no weights/biases)");

    return layout(expected_data_type, expected_format, expected_tensor);
}

layout layout_optimizer::get_expected_layout(layout const& current_layout, data_type type, embed_node const& node, layout const& output_or_weights_layout)
{
    auto prim = node.get_primitive();
    auto expected_tensor = current_layout.size;
    auto expected_data_type = current_layout.data_type;
    auto expected_format = current_layout.format;

    if (type == data_type::weights || type == data_type::bias)
    {
        expected_data_type = output_or_weights_layout.data_type;
    }

    switch (type)
    {
    case data_type::bias:
        expected_tensor = cldnn::tensor(1, 1, static_cast<tensor::value_type>(current_layout.count()), 1);
        expected_format = cldnn::format::bfyx;
        break;

    default:
        throw std::runtime_error("Unsupported data type in layout_optimizer::get_expected_layout for embed primitive");
    }

    return layout(expected_data_type, expected_format, expected_tensor);
}

std::pair<std::shared_ptr<cldnn::reorder>, bool>
layout_optimizer::create_reorder_if_needed(const layout& current_layout, const cldnn::primitive_id& memid, layout const& expected_layout)
{
    if (current_layout != expected_layout)
    {
        cache_key ckey{ memid, expected_layout };
        auto itr = _cached_reorders.find(ckey);
        if (itr != _cached_reorders.end())
            return std::make_pair(itr->second, true);

        auto count = _cached_reorders.size();
        std::stringstream ss;
        ss << "reorder_" << count << "_" << memid;

        auto reorder = std::make_shared<cldnn::reorder>(ss.str(), memid, expected_layout);
        _cached_reorders[ckey] = reorder;
        return std::make_pair(reorder, false);
    }

    return std::make_pair(nullptr, true);
}

std::pair<std::shared_ptr<cldnn::generic_layer>, bool>
layout_optimizer::create_reorder_from_given_source(const cldnn::primitive_id& memid, layout const& expected_layout, const kernel_selector::weights_reorder_params& reorder_params)
{
    cache_key ckey{ memid, expected_layout };
    auto itr = _cached_generic_layers.find(ckey);
    if (itr != _cached_generic_layers.end())
        return std::make_pair(itr->second, true);

    auto count = _cached_generic_layers.size();
    std::stringstream ss;
    ss << "generic_layer_" << count << "_" << memid;

    auto reorder = std::make_shared<cldnn::generic_layer>(ss.str(), memid, expected_layout, reorder_params);
    _cached_generic_layers[ckey] = reorder;
    return std::make_pair(reorder, false);
}

std::vector<std::pair<std::shared_ptr<primitive>, bool>> layout_optimizer::get_generic_layer(
    const kernel_selector::weights_reorder_params & reorder_params,
    primitive_id input_id,
    const layout & old_layout,
    data_type type)
{

    if (reorder_params.engine == kernel_selector::weights_reorder_params::Engine::NONE || type != data_type::weights)
        return{};

    std::vector<std::pair<std::shared_ptr<primitive>, bool>> ret;

    if (reorder_params.engine == kernel_selector::weights_reorder_params::Engine::CPU &&
        reorder_params.cpuKernel != nullptr)
    {
        const auto intermediate_format = from_weights_layout(reorder_params.cpuKernel->GetExpectedInputLayout());
        const auto intermediate_type = from_weights_type(reorder_params.cpuKernel->GetExpectedInputType());
        if (intermediate_format != old_layout.format ||
            intermediate_type != old_layout.data_type)
        {
            const layout intermediate_layout = { intermediate_type, intermediate_format, old_layout.size.transform(intermediate_format, 1) };

            auto reorder = create_reorder_if_needed(old_layout, input_id, intermediate_layout);
            if (reorder.first)
            {
                ret.push_back(reorder);
                input_id = reorder.first->id;
            }
        }
    }

    auto new_dtype = from_weights_type(reorder_params.dtype);
    const auto bpp = data_type_traits::size_of(new_dtype);
    tensor expected_size = { 1,1,1,(tensor::value_type)(reorder_params.newBufferSize / bpp) };

    if (reorder_params.toImageType)
        expected_size = old_layout.size;

    layout expected_layout = {
        new_dtype, reorder_params.toImageType ? from_weights_layout(reorder_params.destLayout) : format::bfyx, // simple linear format (flatten to x channel)
        expected_size
    };

    auto reorder = create_reorder_from_given_source(input_id, expected_layout, reorder_params);
    if (reorder.first)
        ret.push_back(reorder);

    return ret;
}

void layout_optimizer::set_optimization_attribute(optimization_attributes_type attribute, int32_t val)
{
    switch (attribute)
    {
    case optimization_attributes_type::splitted_convolution:
        _optimization_attributes.splitted_convolution = val;
        break;
    case optimization_attributes_type::bfyx_only_layer:
        _optimization_attributes.bfyx_only_layer = val;
        break;
    default:
        throw std::out_of_range("unsupported layout optimization attribute");
    }
}
