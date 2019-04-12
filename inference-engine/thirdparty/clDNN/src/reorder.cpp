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
#include "reorder_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

#include <algorithm>

namespace cldnn
{

primitive_type_id reorder_type_id()
{
    static primitive_type_base<reorder> instance;
    return &instance;
}

layout reorder_inst::calc_output_layout(reorder_node const& node)
{
    auto input_layout = node.input().get_output_layout();
    auto ifmt = input_layout.format;

    auto odt = *node.get_primitive()->output_data_type;
    auto ofmt = node.get_primitive()->output_format;
    auto op = node.get_primitive()->output_padding;

    if (ofmt.is_winograd() && ifmt.is_winograd())
    {
        if (ofmt == ifmt)
            return layout(odt, ofmt, input_layout.size, op);

        CLDNN_ERROR_MESSAGE(node.id(), "Reordering between winograd weights and data formats is unsupported");
    }

    //transformation of data from standard to winograd
    if (ofmt == format::winograd_2x3_s1_data)
    {
        //some constants which are defined by F(2,3) with stride 1 -- todo: think about generic way to calculate them for any F(r,m) with stride s
        // NOTE: FOR THE FOLLOWING CONSTANTS 'OUTPUT' MEANS OUTPUT OF WINOGRAD CONV (in standard domain) AND 'INPUT' MEANS INPUT FOR WINOGRAD CONV (in winograd domain),
        // THEREFORE 'INPUT' ACTUALLY REFERS TO THE OUTPUT OF THIS CONVERSION (which is later fed as input for winograd conv)
        constexpr tensor::value_type output_tile_width = 2; //by definition of F(2,3)
        constexpr tensor::value_type filter_width = 3; //by definition of F(2,3)
        constexpr tensor::value_type filter_stride = 1; //by definition of format::winograd_2x3_s1_data (our assumption)

        constexpr tensor::value_type input_tile_width = filter_width + (output_tile_width - 1) * filter_stride; //input tile should be large enought to hold data for computations of output tile (for given filter size and stride)

        auto input_offset = node.get_input_offset();

        //how many tiles do we need to produce
        // each input tile produces one output tile so we can find no. of input tiles by calculating no. of output tiles (which is equal to width of an output divided by output tile width)
        tensor::value_type conv_output_width = input_layout.size.spatial[0] - input_offset.spatial[0] - filter_width + 1;
        tensor::value_type input_tiles_count_x = conv_output_width / output_tile_width;
        tensor::value_type output_width = input_tiles_count_x * input_tile_width;
        tensor::value_type output_height = input_layout.size.spatial[1] - input_offset.spatial[1];

        tensor::value_type padd_x = 0;
        tensor::value_type padd_y = (8 - ((output_height - 2) % 8)) % 8;
        if (conv_output_width % output_tile_width != 0) //leftovers
        {
            output_width += 3; //one tile is 4 elements from which only 3 first are used to generate first output value
            padd_x = 1;
        }

        auto data_size = tensor{ input_layout.size.batch[0], input_layout.size.feature[0], output_width, output_height };
        tensor upper_padd = tensor{ 0, 0, padd_x, padd_y };
        return layout(odt, ofmt, data_size, padding{ { 0,0,0,0}, upper_padd.sizes() });
    }
    
    //transformation of weights from standard to winograd
    if (ofmt == format::winograd_2x3_s1_weights || ofmt == format::winograd_2x3_s1_fused_weights)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "input_layout.size.spatial[0]", input_layout.size.spatial[0], "expected value", 3, "input for conversion to winograd_2x3_s1 weights format should have spatial size 3x3");
        CLDNN_ERROR_NOT_EQUAL(node.id(), "input_layout.size.spatial[1]", input_layout.size.spatial[1], "expected value", 3, "input for conversion to winograd_2x3_s1 weights format should have spatial size 3x3");

        return layout(odt, ofmt, tensor{ input_layout.size.batch[0], input_layout.size.feature[0], 4, 3 });
    }
    else if(ofmt == format::winograd_6x3_s1_fused_weights)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "input_layout.size.spatial[0]", input_layout.size.spatial[0], "expected value", 3, "input for conversion to winograd_2x3_s1 weights format should have spatial size 3x3");
        CLDNN_ERROR_NOT_EQUAL(node.id(), "input_layout.size.spatial[1]", input_layout.size.spatial[1], "expected value", 3, "input for conversion to winograd_2x3_s1 weights format should have spatial size 3x3");

        return layout(odt, ofmt, tensor{ input_layout.size.batch[0], input_layout.size.feature[0], 8, 3 });
    }

    //transformation of data from winograd to standard
    if (ifmt == format::winograd_2x3_s1_data)
    {
        constexpr tensor::value_type output_tile_width = 2; //by definition of F(2,3)
        constexpr tensor::value_type filter_width = 3; //by definition of F(2,3)
        constexpr tensor::value_type filter_stride = 1; //by definition of format::winograd_2x3_s1_data (our assumption)

        constexpr tensor::value_type input_tile_width = filter_width + (output_tile_width - 1) * filter_stride; //input tile should be large enought to hold data for computations of output tile (for given filter size and stride)

        auto output_width = input_layout.size.spatial[0] / input_tile_width * output_tile_width;
        if (input_layout.size.spatial[0] % input_tile_width != 0) //leftovers
            ++output_width; //output tile is 2 by default, so we can have only 1 value as leftover

        return layout(odt, ofmt, tensor{ input_layout.size.batch[0], input_layout.size.feature[0], output_width, input_layout.size.spatial[1] });
    }

    //transformation of weights from winograd to standard
    if (ifmt == format::winograd_2x3_s1_weights || ifmt == format::winograd_2x3_s1_fused_weights || ifmt == format::winograd_6x3_s1_fused_weights)
    {
        CLDNN_ERROR_MESSAGE(node.id(), "Conversion of weights from winograd to standard domain is currently unsupported");
    }

    if(ofmt == format::bs_xs_xsv8_bsv8 || ofmt == format::bs_xs_xsv8_bsv16 || ofmt == format::bs_x_bsv16)
        return layout(odt, ofmt, input_layout.size.transform(ofmt, 1), op);
    else
        return layout(odt, ofmt, input_layout.size, op);
}

std::string reorder_inst::to_string(reorder_node const& node)
{
    auto desc      = node.get_primitive();
    auto mean      = desc->mean;
    auto node_info = node.desc_to_json();
    auto& input    = node.input();

    std::stringstream  primitive_description;

    json_composite reorder_info;
    reorder_info.add("input id", input.id());
    reorder_info.add("mean", mean);
    if (desc->subtract_per_feature.size() > 0)
    {
        reorder_info.add("subtract per feature", desc->subtract_per_feature);
    } 

    node_info->add("reorder info", reorder_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

reorder_inst::typed_primitive_inst(network_impl& network, reorder_node const& node)
    : parent(network, node, !node.can_be_optimized())
{
    if (node.can_be_optimized())
    {
        build_deps();
        reuse_input();
    }


    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();

    CLDNN_ERROR_LESS_THAN(node.id(), "Input dimension size", input_layout.size.raw.size(), "ouput dimension size", output_layout.size.raw.size(), "Input dimension < output dimension. Reorder primitive woks only with same dimension sizes (reorder) or when input > output (flatten).");
    
    if (!argument.subtract_per_feature.empty())
    {
        CLDNN_ERROR_GREATER_THAN(node.id(), "Input feature dimension size", input_layout.size.feature.size(), "value", 1, "Subtracting values work only for formats that have feature dimension == 1");
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Input feature size[0]", static_cast<size_t>(input_layout.size.feature[0]), "argument subtract per feature size", argument.subtract_per_feature.size(), "Number of features/channels in input does not match the number of features/channels in values to subtract");
    }
}

void reorder_inst::on_execute()
{
    if (node.can_be_optimized())
        reuse_input();
}

void reorder_inst::reuse_input()
{
    if (!node.can_be_optimized())
        return;

    if (node.requires_reinterpret())
    {
        if (!_output || !_network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
            _output = _network.get_engine().reinterpret_buffer(input_memory(), node.get_output_layout());
    }
    else if (!_output)
        _output = &input_memory();
}

}
