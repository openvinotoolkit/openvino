// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>

#include "pass_manager.h"
#include "program_node.h"
#include "mutable_data_inst.h"
#include "tensor_type.h"
#include <memory>
#include <vector>
#include <stdexcept>

/*
This pass checks if data formats (layouts) of output/input in hidden layers match.
If not than required reorder is added to the network.
*/

/*
Add a reorder in between node and usr
*/
void add_required_reorders::add_reorder(program& p, program_node* node, program_node* usr) {
    layout reorder_layout = node->get_output_layout();
    reorder_layout.format = usr->get_output_layout().format;
    reorder_layout.data_type = usr->get_output_layout().data_type;

    auto new_reorder = std::make_shared<reorder>(node->id() + "_reorder_" + usr->id(), node->id(), reorder_layout);
    auto& new_reorder_node = p.get_or_create(new_reorder);
    new_reorder_node.set_output_layout(reorder_layout, false);

    // ToDo: add a method to program class which adds an intermediate node given a node and its user
    auto it = std::find(usr->get_dependencies().begin(), usr->get_dependencies().end(), node);
    if (it == usr->get_dependencies().end()) {
        throw std::runtime_error("Inconcistency in topology description: user of a node is not present among its dependecies.");
    }
    auto idx = it - usr->get_dependencies().begin();
    if (idx < 0 || (size_t)idx >= usr->get_dependencies().size()) {
        throw std::runtime_error("Internal Error: container index out of range exception.");
    }
    p.add_intermediate(new_reorder_node, *usr, idx);
}

void add_required_reorders::run(program& p) {
    auto usr_itr = p.get_processing_order().begin();
    while (usr_itr != p.get_processing_order().end()) {
        auto& usr = *usr_itr++;
        if (usr->get_dependencies().size() == 0)
            continue;  // only nodes with dependencies
        if (usr->is_type<data>())
            continue;
        if (usr->type()->does_an_implementation_exist(*usr)) {
            if (usr->get_preferred_impl_type() != impl_types::onednn) {
                continue;
            } else {
                // oneDNN doesn't support padded memory, so add reorder directly if needed
                for (size_t i = 0; i < usr->get_dependencies().size(); i++) {
                    auto& input = usr->get_dependency(i);
                    if (!input.is_in_data_flow() || input.is_constant())
                        continue;

                    auto in_padding = input.get_output_layout().data_padding;
                    if (static_cast<bool>(in_padding)) {
                        bool spatial_padding = false;
                        for (size_t i = 0; i < in_padding.lower_size().spatial.size(); ++i) {
                            spatial_padding |= (in_padding.lower_size().spatial[i] != 0);
                        }
                        for (size_t i = 0; i < in_padding.upper_size().spatial.size(); ++i) {
                            spatial_padding |= (in_padding.upper_size().spatial[i] != 0);
                        }
                        bool batch_padding = false;
                        for (size_t i = 0; i < in_padding.lower_size().batch.size(); ++i) {
                            batch_padding |= (in_padding.lower_size().batch[i] != 0);
                        }
                        for (size_t i = 0; i < in_padding.upper_size().batch.size(); ++i) {
                            batch_padding |= (in_padding.upper_size().batch[i] != 0);
                        }
                        if (spatial_padding || batch_padding) {
                            cldnn::layout layout_padding = input.get_output_layout();
                            cldnn::layout layout_wo_padding = input.get_output_layout();
                            layout_wo_padding.data_padding = cldnn::padding{};
                            layout_wo_padding.data_padding.lower_size().feature = layout_padding.data_padding.lower_size().feature;
                            layout_wo_padding.data_padding.upper_size().feature = layout_padding.data_padding.upper_size().feature;
                            auto new_reorder = std::make_shared<reorder>(input.id() + "_padding_reorder_" + usr->id(), input.id(), layout_wo_padding);
                            auto& new_reorder_node = p.get_or_create(new_reorder);
                            p.add_intermediate(new_reorder_node, *usr, i);
                        } else {
                            continue;
                        }
                    }
                }
                continue;
            }
        }

        bool correct_layout_selected = false;
        bool weights_data = (usr->is_type<convolution>() || usr->is_type<deconvolution>() ||
                             usr->is_type<deformable_conv>() || usr->is_type<fully_connected>());

        layout original_layout = usr->get_output_layout();

        for (auto& node : usr->get_dependencies()) {
            if (!node->is_in_data_flow() && !weights_data) {
                if (cldnn::format::dimension(original_layout.format) == cldnn::format::dimension(node->get_output_layout().format)) {
                    /*
                        ToDo: Here we should handle also the situation where primitive usr has data inputs in different
                       formats
                    */
                    layout current_layout(original_layout.data_type,
                                          node->get_output_layout().format,
                                          original_layout.size);
                    usr->set_output_layout(current_layout, false);
                    if (usr->type()->does_possible_implementation_exist(*usr)) {
                        correct_layout_selected = true;
                        break;
                    } else if (original_layout.data_type == data_types::i64) {
                        // goal of this section is to use int32 implementation
                        // if int64 is not available for usr primitive
                        current_layout = original_layout;
                        current_layout.data_type = data_types::i32;
                        usr->set_output_layout(current_layout, false);
                        if (usr->type()->does_possible_implementation_exist(*usr)) {
                            correct_layout_selected = true;
                        } else {
                            current_layout = original_layout;
                            current_layout.data_type = data_types::i32;
                            current_layout.format = node->get_output_layout().format;
                            usr->set_output_layout(current_layout, false);
                            if (usr->type()->does_possible_implementation_exist(*usr)) {
                                correct_layout_selected = true;
                            }
                        }

                        if (correct_layout_selected) {
                            // change output_data_type field in usr to i32
                            if ((static_cast<bool>(usr->get_primitive()->output_data_type) == true) &&
                                (*(usr->get_primitive()->output_data_type) == data_types::i64)) {
                                std::const_pointer_cast<primitive>(usr->get_primitive())->output_data_type = data_types::i32;
                            }
                            // add reorders between usr int32 output and inputs of its users
                            auto next_usr_itr = usr->get_users().begin();
                            while (next_usr_itr != usr->get_users().end()) {
                                auto next_usr = *next_usr_itr++;
                                if (!next_usr->is_type<reorder>()) {
                                    if ((next_usr->get_output_layout() != usr->get_output_layout())) {
                                        add_reorder(p, usr, next_usr);
                                    }
                                }
                            }
                            break;
                        }
                    }
                }

                if (!correct_layout_selected) {
                    throw std::runtime_error("Internal Error: no layout format available for " + usr->id() +
                                                " (format: " + std::to_string(original_layout.format.value) +
                                                ", data_type: " + data_type_traits::name(original_layout.data_type) + ") "
                                                "compatible with " + node->id() +
                                                " (format: " + std::to_string(node->get_output_layout().format.value) +
                                                ", data_type: " + data_type_traits::name(node->get_output_layout().data_type) + ")");
                }
            }
        }

        if (!correct_layout_selected) {
            std::vector<cldnn::format> preffered_layout_formats;
            size_t max_in_dims = std::max(cldnn::format::dimension(original_layout.format), static_cast<size_t>(4));
            for (auto& node : usr->get_dependencies()) {
                max_in_dims = std::max(cldnn::format::dimension(node->get_output_layout().format), max_in_dims);
            }
            // This list of preffered layouts has been selected arbitrary due to developers' experience
            if (max_in_dims == 5) {
                preffered_layout_formats = {
                    cldnn::format::bfzyx,
                };
            } else if (max_in_dims == 4) {
                preffered_layout_formats = {
                    cldnn::format::bfyx,
                    cldnn::format::yxfb,
                    cldnn::format::byxf,
                };
            }

            if (usr->get_preferred_impl_type() == impl_types::onednn) {
                usr->set_preferred_impl_type(impl_types::ocl);
                usr->set_output_layout(original_layout, false);
                if (usr->type()->does_possible_implementation_exist(*usr)) {
                    correct_layout_selected = true;
                }
            }

            if (!correct_layout_selected) {
                for (auto new_layout_format : preffered_layout_formats) {
                    layout current_layout(original_layout.data_type, new_layout_format, original_layout.size);
                    usr->set_output_layout(current_layout, false);
                    if (usr->type()->does_possible_implementation_exist(*usr)) {
                        correct_layout_selected = true;
                        break;
                    }
                }
            }

            if (!correct_layout_selected) {
                // goal of this section is to use int32 implementation
                // if int64 is not available for usr primitive
                if (original_layout.data_type == data_types::i64) {
                    layout original_layout_i32(data_types::i32,
                                          original_layout.format,
                                          original_layout.size);

                    usr->set_output_layout(original_layout_i32, false);

                    if (usr->type()->does_possible_implementation_exist(*usr)) {
                        correct_layout_selected = true;
                    }

                    if (!correct_layout_selected) {
                        for (auto new_layout_format : preffered_layout_formats) {
                            layout current_layout_i32(original_layout_i32.data_type,
                                                  new_layout_format,
                                                  original_layout_i32.size);
                            usr->set_output_layout(current_layout_i32, false);
                            if (usr->type()->does_possible_implementation_exist(*usr)) {
                                correct_layout_selected = true;
                                break;
                            }
                        }
                    }

                    if (!correct_layout_selected) {
                        throw std::runtime_error("Internal Error: no implementation for " + usr->id() +
                            " kernel which satisfies output format dependecies.");
                    }

                    // change output_data_type field in usr to i32
                    if ((static_cast<bool>(usr->get_primitive()->output_data_type) == true) &&
                        (*(usr->get_primitive()->output_data_type) == data_types::i64)) {
                        std::const_pointer_cast<primitive>(usr->get_primitive())->output_data_type = data_types::i32;
                    }

                    // add reorders between usr int32 output and inputs of its users
                    auto next_usr_itr = usr->get_users().begin();
                    while (next_usr_itr != usr->get_users().end()) {
                        auto next_usr = *next_usr_itr++;
                        if (!next_usr->is_type<reorder>()) {
                            if ((next_usr->get_output_layout() != usr->get_output_layout())) {
                                add_reorder(p, usr, next_usr);
                            }
                        }
                    }
                }
            }
        }

        // layout is selected now add required reorders
        auto dep_itr = usr->get_dependencies().begin();
        while (dep_itr != usr->get_dependencies().end()) {
            auto node = *dep_itr++;
            // do not add a reorder if usr or node are reorders or does not belong to data_flow
            if (!usr->is_type<reorder>() && node->is_in_data_flow()) {
                if ((usr->get_output_layout() != node->get_output_layout())) {
                    add_reorder(p, node, usr);
                }
            }
        }
    }
}
