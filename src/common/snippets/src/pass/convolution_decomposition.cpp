// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/convolution_decomposition.hpp"

#include "snippets/remarks.hpp"
#include <snippets/itt.hpp>
#include "snippets/op/subgraph.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>
#include <cassert>
#include <queue>
#include <string>
#include <numeric>
#include <climits>

#include "snippets/op/load.hpp"
//#include "snippets/op/conditional_jump.hpp"
#include "snippets/op/convolution_merged_1x1_kernel.hpp"
#include "snippets/op/convolution_merged_dw_kernel.hpp"
//#include "snippets/op/label.hpp"
//#include "snippets/op/loop.hpp"
//#include "snippets/op/auto_loop.hpp"
//#include "snippets/op/scalar_broadcast_load.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

namespace {
void get_nodes_before_result(const std::shared_ptr<Node>& node, const bool check, std::vector<std::shared_ptr<Node>>& nodes) {
    if (is_type<opset1::Result>(node) || (check && node->get_output_size() != 1ul)) {
        return;
    }

    //const auto& target_inputs = node->output(0).get_target_inputs();
    // TODO: just to test
    //if (check && target_inputs.size() != 1ul) {
    //    return;
    //}

    if (check) {
        auto &rt = node->get_rt_info();
        auto it = rt.find("LayoutDependent");
        if (it != rt.end()) {
            return;
        }
    }

    nodes.push_back(node);

    get_nodes_before_result(node->output(0).get_target_inputs().begin()->get_node()->shared_from_this(), true, nodes);
}

void get_nodes(
    const std::shared_ptr<Node>& begin_node,
    const std::shared_ptr<Node>& end_node,
    std::vector<std::shared_ptr<Node>>& nodes) {
    if (begin_node == end_node) {
        return;
    }

    nodes.push_back(begin_node);

    get_nodes(
        begin_node->output(0).get_target_inputs().begin()->get_node()->shared_from_this(),
        end_node,
        nodes);
}
} // namespace

namespace {
std::shared_ptr<Node> get_load(const std::shared_ptr<Node>& node) {
    if (is_type<ov::snippets::op::Load>(node) || is_type<ngraph::opset1::Parameter>(node)) {
        return node;
    }

    if (is_type<opset1::Parameter>(node) || (node->get_input_size() != 1ul)) {
        return nullptr;
    }

    const auto parent = node->get_input_node_shared_ptr(0);
    if (parent->output(0).get_shape() != node->output(0).get_shape()) {
        return nullptr;
    }

    return get_load(parent);
}

//bool decompose_1x1_to_single_optimal(const std::shared_ptr<ngraph::opset1::Convolution>& convolution) {
//    const auto biases_add = convolution->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
//    const auto biases = biases_add->get_input_node_shared_ptr(1ul);
//
//    const auto load = get_load(convolution->get_input_node_shared_ptr(0));
//    assert(load != nullptr);
//
//    //// TODO: will be fixed later: ConvolutionDecomposition will be moved upper by execution flow
//    //const auto scalar_load = std::make_shared<snippets::op::ScalarBroadcastLoad>(load->input(0).get_source_output());
//    //ngraph::copy_runtime_info(load, scalar_load);
//    //scalar_load->set_friendly_name(load->get_friendly_name());
//    //
//    //replace_node(load, scalar_load);
//    auto scalar_load = load;
//
//
//
//    const auto& target_inputs = convolution->output(0).get_target_inputs();
//    if (target_inputs.size() != 1ul) {
//        return false;
//    }
//
//    const auto parent = convolution->get_input_node_shared_ptr(0);
//    // TODO: NCHW
//    // TODO: static
//    const auto input_shape = convolution->get_input_shape(0);
//    const auto output_shape = convolution->output(0).get_shape();
//    // TODO: temporary assert
//    const size_t iterations_count = convolution->input(1).get_source_output().get_shape()[1];
//
//    const auto loop = std::make_shared<snippets::op::Loop>(parent, parent, iterations_count);
//    loop->set_friendly_name(convolution->get_friendly_name() + "_loop");
//    loop->get_rt_info()["order"] = static_cast<size_t>(1ull);
//
//    const auto convolution_kernel = std::make_shared<snippets::op::Convolution1x1Kernel>(
//        loop->output(0),
//        convolution->get_input_node_shared_ptr(1),
//        biases,
//        12);
//    ngraph::copy_runtime_info(convolution, convolution_kernel);
//    convolution_kernel->set_friendly_name(convolution->get_friendly_name());
//    convolution_kernel->get_rt_info()["order"] = static_cast<size_t>(2ull);
//
//    //const auto parent_output = parent->output(0);
//    //loop->input(0).replace_source_output(parent_output);
//    //parent_output.remove_target_input(convolution->input(0));
//
//
//    std::vector<std::shared_ptr<Node>> nodes;
//    // TODO: get the latest only
//    // TODO: return inputs (not nodes)
//    auto next = biases_add->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
//    get_nodes_before_result(next, true, nodes);
//    // TODO: to debug only
//    //assert(nodes.size() == 2ul);
//
//    assert(nodes.size() > 0);
//
//    auto first = nodes[0];
//    auto last = nodes.back();
//    const auto return_input = *last->output(0).get_target_inputs().begin();
//    // TODO: to debug only
//    assert(is_type<opset1::Result>(return_input.get_node()));
//
//
//    const auto auto_loop_jump = std::make_shared<snippets::op::ConditionalJump>(OutputVector{ last->output(0) });
//    auto_loop_jump->set_friendly_name(convolution_kernel->get_friendly_name() + "_auto_loop_jump");
//    auto_loop_jump->get_rt_info()["order"] = static_cast<size_t>(6ull);
//
//    auto auto_loop_inputs = convolution_kernel->outputs();
//    auto_loop_inputs.push_back(auto_loop_jump->output(0));
//    const auto auto_loop = std::make_shared<snippets::op::AutoLoop>(auto_loop_inputs);
//    auto_loop->set_friendly_name(convolution_kernel->get_friendly_name() + "_auto_loop");
//    auto_loop->get_rt_info()["order"] = static_cast<size_t>(3ull);
//
//
//
//    first->input(0).replace_source_output(auto_loop->output(0));
//    first->get_rt_info()["order"] = static_cast<size_t>(4ull);
//
//
//    convolution->clear_control_dependents();
//    convolution->clear_control_dependencies();
//
//    //child_input.replace_source_output(ch_conditional_jump->output(1));
//
//
//    {
//        // TODO: just to check
//        assert(loop->output(0).get_target_inputs().size() == 1ul);
//        //assert(ch_conditional_jump->output(0).get_target_inputs().size() == 1ul);
//        //const auto expected_loop = ch_conditional_jump->output(0).get_target_inputs().begin()->get_node();
//        //assert(expected_loop == ch_loop.get());
//    }
//
//
//    //return_input.replace_source_output(for_loop->output(1));
//    //for_loop->input(1).replace_source_output(auto_loop_jump->output(1));
//
//    last->get_rt_info()["order"] = static_cast<size_t>(5ull);
//
//
//    const auto loop_jump = std::make_shared<snippets::op::ConditionalJump>(OutputVector{ auto_loop_jump->output(1) });
//    loop_jump->set_friendly_name(convolution_kernel->get_friendly_name() + "_loop_jump");
//    loop_jump->get_rt_info()["order"] = static_cast<size_t>(7ull);
//
//    loop->input(1).replace_source_output(loop_jump->output(0));
//
//    return_input.replace_source_output(loop_jump->output(1));
//
//    // TODO: will be covered by tests
//    assert(convolution_kernel->output(0).get_target_inputs().size() == 1ul);
//
//    return true;
//}

//bool decompose_1x1_by_filter_with_loop(
//    const std::shared_ptr<ngraph::opset1::Convolution>& convolution,
//    const std::shared_ptr<opset1::GroupConvolution>& group_convolution) {
//    const auto biases_add = convolution->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
//    const auto biases = biases_add->get_input_node_shared_ptr(1ul);
//    const auto parent = convolution->get_input_node_shared_ptr(0);
//    // TODO: NCHW
//    // TODO: static
//    const auto input_shape = convolution->get_input_shape(0);
//    const auto output_shape = convolution->output(0).get_shape();
//    // TODO: temporary hardcoded
//    const auto filter_volume = 9ull;
//    const auto iterations_count = 1ull;
//
//    const auto loop = std::make_shared<snippets::op::Loop>(parent, parent, iterations_count);
//    loop->set_friendly_name(convolution->get_friendly_name() + "_loop");
//    loop->get_rt_info()["order"] = static_cast<size_t>(1ull);
//
//    const auto convolution_kernel = std::make_shared<snippets::op::ConvolutionMerged1x1Kernel>(
//        loop,
//        convolution->get_input_node_shared_ptr(1),
//        biases,
//        filter_volume);
//    ngraph::copy_runtime_info(convolution, convolution_kernel);
//    convolution_kernel->set_friendly_name(convolution->get_friendly_name());
//    convolution_kernel->get_rt_info()["order"] = static_cast<size_t>(2ull);
//
//    const auto convolution_after = biases_add->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
//
//    auto convolution_child_input = *biases_add->output(0).get_target_inputs().begin();
//    convolution_child_input.replace_source_output(convolution_kernel->output(0));
//    convolution->get_input_source_output(0).remove_target_input(convolution->input(0));
//    convolution->get_input_source_output(1).remove_target_input(convolution->input(1));
//    //convolution->output(0).remove_target_input(*convolution->output(0).get_target_inputs().begin());
//
//
//    std::vector<std::shared_ptr<Node>> nodes1;
//    get_nodes(convolution_after, group_convolution, nodes1);
//    assert(nodes1.size() > 0);
//
//    auto first1 = nodes1[0];
//    auto last1 = nodes1.back();
//    //const auto return_input = *last->output(0).get_target_inputs().begin();
//    //// TODO: to debug only
//    //assert(is_type<opset1::Result>(return_input.get_node()));
//
//    // TODO: do we really need it?
//    //first1->input(0).replace_source_output(convolution_kernel->output(0));
//
//    const auto group_biases_add = group_convolution->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
//    const auto group_biases = group_biases_add->get_input_node_shared_ptr(1ul);
//
//    //std::vector<Output<Node>> inputs;
//    //for (auto i = 0ull; i < convolution_kernel->get_output_size(); ++i) {
//    //    inputs.push_back(last1);
//    //}
//
//    std::map<size_t, std::string> original_names;
//    std::vector<Output<Node>> convolution_dw_kernel_inputs;
//    for (auto i = 0ull; i < filter_volume; ++i) {
//        auto output = convolution_kernel->output(i);
//        for (auto node_index = 0ull; node_index < nodes1.size(); ++node_index) {
//            const auto& node = nodes1[node_index];
//            if (i == 0) {
//                original_names[node_index] = node->get_friendly_name();
//            }
//
//            auto new_node = i == 0 ? node : node->clone_with_new_inputs({ output });
//            new_node->set_friendly_name(original_names[node_index] + "_" + std::to_string(i));
//            output = new_node->output(0);
//
//            if (i == 0) {
//                while (output.get_target_inputs().size() != 0ull) {
//                    output.remove_target_input(*output.get_target_inputs().begin());
//                }
//            }
//        }
//
//        convolution_dw_kernel_inputs.push_back(output);
//    }
//
//    const auto convolution_dw_kernel = std::make_shared<snippets::op::ConvolutionMergedDwKernel>(
//        convolution_dw_kernel_inputs,
//        group_convolution->get_input_node_shared_ptr(1),
//        group_biases,
//        group_convolution->get_strides(),
//        group_convolution->get_pads_begin(),
//        group_convolution->get_pads_end(),
//        group_convolution->get_dilations(),
//        group_convolution->get_auto_pad(),
//        1ul);
//    ngraph::copy_runtime_info(group_convolution, convolution_dw_kernel);
//    convolution_dw_kernel->set_friendly_name(group_convolution->get_friendly_name());
//
//    const auto group_convolution_after = group_biases_add->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
//
//    auto group_convolution_child_input = *group_biases_add->output(0).get_target_inputs().begin();
//    group_convolution_child_input.replace_source_output(convolution_dw_kernel->output(0));
//    group_convolution->get_input_source_output(0).remove_target_input(group_convolution->input(0));
//    group_convolution->get_input_source_output(1).remove_target_input(group_convolution->input(1));
//
//    std::vector<std::shared_ptr<Node>> nodes2;
//    get_nodes_before_result(group_convolution_after, false, nodes2);
//    assert(nodes2.size() > 0);
//
//    auto first2 = nodes2[0];
//    auto last2 = nodes2.back();
//
//    const auto& result_target_inputs = last2->output(0).get_target_inputs();
//
//    // TODO: not neccessary
//    //std::vector<Output<Node>> conditional_jump_inputs(filter_volume);
//    //for (auto i = 0ull; i < filter_volume; ++i) {
//    //    auto output = convolution_dw_kernel->output(i);
//    //    for (auto node_index = 0ull; node_index < nodes2.size(); ++node_index) {
//    //        const auto& node = nodes2[node_index];
//    //        if (i == 0) {
//    //            original_names[node_index] = node->get_friendly_name();
//    //        }
//
//    //        auto new_node = i == 0 ? node : node->clone_with_new_inputs({ output });
//    //        new_node->set_friendly_name(original_names[node_index] + "_" + std::to_string(i));
//    //        output = new_node->output(0);
//
//    //        if (i == 0) {
//    //            while (output.get_target_inputs().size() != 0ull) {
//    //                output.remove_target_input(*output.get_target_inputs().begin());
//    //            }
//    //        }
//    //    }
//    //    conditional_jump_inputs[i] = output;
//    //}
//    //const auto conditional_jump = std::make_shared<snippets::op::ConditionalJump>(conditional_jump_inputs);
//
//    const auto conditional_jump = std::make_shared<snippets::op::ConditionalJump>(std::vector<Output<Node>>{ last2->output(0) });
//
//    conditional_jump->set_friendly_name(convolution->get_friendly_name() + "_jump");
//    conditional_jump->get_rt_info()["order"] = static_cast<size_t>(3ull);
//
//    loop->input(1).replace_source_output(conditional_jump->output(0));
//
//    const auto result_input = result_target_inputs.begin();
//    result_input->replace_source_output(conditional_jump->output(1));
//
//    convolution->clear_control_dependents();
//    convolution->clear_control_dependencies();
//
//    group_convolution->clear_control_dependents();
//    group_convolution->clear_control_dependencies();
//
//    return true;
//}

bool decompose_1x1_by_filter(
        const std::shared_ptr<ngraph::opset1::Convolution>& convolution,
        const std::shared_ptr<opset1::GroupConvolution>& group_convolution) {
    const auto biases_add = convolution->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
    const auto biases = biases_add->get_input_node_shared_ptr(1ul);
    const auto parent = convolution->get_input_node_shared_ptr(0);
    // TODO: NCHW
    // TODO: static
    const auto input_shape = convolution->get_input_shape(0);
    const auto output_shape = convolution->output(0).get_shape();
    // TODO: temporary hardcoded
    const auto filter_volume = 9ull;

    const auto convolution_kernel = std::make_shared<snippets::op::ConvolutionMerged1x1Kernel>(
            parent,
            convolution->get_input_node_shared_ptr(1),
            biases,
            filter_volume);
    ngraph::copy_runtime_info(convolution, convolution_kernel);
    convolution_kernel->set_friendly_name(convolution->get_friendly_name());
    convolution_kernel->get_rt_info()["order"] = static_cast<size_t>(2ull);

    const auto convolution_after = biases_add->output(0).get_target_inputs().begin()->get_node()->shared_from_this();

    auto convolution_child_input = *biases_add->output(0).get_target_inputs().begin();
    convolution_child_input.replace_source_output(convolution_kernel->output(0));
    convolution->get_input_source_output(0).remove_target_input(convolution->input(0));
    convolution->get_input_source_output(1).remove_target_input(convolution->input(1));
    //convolution->output(0).remove_target_input(*convolution->output(0).get_target_inputs().begin());


    std::vector<std::shared_ptr<Node>> nodes1;
    get_nodes(convolution_after, group_convolution, nodes1);
    assert(nodes1.size() > 0);

    auto first1 = nodes1[0];
    auto last1 = nodes1.back();
    //const auto return_input = *last->output(0).get_target_inputs().begin();
    //// TODO: to debug only
    //assert(is_type<opset1::Result>(return_input.get_node()));

    // TODO: do we really need it?
    //first1->input(0).replace_source_output(convolution_kernel->output(0));

    const auto group_biases_add = group_convolution->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
    const auto group_biases = group_biases_add->get_input_node_shared_ptr(1ul);

    //std::vector<Output<Node>> inputs;
    //for (auto i = 0ull; i < convolution_kernel->get_output_size(); ++i) {
    //    inputs.push_back(last1);
    //}

    std::map<size_t, std::string> original_names;
    std::vector<Output<Node>> convolution_dw_kernel_inputs;
    for (auto i = 0ull; i < filter_volume; ++i) {
        auto output = convolution_kernel->output(i);
        for (auto node_index = 0ull; node_index < nodes1.size(); ++node_index) {
            const auto& node = nodes1[node_index];
            if (i == 0) {
                original_names[node_index] = node->get_friendly_name();
            }

            auto new_node = i == 0 ? node : node->clone_with_new_inputs({ output });
            new_node->set_friendly_name(original_names[node_index] + "_" + std::to_string(i));
            output = new_node->output(0);

            if (i == 0) {
                while (output.get_target_inputs().size() != 0ull) {
                    output.remove_target_input(*output.get_target_inputs().begin());
                }
            }
        }

        convolution_dw_kernel_inputs.push_back(output);
    }

    const auto convolution_dw_kernel = std::make_shared<snippets::op::ConvolutionMergedDwKernel>(
            convolution_dw_kernel_inputs,
            group_convolution->get_input_node_shared_ptr(1),
            group_biases,
            group_convolution->get_strides(),
            group_convolution->get_pads_begin(),
            group_convolution->get_pads_end(),
            group_convolution->get_dilations(),
            group_convolution->get_auto_pad(),
            1ul);
    ngraph::copy_runtime_info(group_convolution, convolution_dw_kernel);
    convolution_dw_kernel->set_friendly_name(group_convolution->get_friendly_name());

    const auto group_convolution_after = group_biases_add->output(0).get_target_inputs().begin()->get_node()->shared_from_this();

    auto group_convolution_child_input = *group_biases_add->output(0).get_target_inputs().begin();
    group_convolution_child_input.replace_source_output(convolution_dw_kernel->output(0));
    group_convolution->get_input_source_output(0).remove_target_input(group_convolution->input(0));
    group_convolution->get_input_source_output(1).remove_target_input(group_convolution->input(1));

    std::vector<std::shared_ptr<Node>> nodes2;
    get_nodes_before_result(group_convolution_after, false, nodes2);
    assert(nodes2.size() > 0);

    auto first2 = nodes2[0];
    auto last2 = nodes2.back();

    const auto& result_target_inputs = last2->output(0).get_target_inputs();

    const auto result_input = result_target_inputs.begin();
    result_input->replace_source_output(last2->output(0));

    convolution->clear_control_dependents();
    convolution->clear_control_dependencies();

    group_convolution->clear_control_dependents();
    group_convolution->clear_control_dependencies();

    return true;
}

//bool decompose_dw(const std::shared_ptr<ngraph::opset1::GroupConvolution>& convolution) {
//    const auto biases_add = convolution->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
//    const auto biases = biases_add->get_input_node_shared_ptr(1ul);
//
//    const auto load = get_load(convolution->get_input_node_shared_ptr(0));
//    assert(load != nullptr);
//
//    //// TODO: will be fixed later: ConvolutionDecomposition will be moved upper by execution flow
//    //const auto scalar_load = std::make_shared<snippets::op::ScalarBroadcastLoad>(load->input(0).get_source_output());
//    //ngraph::copy_runtime_info(load, scalar_load);
//    //scalar_load->set_friendly_name(load->get_friendly_name());
//    //
//    //replace_node(load, scalar_load);
//    auto scalar_load = load;
//
//
//
//    const auto& target_inputs = convolution->output(0).get_target_inputs();
//    if (target_inputs.size() != 1ul) {
//        return false;
//    }
//
//    const auto parent = convolution->get_input_node_shared_ptr(0);
//    // TODO: NCHW
//    // TODO: static
//    const auto input_shape = convolution->get_input_shape(0);
//    const auto output_shape = convolution->output(0).get_shape();
//    // TODO: temporary assert
//    const size_t iterations_count = convolution->input(1).get_source_output().get_shape()[1];
//
//    const auto loop = std::make_shared<snippets::op::Loop>(parent, parent, iterations_count);
//    loop->set_friendly_name(convolution->get_friendly_name() + "_loop");
//    loop->get_rt_info()["order"] = static_cast<size_t>(1ull);
//
//    const auto convolution_kernel = std::make_shared<snippets::op::ConvolutionDwKernel>(
//        loop->output(0),
//        convolution->get_input_node_shared_ptr(1),
//        biases,
//        12);
//    ngraph::copy_runtime_info(convolution, convolution_kernel);
//    convolution_kernel->set_friendly_name(convolution->get_friendly_name());
//    convolution_kernel->get_rt_info()["order"] = static_cast<size_t>(2ull);
//
//    //const auto parent_output = parent->output(0);
//    //loop->input(0).replace_source_output(parent_output);
//    //parent_output.remove_target_input(convolution->input(0));
//
//
//    std::vector<std::shared_ptr<Node>> nodes;
//    // TODO: get the latest only
//    // TODO: return inputs (not nodes)
//    auto next = biases_add->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
//    get_nodes_before_result(next, true, nodes);
//    // TODO: to debug only
//    assert(nodes.size() == 2ul);
//
//    assert(nodes.size() > 0);
//
//    auto first = nodes[0];
//    auto last = nodes.back();
//    const auto return_input = *last->output(0).get_target_inputs().begin();
//    // TODO: to debug only
//    assert(is_type<opset1::Result>(return_input.get_node()));
//
//
//    const auto auto_loop_jump = std::make_shared<snippets::op::ConditionalJump>(OutputVector{ last->output(0) });
//    auto_loop_jump->set_friendly_name(convolution_kernel->get_friendly_name() + "_auto_loop_jump");
//    auto_loop_jump->get_rt_info()["order"] = static_cast<size_t>(6ull);
//
//    auto auto_loop_inputs = convolution_kernel->outputs();
//    auto_loop_inputs.push_back(auto_loop_jump->output(0));
//    const auto auto_loop = std::make_shared<snippets::op::AutoLoop>(auto_loop_inputs);
//    auto_loop->set_friendly_name(convolution_kernel->get_friendly_name() + "_auto_loop");
//    auto_loop->get_rt_info()["order"] = static_cast<size_t>(3ull);
//
//
//
//    first->input(0).replace_source_output(auto_loop->output(0));
//    first->get_rt_info()["order"] = static_cast<size_t>(4ull);
//
//
//    convolution->clear_control_dependents();
//    convolution->clear_control_dependencies();
//
//    //child_input.replace_source_output(ch_conditional_jump->output(1));
//
//
//    {
//        // TODO: just to check
//        assert(loop->output(0).get_target_inputs().size() == 1ul);
//        //assert(ch_conditional_jump->output(0).get_target_inputs().size() == 1ul);
//        //const auto expected_loop = ch_conditional_jump->output(0).get_target_inputs().begin()->get_node();
//        //assert(expected_loop == ch_loop.get());
//    }
//
//
//    //return_input.replace_source_output(for_loop->output(1));
//    //for_loop->input(1).replace_source_output(auto_loop_jump->output(1));
//
//    last->get_rt_info()["order"] = static_cast<size_t>(5ull);
//
//
//    const auto loop_jump = std::make_shared<snippets::op::ConditionalJump>(OutputVector{ auto_loop_jump->output(1) });
//    loop_jump->set_friendly_name(convolution_kernel->get_friendly_name() + "_loop_jump");
//    loop_jump->get_rt_info()["order"] = static_cast<size_t>(7ull);
//
//    loop->input(1).replace_source_output(loop_jump->output(0));
//
//    return_input.replace_source_output(loop_jump->output(1));
//
//    // TODO: will be covered by tests
//    assert(convolution_kernel->output(0).get_target_inputs().size() == 1ul);
//
//    return true;
//}

} // namespace

ConvolutionDecomposition::ConvolutionDecomposition() {
    MATCHER_SCOPE(ConvolutionDecomposition);

    //auto matcher = ngraph::pattern::wrap_type<opset1::Convolution>();

    //auto matcher = ngraph::pattern::wrap_type<opset1::Convolution>({
    //    ngraph::pattern::wrap_type<opset1::Multiply>(),
    //    std::make_shared<pattern::op::Or>(OutputVector {
    //        pattern::wrap_type<opset1::Multiply>(),
    //        pattern::wrap_type<opset1::FakeQuantize>()
    //    })
    //    });


    auto convolution = pattern::wrap_type<opset1::Convolution>();
    auto groupConvolution = pattern::wrap_type<opset1::GroupConvolution>();
    auto matcher = std::make_shared<pattern::op::Or>(OutputVector{ convolution, groupConvolution });

    ngraph::graph_rewrite_callback callback = [&](ngraph::pattern::Matcher &m) -> bool {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::ConvolutionDecomposition, "Snippets::ConvolutionDecomposition")
        auto root = m.get_match_root();
        if (transformation_callback(root)) {
            return false;
        }

        // TODO: nGraph has to be premarket before
        Shape target_convolution_filter;

        const auto convolution = as_type_ptr<opset1::Convolution>(root);
        if (convolution != nullptr) {
            std::shared_ptr<opset1::GroupConvolution> group_convolution;

            std::stack<Output<Node>> outputs;
            outputs.push(convolution->output(0));

            while (outputs.size() != 0) {
                auto output = outputs.top();
                outputs.pop();
                const auto& target_inputs = output.get_target_inputs();
                assert(target_inputs.size() == 1ull);

                auto node = target_inputs.begin()->get_node()->shared_from_this();
                if (is_type<opset1::Result>(node) || (node->get_output_size() == 0ull)) {
                    break;
                }

                assert(node->get_output_size() == 1ull);

                group_convolution = as_type_ptr<opset1::GroupConvolution>(node);
                if (group_convolution != nullptr) {
                    const auto shape = group_convolution->get_input_node_shared_ptr(1)->output(0).get_shape();
                    target_convolution_filter = {shape[shape.size() - 4ull], shape[shape.size() - 3ull]};
                    break;
                }

                outputs.push(node->output(0));
            }

            //return group_convolution == nullptr ?
            //    decompose_1x1_to_single_optimal(convolution) :
            //    decompose_1x1_by_filter(convolution, group_convolution);

            assert(group_convolution != nullptr);
            return decompose_1x1_by_filter(convolution, group_convolution);
        }

        // decompose_1x1_by_kernel

        //const auto group_convolution = as_type_ptr<opset1::GroupConvolution>(root);
        //if (group_convolution != nullptr) {
        //    return decompose_dw(group_convolution);
        //}

        throw ov::Exception("unexpected convolution");
    };

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name), callback);
}

} // namespace pass
} // namespace snippets
} // namespace ngraph
