// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "fuse_conv_bias_activation.hpp"

#include <memory>
#include <type_traits>
#include <utility>

#include "exec_graph_info.hpp"
#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ops/gna_convolution.hpp"
#include "rt_info/gna_node_id.hpp"

using namespace ov::pass::pattern;
using namespace ov::intel_gna::op;
using namespace ov::intel_gna::rt_info;
using namespace ov::opset10;

namespace {
template <class A, class B>
std::pair<std::shared_ptr<A>, std::shared_ptr<B>> parse_eltwise_inputs(std::shared_ptr<ov::Node> node) {
    auto eltwise = std::dynamic_pointer_cast<A>(node->input(0).get_source_output().get_node_shared_ptr());
    auto constant = std::dynamic_pointer_cast<B>(node->input(1).get_source_output().get_node_shared_ptr());

    if (!eltwise) {
        eltwise = std::dynamic_pointer_cast<A>(node->input(1).get_source_output().get_node_shared_ptr());
        constant = std::dynamic_pointer_cast<B>(node->input(0).get_source_output().get_node_shared_ptr());
    }

    if (!eltwise || !constant) {
        return {nullptr, nullptr};
    }

    return {eltwise, constant};
}

struct GnaConvCallbacks {
    static bool gna_convolution_with_biasadd(Matcher& m) {
        auto eltwise = m.get_match_root();
        auto m_conv_const_pair = parse_eltwise_inputs<GNAConvolution, Constant>(eltwise);
        auto m_conv = m_conv_const_pair.first;
        auto m_const = m_conv_const_pair.second;

        if (!m_conv || !m_const) {
            return false;
        }

        if (m_conv->inputs().size() != 2) {
            return false;
        }

        if (std::dynamic_pointer_cast<Add>(eltwise) == nullptr) {
            return false;
        }

        const ov::Output<ov::Node>& data = m_conv->input(0).get_source_output();
        const ov::Output<ov::Node>& filters = m_conv->input(1).get_source_output();
        const ov::Output<ov::Node>& bias = m_const->output(0);

        std::shared_ptr<ov::Node> gna_conv = std::make_shared<GNAConvolution>(data,
                                                                              filters,
                                                                              bias,
                                                                              m_conv->get_strides(),
                                                                              m_conv->get_pads_begin(),
                                                                              m_conv->get_pads_end(),
                                                                              m_conv->get_dilations(),
                                                                              m_conv->get_auto_pad());

        gna_conv->set_friendly_name(eltwise->get_friendly_name());

        ov::copy_runtime_info({m_conv, eltwise}, gna_conv);
        set_node_id(gna_conv, get_node_id(eltwise));

        const std::string originalLayers = eltwise->get_friendly_name() + "," + m_conv->get_friendly_name();
        gna_conv->get_rt_info()[ExecGraphInfoSerialization::ORIGINAL_NAMES] = originalLayers;

        ov::replace_node(eltwise, gna_conv);
        return true;
    }

    static std::pair<std::shared_ptr<GNAConvolution>, std::shared_ptr<ov::Node>> parse_gna_conv_inputs(
        std::shared_ptr<ov::Node> add) {
        std::shared_ptr<GNAConvolution> gna_conv = nullptr;

        auto input0 = add->input(0).get_source_output().get_node_shared_ptr();
        auto input1 = add->input(1).get_source_output().get_node_shared_ptr();

        auto gna_conv0 = std::dynamic_pointer_cast<GNAConvolution>(input0);
        auto gna_conv1 = std::dynamic_pointer_cast<GNAConvolution>(input1);

        auto can_be_fused = [](const std::shared_ptr<ov::Node>& target, const std::shared_ptr<ov::Node>& fused_input) {
            return (target && fused_input &&
                    (get_node_id(target) > get_node_id(fused_input) || ov::op::util::is_constant(fused_input)));
        };

        if (gna_conv0 && gna_conv1) {
            if (can_be_fused(gna_conv0, input1)) {
                return {gna_conv0, input1};
            } else if (can_be_fused(gna_conv1, input0)) {
                return {gna_conv1, input0};
            }
        }

        if (gna_conv0 && can_be_fused(gna_conv0, input1)) {
            return {gna_conv0, input1};
        }

        if (gna_conv1 && can_be_fused(gna_conv1, input0)) {
            return {gna_conv1, input0};
        }
        return {nullptr, nullptr};
    }

    static bool sink_add_to_gna_convolution(Matcher& m) {
        auto add = std::dynamic_pointer_cast<Add>(m.get_match_root());
        auto gna_conv_node_pair = parse_gna_conv_inputs(m.get_match_root());
        auto gna_conv = gna_conv_node_pair.first;
        auto node = gna_conv_node_pair.second;

        if (!gna_conv || !node) {
            return false;
        }

        if (gna_conv->has_bias() || gna_conv->get_activation() != ActivationType::NO_ACTIVATION) {
            return false;
        }

        const ov::Output<ov::Node>& data = gna_conv->input(0).get_source_output();
        const ov::Output<ov::Node>& filters = gna_conv->input(1).get_source_output();
        const ov::Output<ov::Node>& bias = gna_conv->input(2).get_source_output();

        std::shared_ptr<ov::Node> gna_conv_add = std::make_shared<GNAConvolution>(data,
                                                                                  filters,
                                                                                  bias,
                                                                                  gna_conv->get_strides(),
                                                                                  gna_conv->get_pads_begin(),
                                                                                  gna_conv->get_pads_end(),
                                                                                  gna_conv->get_dilations(),
                                                                                  gna_conv->get_auto_pad());

        gna_conv_add->set_friendly_name(add->get_friendly_name());
        ov::copy_runtime_info({node, gna_conv}, gna_conv_add);
        set_node_id(gna_conv_add, get_node_id(add));

        auto& rt_info = gna_conv->get_rt_info();
        if (rt_info.count(ExecGraphInfoSerialization::ORIGINAL_NAMES) > 0) {
            auto& rt_info_layer_names = rt_info[ExecGraphInfoSerialization::ORIGINAL_NAMES];
            const auto original_names = rt_info_layer_names.template as<std::string>();
            const std::string original_names_with_activation = add->get_friendly_name() + "," + original_names;
            rt_info_layer_names = original_names_with_activation;
        }

        ov::replace_node(gna_conv, gna_conv_add);
        ov::replace_node(m.get_match_root(), gna_conv_add);

        return true;
    }

    static bool sink_activation_to_gna_convolution(Matcher& m) {
        auto activation_node = m.get_match_root();
        auto gna_conv = std::dynamic_pointer_cast<GNAConvolution>(
            activation_node->input(0).get_source_output().get_node_shared_ptr());
        if (gna_conv->get_activation() != ActivationType::NO_ACTIVATION) {
            return false;
        }

        ActivationType activation = ActivationType::NO_ACTIVATION;
        if (ov::is_type<Relu>(activation_node)) {
            activation = ActivationType::RELU;
        } else if (ov::is_type<Sigmoid>(activation_node)) {
            activation = ActivationType::SIGMOID;
        } else if (ov::is_type<Tanh>(activation_node)) {
            activation = ActivationType::TANH;
        } else if (ov::is_type<Log>(activation_node)) {
            activation = ActivationType::LOG;
        } else if (ov::is_type<Abs>(activation_node)) {
            activation = ActivationType::ABS;
        } else if (ov::is_type<Sign>(activation_node)) {
            activation = ActivationType::SIGN;
        } else if (ov::is_type<Clamp>(activation_node)) {
            activation = ActivationType::CLAMP;
        } else {
            return false;
        }
        gna_conv->set_activation(activation);

        gna_conv->set_friendly_name(activation_node->get_friendly_name());
        set_node_id(gna_conv, get_node_id(activation_node));

        auto& rt_info = gna_conv->get_rt_info();
        if (rt_info.count(ExecGraphInfoSerialization::ORIGINAL_NAMES) > 0) {
            auto& rt_info_layer_names = rt_info[ExecGraphInfoSerialization::ORIGINAL_NAMES];
            const auto original_names = rt_info_layer_names.template as<std::string>();
            const std::string original_names_with_activation =
                activation_node->get_friendly_name() + "," + original_names;
            rt_info_layer_names = original_names_with_activation;
        }

        ov::replace_node(m.get_match_root(), gna_conv);

        return true;
    }
};  // struct GnaConvCallbacks

bool is_bias_to_be_fused(const ov::Output<ov::Node>& output) {
    constexpr auto conv_bias_rank_min{3};
    constexpr auto conv_bias_rank_max{5};
    auto node = std::dynamic_pointer_cast<Add>(output.get_node_shared_ptr());
    if (!node) {
        return false;
    }

    auto input0 = node->input(0);
    auto input1 = node->input(1);

    const auto partial_shape0 = node->input(0).get_partial_shape();
    const auto partial_shape1 = node->input(1).get_partial_shape();

    if (partial_shape0.is_dynamic() || partial_shape1.is_dynamic()) {
        return false;
    }

    if (node->get_autob() != ov::op::AutoBroadcastType::NUMPY) {
        return false;
    }

    if (input0.get_element_type() != input1.get_element_type()) {
        return false;
    }

    const auto conv_shape = partial_shape0.to_shape();
    const auto bias_shape = partial_shape1.to_shape();
    const auto bias_rank = bias_shape.size();
    if (bias_rank < conv_bias_rank_min || bias_rank > conv_bias_rank_max) {
        return false;
    }

    // NHWC or HWC
    size_t bias_channel_index = bias_shape.size() - 1;
    size_t conv_channel_index = conv_shape.size() - 1;
    if (bias_shape.at(bias_channel_index) != conv_shape.at(conv_channel_index) &&
        bias_shape.at(bias_channel_index) != 1) {
        return false;
    }
    for (size_t i = 0; i < bias_shape.size(); i++) {
        if ((i != bias_channel_index) && (bias_shape.at(i) != 1))
            return false;
    }
    return true;
}
bool is_add_to_be_fused(const ov::Output<ov::Node>& output) {
    auto node = std::dynamic_pointer_cast<Add>(output.get_node_shared_ptr());
    if (!node) {
        return false;
    }

    auto input0 = node->input(0);
    auto input1 = node->input(1);

    const auto partial_shape0 = node->input(0).get_partial_shape();
    const auto partial_shape1 = node->input(1).get_partial_shape();

    if (input0.get_element_type() != input1.get_element_type()) {
        return false;
    }

    if (partial_shape0.is_dynamic() || partial_shape1.is_dynamic()) {
        return false;
    }
    return (partial_shape0.to_shape() == partial_shape1.to_shape());
}

bool set_nodes_order(const std::shared_ptr<ov::Model>& model, uint64_t& id) {
    for (auto& node : model->get_ordered_ops()) {
        if (auto sub_graph_node = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(node)) {
            size_t sub_graphs_num = sub_graph_node->get_internal_subgraphs_size();
            for (size_t sub_graph_ind = 0; sub_graph_ind < sub_graphs_num; ++sub_graph_ind) {
                const std::shared_ptr<ov::Model>& sub_model =
                    sub_graph_node->get_function(static_cast<int>(sub_graph_ind));
                set_nodes_order(sub_model, id);
            }
        }
        set_node_id(node, id++);
    }
    return true;
}

bool reset_nodes_order(const std::shared_ptr<ov::Model>& model) {
    for (auto& node : model->get_ordered_ops()) {
        if (auto sub_graph_node = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(node)) {
            size_t sub_graphs_num = sub_graph_node->get_internal_subgraphs_size();
            for (size_t sub_graph_ind = 0; sub_graph_ind < sub_graphs_num; ++sub_graph_ind) {
                const std::shared_ptr<ov::Model>& sub_model =
                    sub_graph_node->get_function(static_cast<int>(sub_graph_ind));
                reset_nodes_order(sub_model);
            }
        }
        remove_node_id(node);
    }
    return true;
}

}  // namespace

bool ov::intel_gna::pass::GnaFuseMarkUpNodesOrder::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(GnaFuseMarkUpNodesOrder);
    uint64_t init_id = 0;
    return set_nodes_order(m, init_id);
}

bool ov::intel_gna::pass::GnaFuseCleanUpNodesOrder::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(GnaFuseCleanUpNodesOrder);
    return reset_nodes_order(m);
}

ov::intel_gna::pass::FuseConvolutionWithBiasAdd::FuseConvolutionWithBiasAdd() {
    MATCHER_SCOPE(FuseConvolutionWithBiasAdd);
    auto conv = wrap_type<GNAConvolution>(consumers_count(1));
    auto bias = wrap_type<Constant>();
    auto add = wrap_type<Add>({conv, bias}, is_bias_to_be_fused);

    matcher_pass_callback callback = [](Matcher& m) {
        return GnaConvCallbacks::gna_convolution_with_biasadd(m);
    };

    auto m = std::make_shared<Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::intel_gna::pass::FuseConvolutionWithBiasAddAdd::FuseConvolutionWithBiasAddAdd() {
    MATCHER_SCOPE(FuseConvolutionWithBiasAddAdd);
    auto gna_convolution = wrap_type<GNAConvolution>(consumers_count(1));
    auto add1 = wrap_type<Add>({gna_convolution, any_input()}, is_add_to_be_fused);
    auto add2 = wrap_type<Add>({any_input(), gna_convolution}, is_add_to_be_fused);
    auto add = std::make_shared<::op::Or>(ov::OutputVector{add1, add2});

    matcher_pass_callback callback = [](Matcher& m) {
        return GnaConvCallbacks::sink_add_to_gna_convolution(m);
    };

    auto m = std::make_shared<Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::intel_gna::pass::SinkActivationToGnaConvolution::SinkActivationToGnaConvolution() {
    MATCHER_SCOPE(SinkActivationToGnaConvolution);
    auto gna_convolution = wrap_type<GNAConvolution>(consumers_count(1));
    auto activation = wrap_type<Relu, Sigmoid, Tanh, Abs, Log, Clamp, Sign>({gna_convolution});

    matcher_pass_callback callback = [](Matcher& m) {
        return GnaConvCallbacks::sink_activation_to_gna_convolution(m);
    };

    auto m = std::make_shared<Matcher>(activation, matcher_name);
    register_matcher(m, callback);
}

bool ov::intel_gna::pass::GnaConvolutionFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(GnaConvolutionFusion);
    ov::pass::Manager manager(get_pass_config());

    manager.register_pass<GnaFuseMarkUpNodesOrder>();

    auto fuse_conv_bias_add_activation = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(fuse_conv_bias_add_activation, FuseConvolutionWithBiasAdd)
    ADD_MATCHER(fuse_conv_bias_add_activation, FuseConvolutionWithBiasAddAdd)
    ADD_MATCHER(fuse_conv_bias_add_activation, SinkActivationToGnaConvolution)
    fuse_conv_bias_add_activation->set_name("ov::intel_gna::pass::fuse_conv_bias_add_activation");

    manager.register_pass<GnaFuseCleanUpNodesOrder>();

    manager.run_passes(m);
    return false;
}
