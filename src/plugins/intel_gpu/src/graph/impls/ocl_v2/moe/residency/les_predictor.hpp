// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/broadcast.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/moe_router_fused.hpp"
#include "lookahead_router.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/util/topk_base.hpp"
#include "openvino/runtime/core.hpp"

namespace ov::intel_gpu::ocl::moe {

// One expert-predictor head: experts = topk(sigmoid(W * cur + W' * prev) + bias).
struct LesHeadWeights {
    size_t experts = 0;
    size_t hidden = 0;
    std::vector<float> weight;
    std::vector<float> previous_weight;
    std::vector<float> bias;
};

struct LesPredictorModel {
    size_t top_k = 0;
    std::map<size_t, LesHeadWeights> heads;  // keyed by the decoder layer the head predicts
};

namespace les_detail {

inline std::shared_ptr<ov::op::v0::Constant> as_constant(ov::Output<ov::Node> output) {
    auto node = output.get_node_shared_ptr();
    while (ov::is_type<ov::op::v0::Convert>(node)) {
        node = node->input_value(0).get_node_shared_ptr();
    }
    return ov::as_type_ptr<ov::op::v0::Constant>(node);
}

inline void collect_matmuls(const ov::Output<ov::Node>& output, std::vector<std::shared_ptr<ov::op::v0::MatMul>>& matmuls) {
    auto node = output.get_node_shared_ptr();
    if (auto add = ov::as_type_ptr<ov::op::v1::Add>(node)) {
        collect_matmuls(add->input_value(0), matmuls);
        collect_matmuls(add->input_value(1), matmuls);
    } else if (auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node)) {
        matmuls.push_back(matmul);
    } else {
        OPENVINO_THROW("LES predictor: unsupported logits node ", node->get_type_name(), " ", node->get_friendly_name());
    }
}

// Returns weights as row-major [experts, hidden].
inline std::vector<float> matmul_weights(const std::shared_ptr<ov::op::v0::MatMul>& matmul, size_t& experts, size_t& hidden) {
    auto constant = as_constant(matmul->input_value(1));
    OPENVINO_ASSERT(constant && !matmul->get_transpose_a(), "LES predictor: head MatMul needs constant weights");
    const auto shape = constant->get_shape();
    OPENVINO_ASSERT(shape.size() == 2, "LES predictor: weights must be 2D");
    auto data = constant->cast_vector<float>();
    if (matmul->get_transpose_b()) {
        experts = shape[0];
        hidden = shape[1];
        return data;
    }
    experts = shape[1];
    hidden = shape[0];
    std::vector<float> transposed(data.size());
    for (size_t row = 0; row < shape[0]; ++row) {
        for (size_t column = 0; column < shape[1]; ++column) {
            transposed[column * shape[0] + row] = data[row * shape[1] + column];
        }
    }
    return transposed;
}

}  // namespace les_detail

// Reads a predictor IR whose model outputs are named `les_experts.<layer>` (see LesPredictor).
inline LesPredictorModel load_les_predictor_model(const std::shared_ptr<ov::Model>& model) {
    LesPredictorModel result;
    static const std::regex output_name("les_experts\\.(\\d+)");
    for (const auto& output : model->outputs()) {
        std::smatch match;
        std::string name;
        for (const auto& candidate : output.get_names()) {
            if (std::regex_match(candidate, match, output_name)) {
                name = candidate;
                break;
            }
        }
        if (name.empty()) {
            continue;
        }
        const size_t layer = std::stoul(match[1].str());
        const auto source = output.get_node()->input_value(0);
        auto topk = ov::as_type_ptr<ov::op::util::TopKBase>(source.get_node_shared_ptr());
        OPENVINO_ASSERT(topk && source.get_index() == 1, "LES predictor: output ", name, " must come from TopK indices");
        auto selection = ov::as_type_ptr<ov::op::v1::Add>(topk->input_value(0).get_node_shared_ptr());
        OPENVINO_ASSERT(selection, "LES predictor: TopK input of ", name, " must be Add(sigmoid, bias)");
        LesHeadWeights head;
        std::shared_ptr<ov::op::v0::Constant> bias;
        ov::Output<ov::Node> scores;
        for (size_t port = 0; port < 2; ++port) {
            if (auto constant = les_detail::as_constant(selection->input_value(port))) {
                bias = constant;
            } else {
                scores = selection->input_value(port);
            }
        }
        auto sigmoid = ov::as_type_ptr<ov::op::v0::Sigmoid>(scores.get_node_shared_ptr());
        OPENVINO_ASSERT(bias && sigmoid, "LES predictor: ", name, " must be TopK(Sigmoid(logits) + bias)");
        std::vector<std::shared_ptr<ov::op::v0::MatMul>> matmuls;
        les_detail::collect_matmuls(sigmoid->input_value(0), matmuls);
        for (const auto& matmul : matmuls) {
            auto parameter = ov::as_type_ptr<ov::op::v0::Parameter>(matmul->input_value(0).get_node_shared_ptr());
            OPENVINO_ASSERT(parameter, "LES predictor: head MatMul input must be a Parameter");
            const bool previous = parameter->get_friendly_name().find("prev") != std::string::npos;
            auto& destination = previous ? head.previous_weight : head.weight;
            OPENVINO_ASSERT(destination.empty(), "LES predictor: duplicate ", previous ? "previous" : "current", " input in ", name);
            size_t experts = 0;
            size_t hidden = 0;
            destination = les_detail::matmul_weights(matmul, experts, hidden);
            OPENVINO_ASSERT(head.experts == 0 || (head.experts == experts && head.hidden == hidden), "LES predictor: weight shape mismatch in ", name);
            head.experts = experts;
            head.hidden = hidden;
        }
        OPENVINO_ASSERT(!head.weight.empty() && !head.previous_weight.empty(), "LES predictor: ", name, " needs current and previous inputs");
        head.bias = bias->cast_vector<float>();
        OPENVINO_ASSERT(head.bias.size() == head.experts, "LES predictor: bias size mismatch in ", name);
        // Only the ranking matters; keep the bias small so it survives f16.
        const float maximum = *std::max_element(head.bias.begin(), head.bias.end());
        for (auto& value : head.bias) {
            value -= maximum;
        }
        OPENVINO_ASSERT(result.top_k == 0 || result.top_k == topk->get_k(), "LES predictor: heads disagree on top_k");
        result.top_k = topk->get_k();
        result.heads.emplace(layer, std::move(head));
    }
    OPENVINO_ASSERT(!result.heads.empty(), "LES predictor: no les_experts.<layer> outputs in ", model->get_friendly_name());
    return result;
}

// One dynamic FC and router per layer, with [current | previous] packed into each f16 input row.
class LesPredictor {
public:
    struct Head {
        uint32_t hidden = 0;
        cldnn::memory::ptr previous;
        cldnn::network::ptr network;
        LookaheadReadback readback;

        LookaheadRouter::Result execute(const cldnn::memory::ptr& current, const cldnn::event::ptr& ready, size_t count) {
            const auto rows = static_cast<int64_t>(current->get_layout().count() / hidden);
            cldnn::layout layout{ov::PartialShape{rows, 1, hidden}, cldnn::data_types::f16, cldnn::format::bfyx};
            network->set_input_data("les_current", network->get_engine().reinterpret_buffer(*current, layout));
            network->set_input_data("les_previous", previous);
            network->execute({ready});
            auto router = network->get_primitive("les_router");
            auto routed = network->get_stream().enqueue_marker({}, true);
            return readback.read(network->get_stream(), router->output_memory_ptr(1), router->output_memory_ptr(0), routed, count);
        }
    };

    LesPredictor(cldnn::network& main, const LesPredictorModel& model) {
        auto& engine = main.get_engine();
        auto stream = main.get_stream_ptr();
        constexpr auto f16 = cldnn::data_types::f16;
        cldnn::ExecutionConfig predictor_config{ov::intel_gpu::allow_new_shape_infer(true),
                            ov::intel_gpu::use_onednn(main.get_config().get_use_onednn()),
                            ov::enable_profiling(main.get_config().get_enable_profiling())};
        for (const auto& [layer, head] : model.heads) {
            cldnn::topology topology;
            const auto id = std::to_string(layer);
            const auto experts = static_cast<int64_t>(head.experts);
            const auto hidden = static_cast<int64_t>(head.hidden);
            const cldnn::layout row_layout{ov::PartialShape{1, 1, hidden}, f16, cldnn::format::bfyx};
            topology.add(cldnn::input_layout("les_current", cldnn::layout{ov::PartialShape{-1, 1, hidden}, f16, cldnn::format::bfyx}));
            topology.add(cldnn::input_layout("les_previous", row_layout));
            topology.add(cldnn::shape_of("les_shape", cldnn::input_info("les_current"), cldnn::data_types::i32));
            topology.add(cldnn::broadcast("les_previous_rows", cldnn::input_info("les_previous"), cldnn::input_info("les_shape"),
                                          ov::AxisSet{}, ov::op::BroadcastType::NUMPY));
            topology.add(cldnn::concatenation("les_in", {cldnn::input_info("les_current"), cldnn::input_info("les_previous_rows")}, 2));

            // Row e of the weights is [W[e, :] | W'[e, :]], so one FC computes both products.
            std::vector<ov::float16> packed(head.experts * 2 * head.hidden);
            for (size_t expert = 0; expert < head.experts; ++expert) {
                for (size_t column = 0; column < head.hidden; ++column) {
                    packed[expert * 2 * head.hidden + column] = ov::float16(head.weight[expert * head.hidden + column]);
                    packed[expert * 2 * head.hidden + head.hidden + column] = ov::float16(head.previous_weight[expert * head.hidden + column]);
                }
            }
            auto weights = engine.allocate_memory(cldnn::layout{ov::PartialShape{experts, 2 * hidden}, f16, cldnn::format::bfyx});
            weights->copy_from(*stream, packed.data(), true);
            topology.add(cldnn::data("les_w." + id, weights));
            topology.add(cldnn::fully_connected("les_fc." + id, cldnn::input_info("les_in"), "les_w." + id, "", 3));

            std::vector<ov::float16> bias(head.bias.begin(), head.bias.end());
            auto bias_memory = engine.allocate_memory(cldnn::layout{ov::PartialShape{experts}, f16, cldnn::format::bfyx});
            bias_memory->copy_from(*stream, bias.data(), true);
            topology.add(cldnn::data("les_bias." + id, bias_memory));
            auto eps_memory = engine.allocate_memory(cldnn::layout{ov::PartialShape{1}, f16, cldnn::format::bfyx});
            const ov::float16 eps(1e-6f);
            eps_memory->copy_from(*stream, &eps, true);
            topology.add(cldnn::data("les_eps." + id, eps_memory));

            cldnn::MoERouterFused::Config config;
            config.num_expert = head.experts;
            config.top_k = model.top_k;
            config.routing_type = cldnn::MoERouterFused::RoutingType::SIGMOID_BIAS;
            topology.add(cldnn::moe_router_fused("les_router",
                                                 {cldnn::input_info("les_fc." + id), cldnn::input_info("les_bias." + id), cldnn::input_info("les_eps." + id)},
                                                 config));
            auto program = cldnn::program::build_program(engine, topology, predictor_config, true);
            auto& compiled = m_heads[layer];
            compiled.network = cldnn::network::allocate_network(stream, program, true, true);
            compiled.hidden = static_cast<uint32_t>(hidden);
            compiled.previous = engine.allocate_memory(row_layout, cldnn::allocation_type::usm_device, false);
            compiled.previous->fill(*stream, 0);
        }
        stream->finish();
        m_top_k = model.top_k;
    }

    Head* find(size_t layer) {
        const auto found = m_heads.find(layer);
        return found == m_heads.end() ? nullptr : &found->second;
    }

    size_t top_k() const {
        return m_top_k;
    }

    size_t head_count() const {
        return m_heads.size();
    }

private:
    std::map<size_t, Head> m_heads;
    size_t m_top_k = 0;
};

inline std::optional<size_t> les_layer_of(const std::string& primitive_id) {
    static const std::regex layer_name("layers\\.(\\d+)\\.");
    std::smatch match;
    if (std::regex_search(primitive_id, match, layer_name)) {
        return std::stoul(match[1].str());
    }
    return std::nullopt;
}

}  // namespace ov::intel_gpu::ocl::moe
