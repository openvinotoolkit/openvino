// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "token_lookup.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../serialization.hpp"
#include "../../util.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

constexpr std::size_t chunk_size = 64;

bool depends_only_on_constants(const std::shared_ptr<ov::Node>& root) {
    std::vector<std::shared_ptr<ov::Node>> pending{root};
    std::unordered_set<const ov::Node*> visited;
    while (!pending.empty()) {
        const auto node = pending.back();
        pending.pop_back();
        if (!visited.insert(node.get()).second) {
            continue;
        }
        if (node->get_input_size() == 0 && !ov::is_type<ov::op::v0::Constant>(node)) {
            return false;
        }
        for (const auto& input : node->input_values()) {
            pending.push_back(input.get_node_shared_ptr());
        }
    }
    return true;
}

std::pair<std::size_t, std::size_t> validate_model(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(
        model->get_parameters().size() == 1 && model->get_results().size() == 1 && model->get_variables().empty(),
        "NPUW token lookup requires a stateless, single-input model.");
    const auto& input = model->get_parameters().front();
    OPENVINO_ASSERT(input->get_element_type() == ov::element::i64 && input->get_partial_shape().rank() == 2,
                    "NPUW token lookup requires a rank-2 int64 input.");

    auto output_node = model->get_results().front()->input_value(0).get_node_shared_ptr();
    if (const auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(output_node)) {
        const auto left = multiply->input_value(0).get_node_shared_ptr();
        const auto right = multiply->input_value(1).get_node_shared_ptr();
        if (depends_only_on_constants(right) && multiply->input_value(1).get_partial_shape().is_static() &&
            ov::shape_size(multiply->input_value(1).get_shape()) == 1) {
            output_node = left;
        } else if (depends_only_on_constants(left) && multiply->input_value(0).get_partial_shape().is_static() &&
                   ov::shape_size(multiply->input_value(0).get_shape()) == 1) {
            output_node = right;
        }
    }
    const auto gather = ov::as_type_ptr<ov::op::util::GatherBase>(output_node);
    OPENVINO_ASSERT(gather && gather->get_axis() == 0 && gather->get_batch_dims() == 0,
                    "NPUW token lookup models require a token-wise vocabulary lookup.");
    auto indices = gather->input_value(1).get_node_shared_ptr();
    while (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(indices)) {
        indices = convert->input_value(0).get_node_shared_ptr();
    }
    const auto vocabulary_shape = gather->input_value(0).get_partial_shape();
    OPENVINO_ASSERT(indices == input && depends_only_on_constants(gather->input_value(0).get_node_shared_ptr()) &&
                        vocabulary_shape.rank() == 2 && vocabulary_shape.is_static() &&
                        model->output().get_element_type() == ov::element::f32,
                    "NPUW token lookup requires a constant rank-2 vocabulary and float32 output.");
    return {static_cast<std::size_t>(vocabulary_shape[0].get_length()),
            static_cast<std::size_t>(vocabulary_shape[1].get_length())};
}

}  // namespace

ov::npuw::token_lookup::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                     const std::shared_ptr<const ov::IPlugin>& plugin,
                                                     const ov::AnyMap& properties)
    : ov::npuw::ICompiledModel(model, plugin) {
    std::tie(m_vocabulary_size, m_hidden_size) = validate_model(model);
    OPENVINO_ASSERT(m_vocabulary_size > 0 && m_hidden_size > 0, "NPUW token lookup vocabulary must not be empty.");
    auto inner_model = model->clone();
    inner_model->reshape({{inner_model->input().get_any_name(), ov::PartialShape{1, chunk_size}}});
    auto inner_properties = properties;
    inner_properties.erase(ov::intel_npu::npuw::token_lookup::enabled.name());
    // Keep quantized vocabulary lookup together unless the caller overrides the partitioning policy.
    inner_properties.emplace("NPUW_ONLINE_PIPELINE", "NONE");
    m_inner = ov::npuw::ICompiledModel::create(inner_model, plugin, inner_properties);
}

ov::npuw::token_lookup::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                     const std::shared_ptr<const ov::IPlugin>& plugin,
                                                     std::shared_ptr<ov::npuw::ICompiledModel> inner,
                                                     std::size_t vocabulary_size,
                                                     std::size_t hidden_size)
    : ov::npuw::ICompiledModel(model, plugin),
      m_inner(std::move(inner)),
      m_vocabulary_size(vocabulary_size),
      m_hidden_size(hidden_size) {
    OPENVINO_ASSERT(m_inner && m_vocabulary_size > 0 && m_hidden_size > 0, "Invalid NPUW token lookup blob.");
}

void ov::npuw::token_lookup::CompiledModel::export_model(std::ostream& stream) const {
    ov::npuw::s11n::write_header(stream, NPUW_TOKEN_LOOKUP_COMPILED_MODEL_INDICATOR);
    ov::npuw::s11n::write(stream, inputs());
    ov::npuw::s11n::write(stream, outputs());
    ov::npuw::s11n::write(stream, m_vocabulary_size);
    ov::npuw::s11n::write(stream, m_hidden_size);
    m_inner->export_model(stream);
}

std::shared_ptr<ov::npuw::token_lookup::CompiledModel> ov::npuw::token_lookup::CompiledModel::import_model(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    ov::npuw::s11n::read_and_check_header(stream, NPUW_TOKEN_LOOKUP_COMPILED_MODEL_INDICATOR, "token lookup");
    ov::ParameterVector parameters;
    ov::NodeVector results;
    std::size_t vocabulary_size = 0;
    std::size_t hidden_size = 0;
    ov::npuw::s11n::read(stream, parameters);
    ov::npuw::s11n::read(stream, results);
    ov::npuw::s11n::read(stream, vocabulary_size);
    ov::npuw::s11n::read(stream, hidden_size);
    OPENVINO_ASSERT(parameters.size() == 1 && results.size() == 1, "Invalid NPUW token lookup blob ports.");
    auto model = std::make_shared<ov::Model>(ov::as_output_vector(results), parameters);
    auto inner = ov::npuw::ICompiledModel::import_model(stream, plugin, properties);
    return std::shared_ptr<CompiledModel>(
        new CompiledModel(model, plugin, std::move(inner), vocabulary_size, hidden_size));
}

std::shared_ptr<const ov::Model> ov::npuw::token_lookup::CompiledModel::get_runtime_model() const {
    return m_inner->get_runtime_model();
}

void ov::npuw::token_lookup::CompiledModel::set_property(const ov::AnyMap& properties) {
    m_inner->set_property(properties);
}

ov::Any ov::npuw::token_lookup::CompiledModel::get_property(const std::string& name) const {
    if (name == ov::intel_npu::npuw::token_lookup::enabled.name()) {
        return true;
    }
    return m_inner->get_property(name);
}

void ov::npuw::token_lookup::CompiledModel::release_memory() {
    m_inner->release_memory();
}

std::size_t ov::npuw::token_lookup::CompiledModel::vocabulary_size() const {
    return m_vocabulary_size;
}

std::size_t ov::npuw::token_lookup::CompiledModel::hidden_size() const {
    return m_hidden_size;
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::token_lookup::CompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const CompiledModel>(shared_from_this());
    return std::make_shared<InferRequest>(self, m_inner->create_infer_request());
}

ov::npuw::token_lookup::InferRequest::InferRequest(const std::shared_ptr<const CompiledModel>& compiled_model,
                                                   std::shared_ptr<ov::IAsyncInferRequest> inner)
    : ov::ISyncInferRequest(compiled_model),
      m_model(compiled_model),
      m_inner(std::move(inner)) {
    OPENVINO_ASSERT(m_inner, "NPUW token lookup requires an inner infer request.");
    set_tensor(get_inputs().at(0), ov::get_tensor_impl(ov::Tensor(ov::element::i64, {1, 1})));
    set_tensor(get_outputs().at(0), ov::get_tensor_impl(ov::Tensor(ov::element::f32, {1, 1, m_model->hidden_size()})));
}

void ov::npuw::token_lookup::InferRequest::infer() {
    const auto input = get_tensor(get_inputs().at(0));
    const auto output = get_tensor(get_outputs().at(0));
    OPENVINO_ASSERT(input && output, "NPUW token lookup requires input and output tensors.");
    const auto& shape = input->get_shape();
    OPENVINO_ASSERT(input->get_element_type() == ov::element::i64 && shape.size() == 2 && shape[0] == 1 && shape[1] > 0,
                    "NPUW token lookup requires nonempty int64 token IDs with batch size 1.");
    OPENVINO_ASSERT(output->get_element_type() == ov::element::f32,
                    "NPUW token lookup requires a float32 output tensor.");
    const ov::Shape output_shape{1, shape[1], m_model->hidden_size()};
    if (output->get_shape() != output_shape) {
        output->set_shape(output_shape);
    }
    const auto* source = static_cast<const uint8_t*>(input->data());
    const auto token_stride = input->get_strides().at(1);
    auto chunk_input = m_inner->get_tensor(m_inner->get_inputs().at(0));
    OPENVINO_ASSERT(chunk_input->get_element_type() == ov::element::i64 &&
                        chunk_input->get_shape() == ov::Shape({1, chunk_size}) &&
                        chunk_input->get_strides().at(1) == sizeof(int64_t),
                    "NPUW token lookup inner input has an unexpected shape, type, or stride.");
    for (std::size_t offset = 0; offset < shape[1]; offset += chunk_size) {
        const auto count = std::min(chunk_size, shape[1] - offset);
        auto* ids = chunk_input->data<int64_t>();
        std::fill_n(ids, chunk_size, int64_t{0});
        for (std::size_t index = 0; index < count; ++index) {
            int64_t token_id;
            std::memcpy(&token_id, source + (offset + index) * token_stride, sizeof(token_id));
            OPENVINO_ASSERT(token_id >= 0 && static_cast<std::size_t>(token_id) < m_model->vocabulary_size(),
                            "NPUW token lookup token ID is outside the vocabulary.");
            ids[index] = token_id;
        }
        m_inner->infer();
        const auto chunk_output = m_inner->get_tensor(m_inner->get_outputs().at(0));
        OPENVINO_ASSERT(chunk_output->get_element_type() == ov::element::f32 &&
                            chunk_output->get_shape() == ov::Shape({1, chunk_size, m_model->hidden_size()}),
                        "NPUW token lookup returned an unexpected output shape or type.");
        const auto source_view = ov::npuw::util::view(chunk_output, 1, 0, count);
        const auto output_view = ov::npuw::util::view(output, 1, offset, count);
        source_view->copy_to(output_view._ptr);
    }
}

void ov::npuw::token_lookup::InferRequest::check_tensors() const {}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::token_lookup::InferRequest::query_state() const {
    return {};
}

std::vector<ov::ProfilingInfo> ov::npuw::token_lookup::InferRequest::get_profiling_info() const {
    return m_inner->get_profiling_info();
}
