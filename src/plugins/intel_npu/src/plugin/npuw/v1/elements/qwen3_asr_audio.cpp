// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qwen3_asr_audio.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "../../infer_request_utils.hpp"
#include "../../serialization.hpp"
#include "../../util.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace {

std::size_t token_count(std::size_t frames) {
    return (frames + 7) / 8;
}

ov::Output<const ov::Node> input_port(const std::vector<ov::Output<const ov::Node>>& inputs, const std::string& name) {
    auto port = ov::npuw::util::find_port_by_name(inputs, name);
    OPENVINO_ASSERT(port.has_value(), "Qwen3-ASR NPUW audio encoder is missing input '", name, "'.");
    return port.value();
}

}  // namespace

ov::npuw::qwen3_asr_audio::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                        const std::shared_ptr<const ov::IPlugin>& plugin,
                                                        const ov::AnyMap& properties)
    : ov::npuw::ICompiledModel(model, plugin) {
    OPENVINO_ASSERT(
        model->get_parameters().size() == 4 && model->get_results().size() == 1 && model->get_variables().empty(),
        "Qwen3-ASR NPUW audio encoder requires four inputs and one stateless output.");
    const auto feature = model->input("padded_feature");
    const auto mask = model->input("padded_mask_after_cnn");
    OPENVINO_ASSERT(feature.get_element_type() == ov::element::f32 && feature.get_partial_shape().rank() == 3 &&
                        feature.get_partial_shape()[1].is_static() && mask.get_element_type() == ov::element::boolean &&
                        mask.get_partial_shape().rank() == 2,
                    "Qwen3-ASR NPUW audio encoder has an unsupported feature or mask input.");
    for (const auto& name : {"aftercnn_lens", "cu_seqlens"}) {
        OPENVINO_ASSERT(model->input(name).get_target_inputs().empty(),
                        "Qwen3-ASR NPU audio encoder requires independent windows; input '",
                        name,
                        "' is used by this export.");
    }
    const auto window = properties.find(ov::intel_npu::npuw::qwen3_asr_audio::window_frames.name());
    OPENVINO_ASSERT(window != properties.end(), "Qwen3-ASR NPUW audio encoder requires a window size.");
    const auto frames = window->second.as<uint64_t>();
    OPENVINO_ASSERT(frames > 0 && frames <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - 7 &&
                        (frames + 7) / 8 <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                    "Qwen3-ASR NPUW audio encoder window size is invalid.");
    m_window_frames = static_cast<std::size_t>(frames);
    m_mel_bins = static_cast<std::size_t>(feature.get_partial_shape()[1].get_length());
    const auto output_shape = model->output().get_partial_shape();
    OPENVINO_ASSERT(model->output().get_element_type() == ov::element::f32 && output_shape.rank() == 3 &&
                        output_shape[2].is_static(),
                    "Qwen3-ASR NPUW audio encoder requires a rank-3 float32 output.");
    m_hidden_size = static_cast<std::size_t>(output_shape[2].get_length());
    OPENVINO_ASSERT(m_mel_bins > 0 && m_hidden_size > 0, "Qwen3-ASR NPUW audio dimensions must be positive.");

    auto inner_model = model->clone();
    inner_model->reshape({{"padded_feature", ov::PartialShape{1, m_mel_bins, m_window_frames}},
                          {"padded_mask_after_cnn", ov::PartialShape{1, token_count(m_window_frames)}},
                          {"aftercnn_lens", ov::PartialShape{1}},
                          {"cu_seqlens", ov::PartialShape{2}}});
    ov::preprocess::PrePostProcessor preprocessor(inner_model);
    preprocessor.input("padded_mask_after_cnn").tensor().set_element_type(ov::element::u8);
    inner_model = preprocessor.build();
    auto inner_properties = properties;
    inner_properties.erase(ov::intel_npu::npuw::qwen3_asr_audio::enabled.name());
    inner_properties.erase(ov::intel_npu::npuw::qwen3_asr_audio::window_frames.name());
    m_inner = ov::npuw::ICompiledModel::create(inner_model, plugin, inner_properties);
}

ov::npuw::qwen3_asr_audio::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                        const std::shared_ptr<const ov::IPlugin>& plugin,
                                                        std::shared_ptr<ov::npuw::ICompiledModel> inner,
                                                        std::size_t window_frames,
                                                        std::size_t mel_bins,
                                                        std::size_t hidden_size)
    : ov::npuw::ICompiledModel(model, plugin),
      m_inner(std::move(inner)),
      m_window_frames(window_frames),
      m_mel_bins(mel_bins),
      m_hidden_size(hidden_size) {
    OPENVINO_ASSERT(m_inner && m_window_frames > 0 &&
                        m_window_frames <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - 7 &&
                        token_count(m_window_frames) <= static_cast<std::size_t>(std::numeric_limits<int32_t>::max()) &&
                        m_mel_bins > 0 && m_hidden_size > 0,
                    "Invalid Qwen3-ASR NPUW audio encoder blob.");
}

void ov::npuw::qwen3_asr_audio::CompiledModel::export_model(std::ostream& stream) const {
    ov::npuw::s11n::write_header(stream, NPUW_QWEN3_ASR_AUDIO_COMPILED_MODEL_INDICATOR);
    ov::npuw::s11n::write(stream, inputs());
    ov::npuw::s11n::write(stream, outputs());
    ov::npuw::s11n::write(stream, m_window_frames);
    ov::npuw::s11n::write(stream, m_mel_bins);
    ov::npuw::s11n::write(stream, m_hidden_size);
    m_inner->export_model(stream);
}

std::shared_ptr<ov::npuw::qwen3_asr_audio::CompiledModel> ov::npuw::qwen3_asr_audio::CompiledModel::import_model(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    ov::npuw::s11n::read_and_check_header(stream, NPUW_QWEN3_ASR_AUDIO_COMPILED_MODEL_INDICATOR, "Qwen3-ASR audio");
    ov::ParameterVector parameters;
    ov::NodeVector results;
    std::size_t window_frames = 0;
    std::size_t mel_bins = 0;
    std::size_t hidden_size = 0;
    ov::npuw::s11n::read(stream, parameters);
    ov::npuw::s11n::read(stream, results);
    ov::npuw::s11n::read(stream, window_frames);
    ov::npuw::s11n::read(stream, mel_bins);
    ov::npuw::s11n::read(stream, hidden_size);
    OPENVINO_ASSERT(parameters.size() == 4 && results.size() == 1, "Invalid Qwen3-ASR NPUW audio encoder blob ports.");
    auto model = std::make_shared<ov::Model>(ov::as_output_vector(results), parameters);
    auto inner = ov::npuw::ICompiledModel::import_model(stream, plugin, properties);
    return std::shared_ptr<CompiledModel>(
        new CompiledModel(model, plugin, std::move(inner), window_frames, mel_bins, hidden_size));
}

std::shared_ptr<const ov::Model> ov::npuw::qwen3_asr_audio::CompiledModel::get_runtime_model() const {
    return m_inner->get_runtime_model();
}

void ov::npuw::qwen3_asr_audio::CompiledModel::set_property(const ov::AnyMap& properties) {
    m_inner->set_property(properties);
}

ov::Any ov::npuw::qwen3_asr_audio::CompiledModel::get_property(const std::string& name) const {
    if (name == ov::intel_npu::npuw::qwen3_asr_audio::enabled.name()) {
        return true;
    }
    if (name == ov::intel_npu::npuw::qwen3_asr_audio::window_frames.name()) {
        return static_cast<uint64_t>(m_window_frames);
    }
    return m_inner->get_property(name);
}

void ov::npuw::qwen3_asr_audio::CompiledModel::release_memory() {
    m_inner->release_memory();
}

std::size_t ov::npuw::qwen3_asr_audio::CompiledModel::window_frames() const {
    return m_window_frames;
}

std::size_t ov::npuw::qwen3_asr_audio::CompiledModel::mel_bins() const {
    return m_mel_bins;
}

std::size_t ov::npuw::qwen3_asr_audio::CompiledModel::hidden_size() const {
    return m_hidden_size;
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::qwen3_asr_audio::CompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const CompiledModel>(shared_from_this());
    return std::make_shared<InferRequest>(self, m_inner->create_infer_request());
}

ov::npuw::qwen3_asr_audio::InferRequest::InferRequest(const std::shared_ptr<const CompiledModel>& compiled_model,
                                                      std::shared_ptr<ov::IAsyncInferRequest> inner)
    : ov::ISyncInferRequest(compiled_model),
      m_model(compiled_model),
      m_inner(std::move(inner)) {
    OPENVINO_ASSERT(m_inner, "Qwen3-ASR NPUW audio encoder requires an inner infer request.");
    set_tensor(input_port(get_inputs(), "padded_feature"),
               ov::get_tensor_impl(ov::Tensor(ov::element::f32, {1, m_model->mel_bins(), 1})));
    set_tensor(input_port(get_inputs(), "padded_mask_after_cnn"),
               ov::get_tensor_impl(ov::Tensor(ov::element::boolean, {1, 1})));
    set_tensor(input_port(get_inputs(), "aftercnn_lens"), ov::get_tensor_impl(ov::Tensor(ov::element::i64, {1})));
    set_tensor(input_port(get_inputs(), "cu_seqlens"), ov::get_tensor_impl(ov::Tensor(ov::element::i32, {2})));
    set_tensor(get_outputs().at(0), ov::get_tensor_impl(ov::Tensor(ov::element::f32, {1, 1, m_model->hidden_size()})));
}

void ov::npuw::qwen3_asr_audio::InferRequest::infer() {
    const auto feature = get_tensor(input_port(get_inputs(), "padded_feature"));
    const auto mask = get_tensor(input_port(get_inputs(), "padded_mask_after_cnn"));
    const auto output = get_tensor(get_outputs().at(0));
    OPENVINO_ASSERT(feature && mask && output, "Qwen3-ASR NPUW audio encoder requires input and output tensors.");
    const auto& feature_shape = feature->get_shape();
    OPENVINO_ASSERT(feature->get_element_type() == ov::element::f32 && feature_shape.size() == 3 &&
                        feature_shape[0] > 0 && feature_shape[1] == m_model->mel_bins() && feature_shape[2] > 0 &&
                        feature_shape[2] <= m_model->window_frames(),
                    "Qwen3-ASR NPUW audio features have an unsupported shape or type.");
    const auto padded_tokens = token_count(feature_shape[2]);
    OPENVINO_ASSERT(mask->get_element_type() == ov::element::boolean &&
                        mask->get_shape() == ov::Shape({feature_shape[0], padded_tokens}),
                    "Qwen3-ASR NPUW audio mask must match the feature windows.");
    OPENVINO_ASSERT(output->get_element_type() == ov::element::f32, "Qwen3-ASR NPUW audio output must be float32.");
    const ov::Shape output_shape{feature_shape[0], padded_tokens, m_model->hidden_size()};
    if (output->get_shape() != output_shape) {
        output->set_shape(output_shape);
    }

    auto inner_feature = m_inner->get_tensor(input_port(m_inner->get_inputs(), "padded_feature"));
    auto inner_mask = m_inner->get_tensor(input_port(m_inner->get_inputs(), "padded_mask_after_cnn"));
    auto inner_lengths = m_inner->get_tensor(input_port(m_inner->get_inputs(), "aftercnn_lens"));
    auto inner_offsets = m_inner->get_tensor(input_port(m_inner->get_inputs(), "cu_seqlens"));
    OPENVINO_ASSERT(
        inner_feature->get_element_type() == ov::element::f32 &&
            inner_feature->get_shape() == ov::Shape({1, m_model->mel_bins(), m_model->window_frames()}) &&
            inner_feature->get_strides().at(2) == sizeof(float) &&
            inner_feature->get_strides().at(1) == m_model->window_frames() * sizeof(float) &&
            inner_mask->get_element_type() == ov::element::u8 &&
            inner_mask->get_shape() == ov::Shape({1, token_count(m_model->window_frames())}) &&
            inner_mask->get_strides().at(1) == sizeof(uint8_t) &&
            inner_lengths->get_element_type() == ov::element::i64 && inner_lengths->get_shape() == ov::Shape({1}) &&
            inner_offsets->get_element_type() == ov::element::i32 && inner_offsets->get_shape() == ov::Shape({2}),
        "Qwen3-ASR NPUW audio encoder has an unexpected inner input shape, type, or stride.");
    const auto* feature_data = static_cast<const uint8_t*>(feature->data());
    const auto* mask_data = static_cast<const uint8_t*>(mask->data());
    const auto feature_strides = feature->get_strides();
    const auto mask_strides = mask->get_strides();
    for (std::size_t row = 0; row < feature_shape[0]; ++row) {
        auto* destination = inner_feature->data<float>();
        std::fill_n(destination, inner_feature->get_size(), 0.0f);
        for (std::size_t mel = 0; mel < m_model->mel_bins(); ++mel) {
            const auto* source = feature_data + row * feature_strides[0] + mel * feature_strides[1];
            auto* target = destination + mel * m_model->window_frames();
            if (feature_strides[2] == sizeof(float)) {
                std::memcpy(target, source, feature_shape[2] * sizeof(float));
            } else {
                for (std::size_t frame = 0; frame < feature_shape[2]; ++frame) {
                    std::memcpy(target + frame, source + frame * feature_strides[2], sizeof(float));
                }
            }
        }
        auto* inner_mask_data = inner_mask->data<uint8_t>();
        std::fill_n(inner_mask_data, inner_mask->get_size(), uint8_t{0});
        std::size_t valid_tokens = 0;
        bool padding_seen = false;
        for (std::size_t index = 0; index < padded_tokens; ++index) {
            const bool valid = *(mask_data + row * mask_strides[0] + index * mask_strides[1]) != 0;
            OPENVINO_ASSERT(!valid || !padding_seen, "Qwen3-ASR NPUW audio mask must have a valid prefix.");
            padding_seen = padding_seen || !valid;
            inner_mask_data[index] = valid ? 1 : 0;
            valid_tokens += valid ? 1 : 0;
        }
        inner_lengths->data<int64_t>()[0] = static_cast<int64_t>(valid_tokens);
        inner_offsets->data<int32_t>()[0] = 0;
        inner_offsets->data<int32_t>()[1] = static_cast<int32_t>(valid_tokens);
        m_inner->infer();
        const auto inner_output = m_inner->get_tensor(m_inner->get_outputs().at(0));
        OPENVINO_ASSERT(inner_output->get_element_type() == ov::element::f32 &&
                            inner_output->get_shape() ==
                                ov::Shape({1, token_count(m_model->window_frames()), m_model->hidden_size()}),
                        "Qwen3-ASR NPUW audio encoder returned an unexpected output shape or type.");
        const auto source = ov::npuw::util::view(inner_output, 1, 0, padded_tokens);
        const auto destination_row = ov::npuw::util::view(output, 0, row, 1);
        source->copy_to(destination_row._ptr);
    }
}

void ov::npuw::qwen3_asr_audio::InferRequest::check_tensors() const {}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::qwen3_asr_audio::InferRequest::query_state() const {
    return {};
}

std::vector<ov::ProfilingInfo> ov::npuw::qwen3_asr_audio::InferRequest::get_profiling_info() const {
    return m_inner->get_profiling_info();
}
