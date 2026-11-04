// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vision_merger_compiled_model.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <utility>

#include "../../logging.hpp"
#include "../../util.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace {
std::string port_name(const ov::Output<const ov::Node>& port) {
    const auto& names = port.get_names();
    return names.empty() ? port.get_node()->get_friendly_name() : port.get_any_name();
}

std::vector<std::size_t> dynamic_dims(const ov::PartialShape& shape) {
    OPENVINO_ASSERT(shape.rank().is_static(), "NPUW_VISION_MERGER: ports of a dynamic rank are not supported");
    std::vector<std::size_t> dims;
    for (std::size_t d = 0; d < shape.size(); ++d) {
        if (shape[d].is_dynamic()) {
            dims.push_back(d);
        }
    }
    return dims;
}

// The first `len` elements of `tensor` along each of `dims`.
ov::SoPtr<ov::ITensor> prefix_view(ov::SoPtr<ov::ITensor> tensor,
                                   const std::vector<std::size_t>& dims,
                                   const ov::Shape& valid_shape) {
    for (const auto d : dims) {
        tensor = ov::npuw::util::view(tensor, d, 0, valid_shape[d]);
    }
    return tensor;
}

// A [1, N, N] float mask: valid queries must not attend to the padded keys.
bool is_square_float_mask(const ov::Output<const ov::Node>& port, const std::vector<std::size_t>& dims) {
    return port.get_element_type() == ov::element::f32 && port.get_partial_shape().size() == 3 &&
           dims == std::vector<std::size_t>{1, 2} && port_name(port).find("attention_mask") != std::string::npos;
}

void block_padded_keys(const ov::SoPtr<ov::ITensor>& mask, std::size_t valid_len) {
    const auto& shape = mask->get_shape();
    if (valid_len >= shape[2]) {
        return;
    }
    auto blocked =
        ov::npuw::util::view(ov::npuw::util::view(mask, 1, 0, valid_len), 2, valid_len, shape[2] - valid_len);
    const auto& view_shape = blocked->get_shape();
    const auto& strides = blocked->get_strides();
    auto* base = static_cast<uint8_t*>(blocked->data());
    for (std::size_t b = 0; b < view_shape[0]; ++b) {
        for (std::size_t r = 0; r < view_shape[1]; ++r) {
            for (std::size_t c = 0; c < view_shape[2]; ++c) {
                *reinterpret_cast<float*>(base + b * strides[0] + r * strides[1] + c * strides[2]) =
                    -std::numeric_limits<float>::infinity();
            }
        }
    }
}
}  // anonymous namespace

ov::npuw::VisionMergerCompiledModel::VisionMergerCompiledModel(const std::shared_ptr<ov::Model>& model,
                                                               const std::shared_ptr<const ov::IPlugin>& plugin,
                                                               const ov::AnyMap& properties)
    : ov::npuw::ICompiledModel(model, plugin) {
    const auto enabled_key = ov::intel_npu::npuw::vision_merger::enabled.name();
    const auto max_len_key = ov::intel_npu::npuw::vision_merger::max_len.name();

    const auto max_len_it = properties.find(max_len_key);
    m_max_len = max_len_it != properties.end() ? max_len_it->second.as<std::size_t>()
                                               : ::intel_npu::NPUW_VISION_MERGER_MAX_LEN::defaultValue();
    OPENVINO_ASSERT(m_max_len > 0, "NPUW_VISION_MERGER_MAX_LEN must be greater than 0");

    std::map<std::size_t, ov::PartialShape> static_shapes;
    for (std::size_t i = 0; i < model->inputs().size(); ++i) {
        auto shape = model->input(i).get_partial_shape();
        m_in_dyn_dims.push_back(dynamic_dims(shape));
        for (const auto d : m_in_dyn_dims.back()) {
            shape[d] = m_max_len;
        }
        if (!m_in_dyn_dims.back().empty()) {
            static_shapes[i] = shape;
        }
    }
    OPENVINO_ASSERT(!static_shapes.empty(), "NPUW_VISION_MERGER: the model has no dynamic input dimension to pin");
    for (std::size_t i = 0; i < model->outputs().size(); ++i) {
        m_out_dyn_dims.push_back(dynamic_dims(model->output(i).get_partial_shape()));
    }

    auto static_model = model->clone();
    static_model->reshape(static_shapes);
    OPENVINO_ASSERT(!static_model->is_dynamic(),
                    "NPUW_VISION_MERGER: the model is still dynamic after its inputs are pinned to ",
                    m_max_len);

    auto inner_properties = properties;
    inner_properties.erase(enabled_key);
    inner_properties.erase(max_len_key);
    LOG_INFO("Compiling the vision merger for " << m_max_len << " tokens");
    m_inner = ov::npuw::ICompiledModel::create(static_model, plugin, inner_properties);
}

ov::npuw::VisionMergerCompiledModel::VisionMergerCompiledModel(const std::shared_ptr<ov::Model>& model,
                                                               const std::shared_ptr<ov::npuw::ICompiledModel>& inner,
                                                               const std::shared_ptr<const ov::IPlugin>& plugin,
                                                               std::size_t max_len)
    : ov::npuw::ICompiledModel(model, plugin),
      m_inner(inner),
      m_max_len(max_len) {
    for (const auto& input : model->inputs()) {
        m_in_dyn_dims.push_back(dynamic_dims(input.get_partial_shape()));
    }
    for (const auto& output : model->outputs()) {
        m_out_dyn_dims.push_back(dynamic_dims(output.get_partial_shape()));
    }
}

void ov::npuw::VisionMergerCompiledModel::export_model(std::ostream&) const {
    OPENVINO_THROW("NPUW_VISION_MERGER: exporting the compiled model is not supported");
}

std::shared_ptr<const ov::Model> ov::npuw::VisionMergerCompiledModel::get_runtime_model() const {
    return m_inner->get_runtime_model();
}

void ov::npuw::VisionMergerCompiledModel::set_property(const ov::AnyMap& properties) {
    m_inner->set_property(properties);
}

ov::Any ov::npuw::VisionMergerCompiledModel::get_property(const std::string& name) const {
    if (name == ov::intel_npu::npuw::vision_merger::enabled.name()) {
        return true;
    }
    if (name == ov::intel_npu::npuw::vision_merger::max_len.name()) {
        return static_cast<uint64_t>(m_max_len);
    }
    return m_inner->get_property(name);
}

void ov::npuw::VisionMergerCompiledModel::release_memory() {
    m_inner->release_memory();
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::VisionMergerCompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const VisionMergerCompiledModel>(shared_from_this());
    return std::make_shared<VisionMergerInferRequest>(self, m_inner->create_infer_request());
}

ov::npuw::VisionMergerInferRequest::VisionMergerInferRequest(
    const std::shared_ptr<const VisionMergerCompiledModel>& compiled_model,
    std::shared_ptr<ov::IAsyncInferRequest> inner_request)
    : ov::ISyncInferRequest(compiled_model),
      m_model(compiled_model),
      m_inner(std::move(inner_request)) {
    OPENVINO_ASSERT(m_inner != nullptr, "NPUW_VISION_MERGER: the inner compiled model returned a null request");

    m_profile.report_on_die = ov::npuw::profiling_enabled();
    m_profile.area = "vision_merger/execution";

    // The base request leaves the public tensors empty. Callers (GenAI) read the static dimensions of a
    // dynamic input off its tensor before filling it, so keep them and zero only the dynamic ones.
    for (const auto& ports : {get_inputs(), get_outputs()}) {
        for (const auto& port : ports) {
            allocate_tensor(port, [&port](ov::SoPtr<ov::ITensor>& tensor) {
                const auto& partial_shape = port.get_partial_shape();
                ov::Shape shape;
                for (const auto& dim : partial_shape) {
                    shape.push_back(dim.is_static() ? dim.get_length() : 0);
                }
                tensor = ov::get_tensor_impl(ov::Tensor(port.get_element_type(), shape));
            });
        }
    }
}

std::size_t ov::npuw::VisionMergerInferRequest::valid_len() const {
    const auto& ports = get_inputs();
    std::optional<std::size_t> len;
    for (std::size_t i = 0; i < ports.size(); ++i) {
        const auto& dims = m_model->dynamic_input_dims()[i];
        if (dims.empty()) {
            continue;
        }
        const auto tensor = get_tensor(ports[i]);
        OPENVINO_ASSERT(tensor && tensor->get_shape().size() > dims.back(),
                        "NPUW_VISION_MERGER: input '",
                        port_name(ports[i]),
                        "' is not set");
        for (const auto d : dims) {
            const auto dim_len = tensor->get_shape()[d];
            OPENVINO_ASSERT(!len || *len == dim_len,
                            "NPUW_VISION_MERGER: the dynamic dimensions of the inputs disagree, input '",
                            port_name(ports[i]),
                            "' has ",
                            dim_len,
                            " where ",
                            *len,
                            " is expected");
            len = dim_len;
        }
    }
    OPENVINO_ASSERT(len && *len > 0 && *len <= m_model->max_len(),
                    "NPUW_VISION_MERGER: ",
                    len ? *len : 0,
                    " tokens do not fit NPUW_VISION_MERGER_MAX_LEN=",
                    m_model->max_len());
    return *len;
}

void ov::npuw::VisionMergerInferRequest::infer() {
    std::lock_guard<std::mutex> lock(m_mutex);

    const auto len = valid_len();
    const auto max_len = m_model->max_len();
    const auto& in_ports = get_inputs();
    const auto& out_ports = get_outputs();
    const auto& inner_in_ports = m_model->inner()->inputs();
    const auto& inner_out_ports = m_model->inner()->outputs();

    m_profile["1.pad_inputs"].record([&]() {
        for (std::size_t i = 0; i < in_ports.size(); ++i) {
            const auto& dims = m_model->dynamic_input_dims()[i];
            const auto src = get_tensor(in_ports[i]);
            if (dims.empty()) {
                m_inner->set_tensor(inner_in_ports[i], src);
                continue;
            }
            auto dst = m_inner->get_tensor(inner_in_ports[i]);
            ov::npuw::util::fill_tensor_bytes(dst, 0);
            src->copy_to(prefix_view(dst, dims, src->get_shape())._ptr);
            if (is_square_float_mask(in_ports[i], dims)) {
                block_padded_keys(dst, len);
            }
        }
    });

    m_profile["2.inner_infer"].record([&]() {
        m_inner->infer();
    });

    m_profile["3.copy_outputs"].record([&]() {
        for (std::size_t i = 0; i < out_ports.size(); ++i) {
            const auto& dims = m_model->dynamic_output_dims()[i];
            const auto src = m_inner->get_tensor(inner_out_ports[i]);
            if (dims.empty()) {
                src->copy_to(get_tensor(out_ports[i])._ptr);
                continue;
            }
            auto valid_shape = src->get_shape();
            for (const auto d : dims) {
                // Outputs such as the merged vision tokens shrink by a fixed ratio against the input length.
                OPENVINO_ASSERT((valid_shape[d] * len) % max_len == 0,
                                "NPUW_VISION_MERGER: ",
                                len,
                                " tokens do not map to a whole number of rows of output '",
                                port_name(out_ports[i]),
                                "'");
                valid_shape[d] = valid_shape[d] * len / max_len;
            }
            auto dst = get_tensor(out_ports[i]);
            dst->set_shape(valid_shape);
            prefix_view(src, dims, valid_shape)->copy_to(dst._ptr);
        }
    });
}

void ov::npuw::VisionMergerInferRequest::check_tensors() const {
    // The public ports are dynamic: lengths are validated and padded by infer().
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::VisionMergerInferRequest::query_state() const {
    return {};
}

std::vector<ov::ProfilingInfo> ov::npuw::VisionMergerInferRequest::get_profiling_info() const {
    return m_inner->get_profiling_info();
}
