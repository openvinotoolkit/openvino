// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/extension/holder.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "openvino/frontend/onnx/visibility.hpp"

namespace ov {
namespace frontend {
namespace onnx {

class ONNX_FRONTEND_API FrontEnd : public ov::frontend::FrontEnd {
public:
    using Ptr = std::shared_ptr<FrontEnd>;
    std::shared_ptr<ov::Model> convert(const ov::frontend::InputModel::Ptr& model) const override;
    void convert(const std::shared_ptr<ov::Model>& partially_converted) const override;
    std::shared_ptr<ov::Model> convert_partially(const ov::frontend::InputModel::Ptr& input_model) const override;
    std::shared_ptr<ov::Model> decode(const ov::frontend::InputModel::Ptr& input_model) const override;
    std::string get_name() const override;
    bool supported_impl(const std::vector<ov::Any>& variants) const override;
    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;
    void normalize(const std::shared_ptr<ov::Model>& model) const override;

    /// \brief Single-call conversion that fuses load() and convert() for GraphIterator-backed
    /// inputs. Skips the intermediate ov::AnyVector round-trip and lets the frontend reuse one
    /// unify::InputModel instance directly. Intended for integrations (e.g. ORT EP) that already
    /// hold an in-memory graph and want to avoid the cost of building a ModelProto or the
    /// overhead of the separate load + convert API pair.
    std::shared_ptr<ov::Model> convert_from_iterator(const GraphIterator::Ptr& graph_iterator,
                                                     bool enable_mmap = false,
                                                     bool reuse_const_data = true) const;

protected:
    ov::frontend::InputModel::Ptr load_impl(const std::vector<ov::Any>& params) const override;

    void translate_graph(const InputModel::Ptr& model,
                         bool fail_fast,
                         bool /* no_conversion */,  // future use
                         std::shared_ptr<ov::Model>& ov_model) const;
    std::shared_ptr<ov::Model> convert_unify(const InputModel::Ptr& model) const;
    std::shared_ptr<ov::Model> convert_partially_unify(const InputModel::Ptr& input_model) const;
    std::shared_ptr<ov::Model> decode_unify(const InputModel::Ptr& input_model) const;

    // m_other_extensions should be the first member here,
    // m_other_extensions can contain SO Extension (holder for other Extensions),
    // so it should be released last.
    std::vector<Extension::Ptr> m_other_extensions;
    std::vector<DecoderTransformationExtension::Ptr> m_transformation_extensions;
    ExtensionHolder m_extensions;
    std::once_flag has_legacy_extension;
};

/// \brief Fused single-call conversion for GraphIterator-backed inputs. Constructs an
/// ONNX FrontEnd inside the onnx frontend DLL and invokes its convert_from_iterator
/// directly, bypassing both the FrontEndManager wrapper layer (which would hide the
/// concrete onnx FrontEnd behind a generic wrapper) and the need for cross-DLL
/// dynamic_pointer_cast. Intended for integrations (e.g. ORT EP) that already hold an
/// in-memory graph and want to skip the ov::AnyVector round-trip and I/O re-sort.
ONNX_FRONTEND_API std::shared_ptr<ov::Model> convert_from_iterator(
    const GraphIterator::Ptr& graph_iterator,
    bool enable_mmap = false,
    bool reuse_const_data = true);

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
