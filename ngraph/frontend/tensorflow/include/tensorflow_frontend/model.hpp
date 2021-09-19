// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/input_model.hpp>
#include <frontend_manager/place.hpp>
#include <tensorflow_frontend/utility.hpp>

#include "node_context_impl.hpp"

namespace ngraph {
namespace frontend {
/// Abstract representation for an input model graph that gives nodes in topologically sorted order
class GraphIterator {
public:
    virtual size_t size() const = 0;

    /// Set iterator to the start position
    virtual void reset() = 0;

    /// Moves to the next node in the graph
    virtual void next() = 0;

    /// Returns true if iterator goes out of the range of available nodes
    virtual bool is_end() const = 0;

    /// Return NodeContext for the current node that iterator points to
    virtual std::shared_ptr<::ngraph::frontend::tensorflow::detail::TFNodeDecoder> get() const = 0;

    virtual std::shared_ptr<::ngraph::frontend::DecoderBase> get_new() const = 0;
};

class OpPlaceTF;
class TensorPlaceTF;

class TF_API InputModelTF : public InputModel {
    friend class FrontEndTF;
    class InputModelTFImpl;
    std::shared_ptr<InputModelTFImpl> _impl;

public:
    // TODO: move to private once GraphTranslation will be a part of FrontEndTF component
    std::vector<std::shared_ptr<OpPlaceTF>> get_op_places() const;
    std::map<std::string, std::shared_ptr<TensorPlaceTF>> get_tensor_places() const;
    std::map<std::string, Output<Node>> get_tensor_values() const;

    explicit InputModelTF(const std::string& path);
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    explicit InputModelTF(const std::wstring& path);
#endif
    explicit InputModelTF(const std::vector<std::istream*>& streams);

    std::vector<Place::Ptr> get_inputs() const override;
    std::vector<Place::Ptr> get_outputs() const override;
    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override;
    void override_all_outputs(const std::vector<Place::Ptr>& outputs) override;
    void override_all_inputs(const std::vector<Place::Ptr>& inputs) override;
    void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override;
    void set_partial_shape(Place::Ptr place, const ngraph::PartialShape&) override;
    ngraph::PartialShape get_partial_shape(Place::Ptr place) const override;
    void set_element_type(Place::Ptr place, const ngraph::element::Type&) override;
    void set_tensor_value(Place::Ptr place, const void* value) override;
};

}  // namespace frontend
}  // namespace ngraph
