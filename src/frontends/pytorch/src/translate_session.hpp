// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "input_model.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov::frontend::pytorch {

/// \brief Alias positions marking every element of `value` as not aliasing the base
Output<Node> make_non_alias_positions(const Output<Node>& value);

/// \brief Alias positions of every element of `value` in itself
Output<Node> make_alias_identity_positions(const Output<Node>& value);

/// \brief Replays data movement operations between `base` and `view` on `base_positions`; empty if not possible
Output<Node> replay_alias_positions(const Output<Node>& view,
                                    const Output<Node>& base,
                                    const Output<Node>& base_positions);

/// \brief Positions of `inner` view elements in the root, given `outer` positions of the tensor `inner` is taken from
Output<Node> compose_alias_positions(const Output<Node>& outer, const Output<Node>& inner);

/// For one call of convert and decode method of Frontend, it creates one TranslateSession object to save data for the
/// translation session: telemetry statistics, operation translators (including extensions) registered for this
/// translation session.
class TranslateSession {
public:
    TranslateSession(const frontend::InputModel::Ptr& input_model,
                     const std::unordered_map<std::string, CreatorFunction>& translator_map,
                     const std::shared_ptr<TelemetryExtension>& telemetry);
    ~TranslateSession();
    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(const frontend::InputModel::Ptr& input_model);

    /// \brief Completely convert pytorch_model, creates PtFrameworkNode if not possible to convert node
    /// \param pytorch_model Input model
    /// \param external_tensor_map Is used for recursive calls of convert_pytorch_model and represent the external
    /// context which is visible from nested model. Empty external_tensor_map is used as an indication that this is a
    /// main body conversion.
    /// \return fully converted OV Model
    std::shared_ptr<Model> convert_pytorch_model(std::shared_ptr<TorchDecoder> pytorch_model,
                                                 const TensorMap& external_tensor_map = {},
                                                 const std::shared_ptr<pytorch::InputModel>& input_model = nullptr);

    /// \brief Returns reverseprop operations for direct operation
    Output<Node> get_reverseprop_op(const std::shared_ptr<TorchDecoder>& node,
                                    const Output<Node>& direct_op_output,
                                    const Output<Node>& value,
                                    const Output<Node>& base = {});

    /// \brief Writes pytorch tensor index into openvino tensor
    void encode_tensor_name(Output<Node> tensor_desc,
                            size_t tensor_idx,
                            const std::vector<std::string>& additional_names = {});

    /// \brief Gets pytorch tensor index from openvino tensor
    size_t decode_tensor_name(const Output<Node>& tensor_desc);

    /// \brief Element positions of an alias in its base, created only when a mutation or rebase needs them.
    /// Materialized value is an i64 tensor with the alias shape holding flat base indices, where -1 marks elements
    /// that do not alias the base. An empty output means the alias relation cannot be represented.
    class AliasPositions {
    public:
        explicit AliasPositions(std::function<Output<Node>()> materializer) : m_materializer(std::move(materializer)) {}
        Output<Node> get();

    private:
        std::function<Output<Node>()> m_materializer;
        Output<Node> m_positions;
    };

    struct AliasInfo {
        size_t base_id;
        std::shared_ptr<TorchDecoder> decoder;
        Output<Node> output;
        // Base value used to convert and replay a view.
        Output<Node> base_value;
        // Constructed containers hold references to distinct tensors, not views of input 0.
        std::vector<size_t> element_ids;
        // Set for aliases returned from subgraphs, where the view cannot be expressed by a single view operation.
        std::shared_ptr<AliasPositions> positions = nullptr;
    };
    std::map<size_t, AliasInfo> m_may_be_alias;

    // Aliases of tuple elements returned by inlined subgraphs, keyed by tuple tensor id and element index and
    // registered when an element is selected.
    std::map<std::pair<size_t, int64_t>, AliasInfo> m_tuple_element_aliases;

    /// \brief Alias between a declared output of a converted internal body and one of its parameters
    struct SubgraphOutputAlias {
        size_t output_index;
        // Tensor id of the body parameter that is the alias root.
        size_t root_id;
        // Positions relative to the root parameter.
        std::shared_ptr<AliasPositions> positions;
    };

    /// \brief Returns and forgets aliases recorded for outputs of the internal body converted to `body`
    std::vector<SubgraphOutputAlias> take_subgraph_output_aliases(const std::shared_ptr<Model>& body);

    /// \brief Returns the root tensor id of the alias chain of `tensor_id`
    size_t get_alias_root(size_t tensor_id) const;

    /// \brief Positions of tensor `tensor_id` with current value `value` in tensor `root_id`. Elements of a tensor
    /// which is not a view are marked as not aliased; a view of a different tensor is not representable.
    std::shared_ptr<AliasPositions> get_alias_positions(size_t tensor_id, size_t root_id, const Output<Node>& value);

    /// \brief Returns value of the base after writing `value` to the alias described by `alias_info`
    Output<Node> reverseprop_alias(const AliasInfo& alias_info, const Output<Node>& value);

    /// \brief Returns alias value recomputed from the new base value
    Output<Node> rebase_alias(const AliasInfo& alias_info, const Output<Node>& new_base);

    OutputVector convert_node(const NodeContext& context);

private:
    const frontend::InputModel::Ptr m_input_model;
    const std::unordered_map<std::string, CreatorFunction>& m_translator_map;
    std::shared_ptr<TelemetryExtension> m_telemetry;
    std::shared_ptr<Model> m_ov_model;

    std::map<size_t, std::pair<size_t, Output<Node>>> m_counter_map;
    std::map<std::string, uint64_t> m_op_statistics;
    std::map<std::shared_ptr<Model>, std::vector<SubgraphOutputAlias>> m_subgraph_output_aliases;
    // Set per converted graph in convert_pytorch_model; the decoder type never varies within one.
    bool m_is_fx = false;
};

}  // namespace ov::frontend::pytorch
