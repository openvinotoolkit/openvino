// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/parameter.hpp>
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            /// \brief Abstract base class for sub-graph based ops, i.e ops that have some
            /// sub-graphs
            ///
            class NGRAPH_API MultiSubGraphOp : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                /// \brief Abstract class describes a connection between a MultiSubGraphOp input and
                /// the body.
                class InputDescription
                {
                protected:
                    ///
                    /// \brief      Constructs a new instance.
                    ///
                    /// \param      input_index           Position of the MultiSubGraphOp input
                    /// \param      body_parameter_index  Body parameter to receive input
                    ///
                    InputDescription(uint64_t input_index, uint64_t body_parameter_index);
                    InputDescription() = default;

                public:
                    using type_info_t = DiscreteTypeInfo;
                    virtual ~InputDescription() = default;
                    virtual std::shared_ptr<InputDescription> copy() const = 0;

                    virtual const type_info_t& get_type_info() const = 0;

                    uint64_t m_input_index{0};
                    uint64_t m_body_parameter_index{0};
                };

                /// \brief Abstract class describes how a MultiSubGraphOp output is produced from
                /// the body.
                class OutputDescription
                {
                protected:
                    ///
                    /// \brief      Constructs a new instance.
                    ///
                    /// \param      body_value_index  A body value that produces the output
                    /// \param      output_index      The MultiSubGraphOp output index
                    ///
                    OutputDescription(uint64_t body_value_index, uint64_t output_index);
                    OutputDescription() = default;

                public:
                    using type_info_t = DiscreteTypeInfo;
                    virtual ~OutputDescription() = default;
                    virtual std::shared_ptr<OutputDescription> copy() const = 0;
                    virtual const type_info_t& get_type_info() const = 0;

                    uint64_t m_body_value_index{0};
                    uint64_t m_output_index{0};
                };
                /// \brief Produces an input
                class NGRAPH_API InvariantInputDescription : public InputDescription
                {
                public:
                    static constexpr type_info_t type_info{"InvariantInputDescription", 0};
                    const type_info_t& get_type_info() const override { return type_info; }
                    ///
                    /// \brief      Constructs a new instance.
                    ///
                    /// \param      input_index           Position of the MultiSubGraphOp input
                    /// \param      body_parameter_index  Body parameter to receive input
                    ///
                    InvariantInputDescription(uint64_t input_index, uint64_t body_parameter_index);
                    InvariantInputDescription() = default;
                    std::shared_ptr<InputDescription> copy() const override;
                };

                /// \brief Produces an output
                class NGRAPH_API BodyOutputDescription : public OutputDescription
                {
                public:
                    static constexpr type_info_t type_info{"BodyOutputDescription", 0};
                    const type_info_t& get_type_info() const override { return type_info; }
                    ///
                    /// \brief      Constructs a new instance.
                    ///
                    /// \param      body_value_index  A body value that produces the output
                    /// \param      output_index      The MultiSubGraphOp output index
                    ///
                    BodyOutputDescription(uint64_t body_value_index, uint64_t output_index);
                    BodyOutputDescription() = default;
                    std::shared_ptr<OutputDescription> copy() const override;
                };
                using MultiSubgraphInputDescriptionPtr =
                    std::shared_ptr<MultiSubGraphOp::InputDescription>;
                using MultiSubgraphOutputDescriptionPtr =
                    std::shared_ptr<MultiSubGraphOp::OutputDescription>;
                using MultiSubgraphInputDescriptionVector =
                    std::vector<MultiSubgraphInputDescriptionPtr>;
                using MultiSubgraphOutputDescriptionVector =
                    std::vector<MultiSubgraphOutputDescriptionPtr>;

                /// \brief     Gets internal sub-graph by index in MultiSubGraphOp
                ///
                /// \param     index sub-graph's index in op
                /// \return pointer to ngraph::Function with sub-graph
                virtual std::shared_ptr<Function> get_function(int index)
                {
                    return m_bodies[index];
                };
                /// \brief     Gets internal sub-graph by index in MultiSubGraphOp
                ///
                /// \param     index sub-graph's index in op
                /// \return pointer to ngraph::Function with sub-graph
                virtual std::shared_ptr<const Function> get_function(int index) const
                {
                    return m_bodies[index];
                };
                /// \brief     Adds sub-graph to MultiSubGraphOp
                ///
                /// \param     func new sub_graph as ngraph::Function
                virtual void add_function(const std::shared_ptr<Function>& func)
                {
                    m_bodies.push_back(func);
                }
                /// \brief     Adds sub-graph to MultiSubGraphOp
                ///
                /// \param index   index of new sub-graph
                /// \param func    func new sub_graph as ngraph::Function
                virtual void set_function(int index, const std::shared_ptr<Function>& func)
                {
                    m_bodies[index] = func;
                }
                /// \brief     Gets vector with connections beewtwen operation inputs
                /// and internal sub-graph parameters
                ///
                /// \param index   index of internal sub-graph
                /// \return vector of input descriptions
                const MultiSubgraphInputDescriptionVector& get_input_descriptions(int index) const
                {
                    return m_input_descriptions[index];
                }
                /// \brief     Gets vector with connections beewtwen operation inputs
                /// and internal sub-graph parameters
                ///
                /// \param index   index of internal sub-graph
                /// \return vector of input descriptions
                MultiSubgraphInputDescriptionVector& get_input_descriptions(int index)
                {
                    return m_input_descriptions[index];
                }
                /// \brief     Gets vector with connections beewtwen operation outputs
                /// and internal sub-graph results
                ///
                /// \param index   index of internal sub-graph
                /// \return vector of output descriptions
                const MultiSubgraphOutputDescriptionVector& get_output_descriptions(int index) const
                {
                    return m_output_descriptions[index];
                }
                /// \brief     Gets vector with connections beewtwen operation outputs
                /// and internal sub-graph results
                ///
                /// \param index   index of internal sub-graph
                /// \return vector of output descriptions
                MultiSubgraphOutputDescriptionVector& get_output_descriptions(int index)
                {
                    return m_output_descriptions[index];
                }

                /// \brief     Sets vector with connections beewtwen operation inputs
                /// and internal sub-graph parameters
                ///
                /// \param index   index of internal sub-graph
                /// \param inputs  vector of input descriptions
                void set_input_descriptions(int index, MultiSubgraphInputDescriptionVector inputs)
                {
                    m_input_descriptions[index] = inputs;
                }

                /// \brief     Sets vector with connections beewtwen operation outputs
                /// and internal sub-graph results
                ///
                /// \param index   index of internal sub-graph
                /// \param outputs vector of input descriptions
                void set_output_descriptions(int index,
                                             MultiSubgraphOutputDescriptionVector outputs)
                {
                    m_output_descriptions[index] = outputs;
                }

                ///
                /// \brief     Set input decriptions for MultiSubGraphOp input.
                ///
                /// \param      value              The value supplied as an input to the block.
                /// \param      bodies_parameters  vector of bodies parameters.
                virtual void set_invariant_inputs(const Output<Node>& value,
                                                  const ParameterVector bodies_parameters);
                ///
                /// \brief     Set output decriptions for MultiSubGraphOp output.
                ///
                /// \param      bodies_results  vector of bodies results for one output.
                /// \return     value           Output node for bodies_results.
                virtual Output<Node> set_body_outputs(ResultVector bodies_results);

                MultiSubGraphOp(const MultiSubGraphOp&) = delete;
                MultiSubGraphOp(MultiSubGraphOp&&) = default;

                MultiSubGraphOp& operator=(const MultiSubGraphOp&) = delete;
                MultiSubGraphOp& operator=(MultiSubGraphOp&&) = default;

            protected:
                // Find an input corresponding to value, adding one if necessary.
                Input<Node> input_for_value(const Output<Node>& value);

                MultiSubGraphOp() = default;
                MultiSubGraphOp(const OutputVector& args, size_t bodies_index);
                explicit MultiSubGraphOp(const OutputVector& args);

                std::vector<std::shared_ptr<Function>> m_bodies;
                std::vector<MultiSubgraphInputDescriptionVector> m_input_descriptions;
                std::vector<MultiSubgraphOutputDescriptionVector> m_output_descriptions;
            };
            using MultiSubgraphInputDescriptionPtr =
                util::MultiSubGraphOp::MultiSubgraphInputDescriptionPtr;
            using MultiSubgraphOutputDescriptionPtr =
                util::MultiSubGraphOp::MultiSubgraphOutputDescriptionPtr;
            using MultiSubgraphInputDescriptionVector =
                util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector;
            using MultiSubgraphOutputDescriptionVector =
                util::MultiSubGraphOp::MultiSubgraphOutputDescriptionVector;

        } // namespace util
    }     // namespace op

    template <>
    class NGRAPH_API AttributeAdapter<
        std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::InputDescription>>>
        : public DirectValueAccessor<
              std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::InputDescription>>>
    {
    public:
        AttributeAdapter(
            std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::InputDescription>>&
                value)
            : DirectValueAccessor<std::vector<
                  std::shared_ptr<ngraph::op::util::MultiSubGraphOp::InputDescription>>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::"
            "InputDescription>>>",
            0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    template <>
    class NGRAPH_API AttributeAdapter<
        std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::OutputDescription>>>
        : public DirectValueAccessor<
              std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::OutputDescription>>>
    {
    public:
        AttributeAdapter(
            std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::OutputDescription>>&
                value)
            : DirectValueAccessor<std::vector<
                  std::shared_ptr<ngraph::op::util::MultiSubGraphOp::OutputDescription>>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::"
            "OutputDescription>>>",
            0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
} // namespace ngraph
