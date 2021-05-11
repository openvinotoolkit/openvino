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
            /// \brief Abstract base class for sub-graph based ops, i.e ops that have sub-graph
            ///
            class NGRAPH_API MultiSubGraphOp : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                /// \brief Describes a connection between a SubGraphOp input and the body.
                class InputDescription
                {
                protected:
                    ///
                    /// \brief      Constructs a new instance.
                    ///
                    /// \param      input_index           Position of the SubGraphOp input
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

                /// \brief Describes how a SubGraphOp output is produced from the body.
                class OutputDescription
                {
                protected:
                    ///
                    /// \brief      Constructs a new instance.
                    ///
                    /// \param      body_value_index  A body value that produces the output
                    /// \param      output_index      The SubGraphOp output index
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

                class NGRAPH_API InvariantInputDescription : public InputDescription
                {
                public:
                    static constexpr type_info_t type_info{"InvariantInputDescription", 0};
                    const type_info_t& get_type_info() const override { return type_info; }
                    ///
                    /// \brief      Constructs a new instance.
                    ///
                    /// \param      input_index           Position of the SubGraphOp input
                    /// \param      body_parameter_index  Body parameter to receive input
                    ///
                    InvariantInputDescription(uint64_t input_index, uint64_t body_parameter_index);
                    InvariantInputDescription() = default;
                    std::shared_ptr<InputDescription> copy() const override;
                };

                /// \brief Produces an output from a specific iteration
                class NGRAPH_API BodyOutputDescription : public OutputDescription
                {
                public:
                    static constexpr type_info_t type_info{"BodyOutputDescription", 0};
                    const type_info_t& get_type_info() const override { return type_info; }
                    ///
                    /// \brief      Constructs a new instance.
                    ///
                    /// \param      body_value_index  A body value that produces the output
                    /// \param      output_index      The SubGraphOp output index
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
                virtual std::shared_ptr<Function> get_function(int index)
                {
                    return m_bodies[index];
                };
                virtual std::shared_ptr<const Function> get_function(int index) const
                {
                    return m_bodies[index];
                };
                virtual void add_function(const std::shared_ptr<Function>& func)
                {
                    m_bodies.push_back(func);
                }
                virtual void set_function(int index, const std::shared_ptr<Function>& func)
                {
                    m_bodies[index] = func;
                }
                /// \return a reference to the input descriptions.
                const MultiSubgraphInputDescriptionVector& get_input_descriptions(int index) const
                {
                    return m_input_descriptions[index];
                }
                /// \return a reference to the input descriptions. Can add input descriptions
                /// before
                /// validation.
                MultiSubgraphInputDescriptionVector& get_input_descriptions(int index)
                {
                    return m_input_descriptions[index];
                }
                /// \return a reference to the output descriptions.
                const MultiSubgraphOutputDescriptionVector& get_output_descriptions(int index) const
                {
                    return m_output_descriptions[index];
                }
                /// \return a reference to the output descriptions. Can add output descriptions
                /// before
                /// validation.
                MultiSubgraphOutputDescriptionVector& get_output_descriptions(int index)
                {
                    return m_output_descriptions[index];
                }
                void set_input_descriptions(int index, MultiSubgraphInputDescriptionVector inputs)
                {
                    m_input_descriptions[index] = inputs;
                }
                void set_output_descriptions(int index,
                                             MultiSubgraphOutputDescriptionVector outputs)
                {
                    m_output_descriptions[index] = outputs;
                }
                void reserve_bodies(int num_bodies);
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
