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
            class NGRAPH_API SubGraphOp : public Op
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

                ///
                /// \brief      Describes a body input formed from slices of an input to
                ///             SubGraphOp.
                ///
                class NGRAPH_API SliceInputDescription : public InputDescription
                {
                public:
                    static constexpr type_info_t type_info{"SliceInputDescription", 0};
                    const type_info_t& get_type_info() const override { return type_info; }
                    ///
                    /// \brief      Constructs a new instance.
                    ///
                    /// \param      input_index           Position of the SubGraphOp input
                    /// \param      body_parameter_index  Body parameter position to receive input
                    /// \param      start                 First index for slices
                    /// \param      stride                Step amount for slices
                    /// \param      part_size             Width of slices
                    /// \param      end                   Last index for slices
                    /// \param      axis                  Axis being sliced
                    ///
                    SliceInputDescription(uint64_t input_index,
                                          uint64_t body_parameter_index,
                                          int64_t start,
                                          int64_t stride,
                                          int64_t part_size,
                                          int64_t end,
                                          int64_t axis);
                    SliceInputDescription() = default;
                    std::shared_ptr<InputDescription> copy() const override;
                    int64_t m_start{0};
                    int64_t m_stride{0};
                    int64_t m_part_size{0};
                    int64_t m_end{0};
                    int64_t m_axis{0};
                };

                ///
                /// \brief      Describes a body input initialized from a SubGraphOp input on
                ///             the first iteration, and then a body output thereafter.
                ///
                class NGRAPH_API MergedInputDescription : public InputDescription
                {
                public:
                    static constexpr type_info_t type_info{"MergedInputDescription", 0};
                    const type_info_t& get_type_info() const override { return type_info; }
                    ///
                    /// \brief      Constructs a new instance.
                    ///
                    /// \param      input_index           Position of the SubGraphOp input
                    ///                                   supplying a value to body_parameter for
                    ///                                   the initial iteration.
                    /// \param      body_parameter_index  Body parameter position to receive input.
                    /// \param      body_value_index      Body value to supply body_parameter for
                    /// successive
                    ///                                   iterations.
                    ///
                    MergedInputDescription(uint64_t input_index,
                                           uint64_t body_parameter_index,
                                           uint64_t body_value_index);
                    MergedInputDescription() = default;
                    std::shared_ptr<InputDescription> copy() const override;
                    uint64_t m_body_value_index{0};
                };

                ///
                /// \brief      Describes a body input initialized from a SubGraphOp input on
                ///             the first iteration, and invariant thereafter.
                ///
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

                /// \brief Produces an output by concatenating an output from each iteration
                class NGRAPH_API ConcatOutputDescription : public OutputDescription
                {
                public:
                    static constexpr type_info_t type_info{"ConcatOutputDescription", 0};
                    const type_info_t& get_type_info() const override { return type_info; }
                    ///
                    /// \brief      Constructs a new instance.
                    ///
                    /// \param      body_value_index  A body value that produces the output
                    /// \param      output_index      The SubGraphOp output index
                    /// \param      start             First index for slices
                    /// \param      stride            Step amount for slices
                    /// \param      part_size         Width of slices
                    /// \param      end               Last index for slices
                    /// \param      axis              Axis being sliced
                    ///
                    ConcatOutputDescription(uint64_t body_value_index,
                                            uint64_t output_index,
                                            int64_t start,
                                            int64_t stride,
                                            int64_t part_size,
                                            int64_t end,
                                            int64_t axis);
                    ConcatOutputDescription() = default;

                    std::shared_ptr<OutputDescription> copy() const override;
                    int64_t m_start{0};
                    int64_t m_stride{0};
                    int64_t m_part_size{0};
                    int64_t m_end{0};
                    int64_t m_axis{0};
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
                    /// \param      iteration         which iteration (typically -1, final) will
                    ///                               supply the value
                    ///
                    BodyOutputDescription(uint64_t body_value_index,
                                          uint64_t output_index,
                                          int64_t iteration);
                    BodyOutputDescription() = default;
                    std::shared_ptr<OutputDescription> copy() const override;
                    int64_t m_iteration{0};
                };

                virtual std::shared_ptr<Function> get_function() { return m_body; };
                virtual std::shared_ptr<const Function> get_function() const { return m_body; };
                virtual void set_function(const std::shared_ptr<Function>& func) { m_body = func; };
                /// \return a reference to the input descriptions.
                const std::vector<std::shared_ptr<InputDescription>>& get_input_descriptions() const
                {
                    return m_input_descriptions;
                }
                /// \return a reference to the input descriptions. Can add input descriptions
                /// before
                /// validation.
                std::vector<std::shared_ptr<InputDescription>>& get_input_descriptions()
                {
                    return m_input_descriptions;
                }
                /// \return a reference to the output descriptions.
                const std::vector<std::shared_ptr<OutputDescription>>&
                    get_output_descriptions() const
                {
                    return m_output_descriptions;
                }
                /// \return a reference to the output descriptions. Can add output descriptions
                /// before
                /// validation.
                std::vector<std::shared_ptr<OutputDescription>>& get_output_descriptions()
                {
                    return m_output_descriptions;
                }

                ///
                /// \brief      Indicate that a body parameter comes from slices of a value
                ///
                /// \param      parameter  The parameter to receive the slices
                /// \param      value      The value to be sliced. This will be added as an input to
                ///                        SubGraphOp.
                /// \param      start      First index on axis of the slicing
                /// \param      stride     Stepping of the slice
                /// \param      part_size  Size of the slice on axis
                /// \param      end        The last index on axis of the slicing
                /// \param      axis       The axis to slice along
                ///
                virtual void set_sliced_input(const std::shared_ptr<Parameter>& parameter,
                                              const Output<Node>& value,
                                              int64_t start,
                                              int64_t stride,
                                              int64_t part_size,
                                              int64_t end,
                                              int64_t axis);
                ///
                /// \brief      Indicates that a body parameter has an initial value in the first
                ///             iteration and computed value thereafter
                ///
                /// \param[in]  body_parameter    The body parameter
                /// \param      initial_value     Value for the parameter in first iteration. This
                ///                               will be added as an input to Loop.
                /// \param      successive_value  Value for the parameter in successive iterations.
                ///                               The value is what is active in the most recent
                ///                               completed iteration.
                ///
                virtual void set_merged_input(const std::shared_ptr<Parameter>& body_parameter,
                                              const Output<Node>& initial_value,
                                              const Output<Node>& successive_value);
                ///
                /// \brief      Indicates that a body parameter has an invariant value during
                ///             iteration that may depend on values computed outside of the
                ///             iteration.
                ///
                /// \param      body_parameter  The body parameter
                /// \param      value           The value supplied as an input to the block
                ///
                virtual void set_invariant_input(const std::shared_ptr<Parameter>& body_parameter,
                                                 const Output<Node>& value);
                ///
                /// \brief      Gets a value for a particular iteration point
                ///
                /// \param      body_value  The value
                /// \param      iteration   The iteration that supplies the value. Negative values
                ///                         are from the last iteration.
                ///                         Default value -1 (the last iteration).
                ///
                /// \return     The iterator value.
                ///
                virtual Output<Node> get_iter_value(const Output<Node>& body_value,
                                                    int64_t iteration = -1);
                ///
                /// \brief      Concatenates slices from all iterations
                ///
                /// \param      value      The value supplying slice values from each iteration.
                /// \param      start      First index on axis of the slicing
                /// \param      stride     Stepping of the slice
                /// \param      part_size  Size of the slice on axis
                /// \param      end        The last index on axis of the slicing
                /// \param      axis       The axis to slice along
                ///
                /// \return     The concatenated slices.
                ///
                virtual Output<Node> get_concatenated_slices(const Output<Node>& value,
                                                             int64_t start,
                                                             int64_t stride,
                                                             int64_t part_size,
                                                             int64_t end,
                                                             int64_t axis);

                SubGraphOp(const SubGraphOp&) = delete;
                SubGraphOp(SubGraphOp&&) = default;

                SubGraphOp& operator=(const SubGraphOp&) = delete;
                SubGraphOp& operator=(SubGraphOp&&) = default;

                int64_t get_num_iterations() const { return m_num_iterations; }

            protected:
                int64_t m_num_iterations =
                    -1; // -1 means infinity for Loop op, inconsistent for TensorIterator

                // Find an input corresponding to value, adding one if necessary.
                Input<Node> input_for_value(const Output<Node>& value);

                SubGraphOp() = default;

                explicit SubGraphOp(const OutputVector& args);

                std::shared_ptr<Function> m_body;
                std::vector<std::shared_ptr<op::util::SubGraphOp::InputDescription>>
                    m_input_descriptions;
                std::vector<std::shared_ptr<op::util::SubGraphOp::OutputDescription>>
                    m_output_descriptions;
            };
            using InputDescriptionPtr = std::shared_ptr<util::SubGraphOp::InputDescription>;
            using OutputDescriptionPtr = std::shared_ptr<util::SubGraphOp::OutputDescription>;
            using InputDescriptionVector = std::vector<InputDescriptionPtr>;
            using OutputDescriptionVector = std::vector<OutputDescriptionPtr>;
        } // namespace util
    }     // namespace op

    template <>
    class NGRAPH_API AttributeAdapter<
        std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>>>
        : public DirectValueAccessor<
              std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>>>
    {
    public:
        AttributeAdapter(
            std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>>& value)
            : DirectValueAccessor<
                  std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>>>(
                  value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::"
            "InputDescription>>>",
            0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    template <>
    class NGRAPH_API AttributeAdapter<
        std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>>>
        : public DirectValueAccessor<
              std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>>>
    {
    public:
        AttributeAdapter(
            std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>>& value)
            : DirectValueAccessor<
                  std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>>>(
                  value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::"
            "OutputDescription>>>",
            0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
} // namespace ngraph
