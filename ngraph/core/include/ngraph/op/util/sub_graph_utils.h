// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/op/parameter.hpp>
#include "ngraph/op/op.hpp"


namespace ngraph
{
    namespace op
    {
        namespace util
        {
            namespace sub_graph_utils
            {
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

                    uint64_t m_input_index{ 0 };
                    uint64_t m_body_parameter_index{ 0 };
                };

                ///
                /// \brief      Describes a body input formed from slices of an input to
                ///             SubGraphOp.
                ///
                class NGRAPH_API SliceInputDescription : public InputDescription
                {
                public:
                    static constexpr type_info_t type_info{ "SliceInputDescription", 0 };
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
                    int64_t m_start{ 0 };
                    int64_t m_stride{ 0 };
                    int64_t m_part_size{ 0 };
                    int64_t m_end{ 0 };
                    int64_t m_axis{ 0 };
                };

                ///
                /// \brief      Describes a body input initialized from a SubGraphOp input on
                ///             the first iteration, and then a body output thereafter.
                ///
                class NGRAPH_API MergedInputDescription : public InputDescription
                {
                public:
                    static constexpr type_info_t type_info{ "MergedInputDescription", 0 };
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
                    uint64_t m_body_value_index{ 0 };
                };

                ///
                /// \brief      Describes a body input initialized from a SubGraphOp input on
                ///             the first iteration, and invariant thereafter.
                ///
                class NGRAPH_API InvariantInputDescription : public InputDescription
                {
                public:
                    static constexpr type_info_t type_info{ "InvariantInputDescription", 0 };
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

                    uint64_t m_body_value_index{ 0 };
                    uint64_t m_output_index{ 0 };
                };

                /// \brief Produces an output by concatenating an output from each iteration
                class NGRAPH_API ConcatOutputDescription : public OutputDescription
                {
                public:
                    static constexpr type_info_t type_info{ "ConcatOutputDescription", 0 };
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
                    int64_t m_start{ 0 };
                    int64_t m_stride{ 0 };
                    int64_t m_part_size{ 0 };
                    int64_t m_end{ 0 };
                    int64_t m_axis{ 0 };
                };

                /// \brief Produces an output from a specific iteration
                class NGRAPH_API BodyOutputDescription : public OutputDescription
                {
                public:
                    static constexpr type_info_t type_info{ "BodyOutputDescription", 0 };
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
                    int64_t m_iteration{ 0 };
                };
            }
        }
    }
}