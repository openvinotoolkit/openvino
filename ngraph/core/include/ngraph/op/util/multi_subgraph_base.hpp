// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/function.hpp>
#include <ngraph/op/parameter.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
namespace util {
/// \brief Abstract base class for sub-graph based ops, i.e ops that have some
/// sub-graphs
///
class NGRAPH_API MultiSubGraphOp : public Op {
public:
    NGRAPH_RTTI_DECLARATION;
    /// \brief Abstract class describes a connection between a MultiSubGraphOp input and
    /// the body.
    class InputDescription {
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
    class OutputDescription {
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

    ///
    /// \brief      Describes a body input formed from slices of an input to
    ///             MultiSubGraphOp.
    ///
    class NGRAPH_API SliceInputDescription : public InputDescription {
    public:
        NGRAPH_RTTI_DECLARATION;
        ///
        /// \brief      Constructs a new instance.
        ///
        /// \param      input_index           Position of the MultiSubGraphOp input
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
    /// \brief      Describes a body input initialized from a MultiSubGraphOp input
    ///             on the first iteration, and then a body output thereafter.
    ///
    class NGRAPH_API MergedInputDescription : public InputDescription {
    public:
        NGRAPH_RTTI_DECLARATION;
        ///
        /// \brief      Constructs a new instance.
        ///
        /// \param      input_index           Position of the MultiSubGraphOp input
        ///                                   supplying a value to body_parameter for
        ///                                   the initial iteration.
        /// \param      body_parameter_index  Body parameter position to receive input.
        /// \param      body_value_index      Body value to supply body_parameter for
        /// successive
        ///                                   iterations.
        ///
        MergedInputDescription(uint64_t input_index, uint64_t body_parameter_index, uint64_t body_value_index);
        MergedInputDescription() = default;
        std::shared_ptr<InputDescription> copy() const override;
        uint64_t m_body_value_index{0};
    };

    /// \brief Produces an output by concatenating an output from each iteration
    class NGRAPH_API ConcatOutputDescription : public OutputDescription {
    public:
        NGRAPH_RTTI_DECLARATION;
        ///
        /// \brief      Constructs a new instance.
        ///
        /// \param      body_value_index  A body value that produces the output
        /// \param      output_index      The MultiSubGraphOp output index
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

    /// \brief Produces an input
    class NGRAPH_API InvariantInputDescription : public InputDescription {
    public:
        NGRAPH_RTTI_DECLARATION;
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

    /// \brief Produces an output from a specific iteration
    class NGRAPH_API BodyOutputDescription : public MultiSubGraphOp::OutputDescription {
    public:
        NGRAPH_RTTI_DECLARATION;
        ///
        /// \brief      Constructs a new instance.
        ///
        /// \param      body_value_index  A body value that produces the output
        /// \param      output_index      The SubGraphOp output index
        /// \param      iteration         which iteration (typically -1, final) will
        ///                               supply the value
        ///
        BodyOutputDescription(uint64_t body_value_index, uint64_t output_index, int64_t iteration = -1);
        BodyOutputDescription() = default;
        std::shared_ptr<MultiSubGraphOp::OutputDescription> copy() const override;
        int64_t m_iteration{0};
    };
    using MultiSubgraphInputDescriptionPtr = std::shared_ptr<MultiSubGraphOp::InputDescription>;
    using MultiSubgraphOutputDescriptionPtr = std::shared_ptr<MultiSubGraphOp::OutputDescription>;
    using MultiSubgraphInputDescriptionVector = std::vector<MultiSubgraphInputDescriptionPtr>;
    using MultiSubgraphOutputDescriptionVector = std::vector<MultiSubgraphOutputDescriptionPtr>;

    /// \brief     Gets internal sub-graph by index in MultiSubGraphOp
    ///
    /// \param     index sub-graph's index in op
    /// \return pointer to ngraph::Function with sub-graph
    virtual const std::shared_ptr<Function>& get_function(int index) const {
        return m_bodies[index];
    };
    /// \brief     Adds sub-graph to MultiSubGraphOp
    ///
    /// \param index   index of new sub-graph
    /// \param func    func new sub_graph as ngraph::Function
    virtual void set_function(int index, const std::shared_ptr<Function>& func) {
        m_bodies[index] = func;
    }
    /// \brief     Gets vector with connections beewtwen operation inputs
    /// and internal sub-graph parameters
    ///
    /// \param index   index of internal sub-graph
    /// \return vector of input descriptions
    const MultiSubgraphInputDescriptionVector& get_input_descriptions(int index) const {
        return m_input_descriptions[index];
    }
    /// \brief     Gets vector with connections beewtwen operation inputs
    /// and internal sub-graph parameters
    ///
    /// \param index   index of internal sub-graph
    /// \return vector of input descriptions
    MultiSubgraphInputDescriptionVector& get_input_descriptions(int index) {
        return m_input_descriptions[index];
    }
    /// \brief     Gets vector with connections beewtwen operation outputs
    /// and internal sub-graph results
    ///
    /// \param index   index of internal sub-graph
    /// \return vector of output descriptions
    const MultiSubgraphOutputDescriptionVector& get_output_descriptions(int index) const {
        return m_output_descriptions[index];
    }
    /// \brief     Gets vector with connections beewtwen operation outputs
    /// and internal sub-graph results
    ///
    /// \param index   index of internal sub-graph
    /// \return vector of output descriptions
    MultiSubgraphOutputDescriptionVector& get_output_descriptions(int index) {
        return m_output_descriptions[index];
    }
    /// \brief     Sets vector with connections beewtwen operation inputs
    /// and internal sub-graph parameters
    ///
    /// \param index   index of internal sub-graph
    /// \param inputs  vector of input descriptions
    void set_input_descriptions(int index, const MultiSubgraphInputDescriptionVector& inputs) {
        m_input_descriptions[index] = inputs;
    }

    /// \brief     Sets vector with connections beewtwen operation outputs
    /// and internal sub-graph results
    ///
    /// \param index   index of internal sub-graph
    /// \param outputs vector of input descriptions
    void set_output_descriptions(int index, const MultiSubgraphOutputDescriptionVector& outputs) {
        m_output_descriptions[index] = outputs;
    }

    ///
    /// \brief     Set input decriptions for MultiSubGraphOp input.
    ///
    /// \param      value              The value supplied as an input to the block.
    /// \param      bodies_parameters  vector of bodies parameters.
    virtual void set_invariant_inputs(const Output<Node>& value, const ParameterVector& bodies_parameters);
    ///
    /// \brief     Set output decriptions for MultiSubGraphOp output.
    ///
    /// \param      bodies_results  vector of bodies results for one output.
    /// \return     value           Output node for bodies_results.
    virtual Output<Node> set_body_outputs(const ResultVector& bodies_results);

    MultiSubGraphOp(const MultiSubGraphOp&) = delete;
    MultiSubGraphOp(MultiSubGraphOp&&) = default;

    MultiSubGraphOp& operator=(const MultiSubGraphOp&) = delete;
    MultiSubGraphOp& operator=(MultiSubGraphOp&&) = default;

protected:
    // Find an input corresponding to value, adding one if necessary.
    Input<Node> input_for_value(const Output<Node>& value);

    MultiSubGraphOp(size_t number_of_bodies);
    MultiSubGraphOp() = default;
    MultiSubGraphOp(const OutputVector& args, size_t number_of_bodies);
    explicit MultiSubGraphOp(const OutputVector& args);

    std::vector<std::shared_ptr<Function>> m_bodies;
    std::vector<MultiSubgraphInputDescriptionVector> m_input_descriptions;
    std::vector<MultiSubgraphOutputDescriptionVector> m_output_descriptions;
};
using MultiSubgraphInputDescriptionPtr = util::MultiSubGraphOp::MultiSubgraphInputDescriptionPtr;
using MultiSubgraphOutputDescriptionPtr = util::MultiSubGraphOp::MultiSubgraphOutputDescriptionPtr;
using MultiSubgraphInputDescriptionVector = util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector;
using MultiSubgraphOutputDescriptionVector = util::MultiSubGraphOp::MultiSubgraphOutputDescriptionVector;

}  // namespace util
}  // namespace op

template <>
class NGRAPH_API AttributeAdapter<std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::InputDescription>>>
    : public DirectValueAccessor<std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::InputDescription>>> {
public:
    AttributeAdapter(std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::InputDescription>>& value)
        : DirectValueAccessor<std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::InputDescription>>>(
              value) {}

    NGRAPH_RTTI_DECLARATION;
};

template <>
class NGRAPH_API AttributeAdapter<std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::OutputDescription>>>
    : public DirectValueAccessor<std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::OutputDescription>>> {
public:
    AttributeAdapter(std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::OutputDescription>>& value)
        : DirectValueAccessor<std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::OutputDescription>>>(
              value) {}

    NGRAPH_RTTI_DECLARATION;
};
}  // namespace ngraph
