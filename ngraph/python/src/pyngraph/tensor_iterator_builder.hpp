//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cctype>
#include <map>
#include <memory>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ngraph/node.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/tensor_iterator.hpp"

namespace py = pybind11;

namespace util
{
    class TensorIteratorBuilder
    {
    public:
        ///
        /// \brief      Initialize TensorIterator node builder.
        ///
        /// \param[in]  arguments   The arguments passed to TensorIterator node.
        /// \param[in]  attributes  The TensorIterator's attributes. This
        ///                         py::dict contains all descriptors for
        ///                         plethora of TensorIterator available inputs
        ///                         and outputs.
        ///
        TensorIteratorBuilder(const ngraph::NodeVector& arguments, const py::dict& attributes);

        ///
        /// \brief      Configure instance of TensorIterator node with set-up parameters.
        ///
        /// \param      ti_node  The TensorIterator node instance to configure.
        ///
        /// \return     TensorIterator node.
        ///
        std::shared_ptr<ngraph::op::TensorIterator>
            configure(std::shared_ptr<ngraph::op::TensorIterator>&& ti_node);

    private:
        ///
        /// \brief      Helper to conduct attribute presence.
        ///
        /// \param[in]  attrs      The attributes
        /// \param[in]  attr_name  The attribute name
        /// \param[in]  desc_name  The description name
        ///
        inline void check_attribute(const py::dict& attrs,
                                    std::string attr_name,
                                    std::string desc_name) const;

        ///
        /// \brief      Retrieve the TI graph body.
        ///
        void get_graph_body();

        ///
        /// \brief      Sets the tensor iterator sliced inputs.
        ///
        /// \param      ti_node  The TI node we will set input to.
        ///
        void set_tensor_iterator_sliced_inputs(
            std::shared_ptr<ngraph::op::TensorIterator>& ti_node) const;

        ///
        /// \brief      Sets the tensor iterator merged inputs.
        ///
        /// \param      ti_node  The TI node we will set inputs to.
        ///
        void set_tensor_iterator_merged_inputs(
            std::shared_ptr<ngraph::op::TensorIterator>& ti_node) const;

        ///
        /// \brief      Sets the tensor iterator invariant inputs.
        ///
        /// \param      ti_node  The TI node we will set inputs to.
        ///
        void set_tensor_iterator_invariant_inputs(
            std::shared_ptr<ngraph::op::TensorIterator>& ti_node) const;

        ///
        /// \brief      Sets the tensor iterator outputs.
        ///
        /// \param      ti_node  The TI node we will set outputs to.
        ///
        void
            set_tensor_iterator_outputs(std::shared_ptr<ngraph::op::TensorIterator>& ti_node) const;

        ///
        /// \brief      Sets the tensor iterator body output.
        ///
        /// \param[in]  desc     The descriptor of the TI body output.
        /// \param      ti_node  The TI node we will set output to.
        ///
        void set_tensor_iterator_body_output(
            const py::dict& desc, std::shared_ptr<ngraph::op::TensorIterator>& ti_node) const;

        ///
        /// \brief      Sets the tensor iterator concatenated body output.
        ///
        /// \param[in]  desc     The descriptor of the TI body output.
        /// \param      ti_node  The TI node we will set output to.
        ///
        void set_tensor_iterator_concatenated_body_output(
            const py::dict& desc, std::shared_ptr<ngraph::op::TensorIterator>& ti_node) const;

        const ngraph::NodeVector& m_arguments;
        const py::dict& m_attributes;
        ngraph::OutputVector m_body_outputs;
        ngraph::ParameterVector m_body_parameters;
        std::shared_ptr<ngraph::op::TensorIterator::BodyLambda> m_body;
        py::list m_slice_input_desc;
        py::list m_merged_input_desc;
        py::list m_invariant_input_desc;
        std::map<int64_t, const py::dict> m_outputs;
    };
} // namespace util
