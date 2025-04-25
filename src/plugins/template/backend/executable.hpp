// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace runtime {

class Executable {
public:
    Executable();
    virtual ~Executable();

    /// \param outputs vector of Tensor used as outputs
    /// \param inputs vector of Tensor used as inputs
    /// \param collect_performance Enable per operation performance statistic
    /// \returns true if iteration is successful, false otherwise
    virtual bool call(std::vector<ov::Tensor>& outputs,
                      const std::vector<ov::Tensor>& inputs,
                      bool collect_performance = false) = 0;

    /// \param outputs vector of Tensor used as outputs
    /// \param inputs vector of Tensor used as inputs
    /// \param context Evaluation context
    /// \param collect_performance Enable per operation performance statistic
    /// \returns true if iteration is successful, false otherwise
    virtual bool call(std::vector<ov::Tensor>& outputs,
                      const std::vector<ov::Tensor>& inputs,
                      const ov::EvaluationContext& context,
                      bool collect_performance = false) = 0;

    /// \brief Cancel and terminate the current execution
    virtual void cancel() = 0;

    /// \brief Executes a single iteration of a Function.
    /// \param outputs vector of Tensor used as outputs
    /// \param inputs vector of Tensor used as inputs
    /// \returns true if iteration is successful, false otherwise
    bool call_with_validate(std::vector<ov::Tensor>& outputs, const std::vector<ov::Tensor>& inputs);

    /// \brief Validates a Function.
    /// \param outputs vector of Tensor used as outputs
    /// \param inputs vector of Tensor used as inputs
    void validate(const std::vector<ov::Tensor>& outputs, const std::vector<ov::Tensor>& inputs);

    /// \brief Query the input Parameters
    /// \returns an ov::op::ParameterVector of all input parameters
    const ov::ParameterVector& get_parameters() const;

    /// \brief Query the output Results
    /// \returns an ov::ResultVector of all input parameters
    const ov::ResultVector& get_results() const;

    /// \brief Query the internal model
    /// \returns model which is used inside executable
    virtual std::shared_ptr<ov::Model> get_model() const = 0;

    /// \brief Create an input Tensor
    /// \param input_index The index position in the input Parameter vector. This would be the same
    /// order of Parameters passed into the inputs in the call() method.
    /// \returns A Tensor
    virtual ov::Tensor create_input_tensor(size_t input_index);

    /// \brief Create an output Tensor
    /// \param output_index The index position in the output Result vector. This would be the same
    /// order of Results passed into the outputs in the call() method.
    /// \returns A Tensor
    virtual ov::Tensor create_output_tensor(size_t output_index);

    /// \brief Create a vector of input Tensors
    /// \param input_index The index position in the input Parameter vector. This would be the same
    /// order of Parameters passed into the inputs in the call() method.
    /// \param pipeline_depth The number of stages in the input pipeline. For double-buffered input
    /// you would specify pipeline_depth=2
    /// \returns A vector of Tensors, one for each stage of the pipeline
    virtual std::vector<ov::Tensor> create_input_tensor(size_t input_index, size_t pipeline_depth);

    /// \brief Create a vector of output Tensors
    /// \param output_index The index position in the output Result vector. This would be the same
    ///                     order of Results passed into the outputs in the call() method.
    /// \param pipeline_depth The number of stages in the output pipeline. For double-buffered
    ///                       output you would specify pipeline_depth=2
    /// \returns A vector of Tensors, one for each stage of the pipeline
    virtual std::vector<ov::Tensor> create_output_tensor(size_t output_index, size_t pipeline_depth);

protected:
    /// \brief Called at the end of compile to the values to be returned by get_parameters
    ///        and get_results
    /// \param func The function with Results fully resolved.
    void set_parameters_and_results(const ov::Model& model);

    ov::ParameterVector m_parameters;
    ov::ResultVector m_results;
};

}  // namespace runtime
}  // namespace ov
