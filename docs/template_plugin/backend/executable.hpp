// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "performance_counter.hpp"

namespace ngraph {
namespace runtime {
class Executable;
}
}  // namespace ngraph

class ngraph::runtime::Executable {
public:
    Executable();
    virtual ~Executable();

    /// \param outputs vector of runtime::Tensor used as outputs
    /// \param inputs vector of runtime::Tensor used as inputs
    /// \returns true if iteration is successful, false otherwise
    virtual bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                      const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) = 0;

    /// \brief Executes a single iteration of a Function.
    /// \param outputs vector of runtime::Tensor used as outputs
    /// \param inputs vector of runtime::Tensor used as inputs
    /// \returns true if iteration is successful, false otherwise
    bool call_with_validate(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                            const std::vector<std::shared_ptr<runtime::Tensor>>& inputs);

    /// \brief Collect performance information gathered on a Function.
    /// \returns Vector of PerformanceCounter information.
    virtual std::vector<PerformanceCounter> get_performance_data() const;

    /// \brief Validates a Function.
    /// \param outputs vector of runtime::Tensor used as outputs
    /// \param inputs vector of runtime::Tensor used as inputs
    void validate(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                  const std::vector<std::shared_ptr<runtime::Tensor>>& inputs);

    /// \brief Query the input Parameters
    /// \returns an ngraph::op::ParameterVector of all input parameters
    const ngraph::ParameterVector& get_parameters() const;

    /// \brief Query the output Results
    /// \returns an ngraph::ResultVector of all input parameters
    const ngraph::ResultVector& get_results() const;

    /// \brief Get the preferred pipeline_depth for this executable
    /// \returns  preferred pipeline_depth
    virtual size_t get_preferred_pipeline_depth() const;

    /// \brief Save this compiled Executable to an output stream.
    ///    Saved stream may be read with Backend::load
    virtual void save(std::ostream& output_stream);

    /// \brief Create an input Tensor
    /// \param input_index The index position in the input Parameter vector. This would be the same
    /// order of Parameters passed into the inputs in the call() method.
    /// \returns A Tensor
    virtual std::shared_ptr<runtime::Tensor> create_input_tensor(size_t input_index);

    /// \brief Create an input Tensor
    /// \param input_index The index position in the input Parameter vector. This would be the same
    /// order of Parameters passed into the inputs in the call() method.
    /// \param memory_pointer A pointer to a buffer used for this tensor. The size of the buffer
    ///     must be sufficient to contain the tensor. The lifetime of the buffer is the
    ///     responsibility of the caller and must outlive the created Tensor.
    /// \returns A Tensor
    virtual std::shared_ptr<runtime::Tensor> create_input_tensor(size_t input_index, void* memory_pointer);

    /// \brief Create an output Tensor
    /// \param output_index The index position in the output Result vector. This would be the same
    /// order of Results passed into the outputs in the call() method.
    /// \returns A Tensor
    virtual std::shared_ptr<runtime::Tensor> create_output_tensor(size_t output_index);

    /// \brief Create an output Tensor
    /// \param output_index The index position in the output Result vector. This would be the same
    /// order of Results passed into the outputs in the call() method.
    /// \param memory_pointer A pointer to a buffer used for this tensor. The size of the buffer
    ///     must be sufficient to contain the tensor. The lifetime of the buffer is the
    ///     responsibility of the caller and must outlive the created Tensor.
    /// \returns A Tensor
    virtual std::shared_ptr<runtime::Tensor> create_output_tensor(size_t output_index, void* memory_pointer);

    /// \brief Create a vector of input Tensors
    /// \param input_index The index position in the input Parameter vector. This would be the same
    /// order of Parameters passed into the inputs in the call() method.
    /// \param pipeline_depth The number of stages in the input pipeline. For double-buffered input
    /// you would specify pipeline_depth=2
    /// \returns A vector of Tensors, one for each stage of the pipeline
    virtual std::vector<std::shared_ptr<runtime::Tensor>> create_input_tensor(size_t input_index,
                                                                              size_t pipeline_depth);

    /// \brief Create a vector of input Tensors
    /// \param input_index The index position in the input Parameter vector. This would be the same
    /// order of Parameters passed into the inputs in the call() method.
    /// \param pipeline_depth The number of stages in the input pipeline. For double-buffered input
    /// you would specify pipeline_depth=2
    /// \param memory_pointers A vector of pointers to buffers used for this tensors. The size of
    ///     the buffer must be sufficient to contain the tensor. The lifetime of the buffers is the
    ///     responsibility of the caller and must outlive the created Tensor.
    /// \returns A vector of Tensors, one for each stage of the pipeline
    virtual std::vector<std::shared_ptr<runtime::Tensor>> create_input_tensor(size_t input_index,
                                                                              size_t pipeline_depth,
                                                                              std::vector<void*> memory_pointers);

    /// \brief Create a vector of output Tensors
    /// \param output_index The index position in the output Result vector. This would be the same
    ///                     order of Results passed into the outputs in the call() method.
    /// \param pipeline_depth The number of stages in the output pipeline. For double-buffered
    ///                       output you would specify pipeline_depth=2
    /// \returns A vector of Tensors, one for each stage of the pipeline
    virtual std::vector<std::shared_ptr<runtime::Tensor>> create_output_tensor(size_t output_index,
                                                                               size_t pipeline_depth);

    /// \brief Create a vector of output Tensors
    /// \param output_index The index position in the output Result vector. This would be the same
    ///                     order of Results passed into the outputs in the call() method.
    /// \param pipeline_depth The number of stages in the output pipeline. For double-buffered
    ///                       output you would specify pipeline_depth=2
    /// \param memory_pointers A vector of pointers to buffers used for this tensors. The size of
    ///     the buffer must be sufficient to contain the tensor. The lifetime of the buffers is the
    ///     responsibility of the caller and must outlive the created Tensor.
    /// \returns A vector of Tensors, one for each stage of the pipeline
    virtual std::vector<std::shared_ptr<runtime::Tensor>> create_output_tensor(size_t output_index,
                                                                               size_t pipeline_depth,
                                                                               std::vector<void*> memory_pointers);

protected:
    /// \brief Called at the end of compile to the values to be returned by get_parameters
    ///        and get_results
    /// \param func The function with Results fully resolved.
    void set_parameters_and_results(const Function& func);

    ngraph::ParameterVector m_parameters;
    ngraph::ResultVector m_results;
};
