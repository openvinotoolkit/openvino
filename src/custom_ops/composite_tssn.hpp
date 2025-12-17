#pragma once

#include "openvino/op/op.hpp"
#include "btl/btl_function_library.hpp"
#include <mutex>

namespace ov {
namespace op {
namespace v0 {

class CompositeTSSN : public Op {
public:
    OPENVINO_OP("CompositeTSSN", "extension");

    CompositeTSSN() = default;

    /// \brief Constructs a CompositeTSSN operation.
    ///
    /// \param x Input tensor [Batch, InputDim]
    /// \param indices Sparse indices [2, N_Synapses] (Rows=InputIdx, Cols=OutputIdx)
    /// \param weights Ternary weights [N_Synapses] (int8 or float)
    /// \param sensitivity Sensitivity weights [N_Synapses] (float)
    /// \param output_dim Output dimension size
    /// \param func_ids List of BTL function IDs to apply sequentially
    CompositeTSSN(const Output<Node>& x,
                  const Output<Node>& indices,
                  const Output<Node>& weights,
                  const Output<Node>& sensitivity,
                  int64_t output_dim,
                  const std::vector<int64_t>& func_ids);

    CompositeTSSN(const Output<Node>& x,
                  const Output<Node>& indices,
                  const Output<Node>& weights,
                  const Output<Node>& sensitivity,
                  const Output<Node>& synapse_counts,
                  const Output<Node>& synapse_starts,
                  int64_t output_dim,
                  const std::vector<int64_t>& func_ids);

    CompositeTSSN(const Output<Node>& x,
                  const Output<Node>& indices,
                  const Output<Node>& weights,
                  const Output<Node>& sensitivity,
                  const Output<Node>& synapse_counts,
                  const Output<Node>& synapse_starts,
                  const Output<Node>& function_ids,
                  int64_t output_dim,
                  const std::vector<int64_t>& func_ids);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    bool has_evaluate() const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;

private:
    int64_t m_output_dim;
    std::vector<int64_t> m_func_ids;
    
    // Optimization flags (mutable for lazy initialization in evaluate)
    mutable bool m_is_dense_initialized = false;
    mutable bool m_is_dense_input = false;

    // Kernel function pointer type
    using KernelFunc = void (*)(const float* x, const int32_t* indices, const float* weights, 
                                const float* sensitivity, float* out, size_t n_synapses, 
                                size_t input_dim, size_t output_dim, const std::vector<int64_t>& func_ids);
    
    // Selected kernel
    mutable KernelFunc m_selected_kernel = nullptr;
    mutable std::mutex m_kernel_mutex;

    // Helper to select kernel
    void select_kernel() const;
};

} // namespace v0
} // namespace op
} // namespace ov
