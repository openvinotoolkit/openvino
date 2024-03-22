// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "inverse.hpp"

#include "nodes/common/cpu_memcpy.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/inverse.hpp"
#include "utils/bfloat16.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

Inverse::Inverse(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        THROW_CPU_NODE_ERR(errorMessage);
    }

    auto inverse_op = as_type_ptr<op::v14::Inverse>(op);
    m_adjoint = inverse_op->get_adjoint();

    constant = ConstantType::StrictNoConst;

    m_const_input = is_type<op::v0::Constant>(op->get_input_node_ptr(INPUT_PORT));
}

bool Inverse::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v14::Inverse::get_type_info_static()) {
            errorMessage = "Only Inverse operation from the opset14 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void Inverse::getSupportedDescriptors() {
    if (getParentEdges().size() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges.");
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges.");
    }
}

void Inverse::initSupportedPrimitiveDescriptors() {
    m_input_precision = getOriginalInputPrecisionAtPort(INPUT_PORT);
    if (!one_of(m_input_precision, ov::element::f32, ov::element::f16, ov::element::bf16)) {
        m_input_precision = ov::element::f32;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, m_input_precision, m_const_input}},
                         {{LayoutType::ncsp, m_input_precision}},
                         ref_any);
}

void Inverse::prepareParams() {
    const auto& input_shape = getParentEdgeAt(INPUT_PORT)->getMemory().getStaticDims();

    if (input_shape.size() < 2) {
        THROW_CPU_NODE_ERR("has incompatible 'data' shape ",
                           PartialShape(input_shape),
                           ". Only tensors of rank at least 2 are allowed.");
    }

    m_side = input_shape.back();
    m_side_squared = m_side * m_side;
    m_batches_count = 1;

    for (size_t i = 0; i < input_shape.size() - 2; ++i) {
        m_batches_count = m_batches_count * input_shape[i];
    }
}

bool Inverse::created() const {
    return getType() == Type::Inverse;
}

void Inverse::execute(dnnl::stream strm) {
    OV_SWITCH(intel_cpu,
              InverseExecute,
              this,
              m_input_precision,
              OV_CASE(ov::element::bf16, bfloat16_t),
              OV_CASE(ov::element::f16, short),
              OV_CASE(ov::element::f32, float))
}

void Inverse::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

template <typename T>
void Inverse::inverse() {
    const auto* data = getSrcDataAtPortAs<const T>(INPUT_PORT);
    auto* output = getDstDataAtPortAs<T>(OUTPUT_PORT);

    std::vector<T> L(m_side_squared);
    std::vector<T> U(m_side_squared);
    std::vector<size_t> P(m_side);

#ifdef INVERSE_PARALLEL
    for (size_t b = 0; b < m_batches_count; ++b) {
        bool sign = true;

        lu_decomposition(data, L, U, P, sign, b);

        lu_solve(output, L, U, P, b);

        if (m_adjoint) {
            // Multiply by det(A) = det(U)
            to_adjoint(output, U, sign, b);
        }
    }
#else
    for (size_t b = 0; b < m_batches_count; ++b) {
        bool sign = true;

        internal::lu_decomposition(data, L, U, P, sign, b, m_side, m_side_squared);

        internal::lu_solve(output, L, U, P, b, m_side, m_side_squared);

        if (adjoint) {
            // Multiply by det(A) = det(U)
            internal::to_adjoint(output, U, sign, b, m_side, m_side_squared);
        }
    }
#endif
}

template <typename T>
void Inverse::lu_decomposition(const T* data,
                               std::vector<T>& L,
                               std::vector<T>& U,
                               std::vector<size_t>& P,
                               bool& sign,
                               size_t b) {
    // Make L identity, U a copy of data and P a range(0, side)
    const auto batch_idx = b * m_side_squared;

    std::fill(L.begin(), L.end(), T{0});
    cpu_parallel_memcpy(&U[0], &data[batch_idx], sizeof(T) * m_side_squared);

    parallel_for(m_side, [&](size_t i) {
        L[i * m_side + i] = T{1};
        P[i] = i;
    });

    for (size_t k = 0; k < m_side; ++k) {
        // Partial Pivoting
        auto pivot_row = k;
        auto pivot_idx = pivot_row * m_side;
        const auto k_idx = k * m_side;

        // Find maximum value pivot - non-parallel
        for (size_t i = (k + 1) * m_side, j = k + 1; i < m_side_squared; i += m_side, ++j) {
            if (std::abs(U[i + k]) > std::abs(U[pivot_idx + k])) {
                pivot_row = j;
                pivot_idx = pivot_row * m_side;
            }
        }

        if (pivot_row != k) {
            // Swap rows in L, U and P
            sign = !sign;
            std::swap(P[k], P[pivot_row]);
            parallel_for(m_side, [&](size_t i) {
                std::swap(L[k_idx + i], L[pivot_idx + i]);
                std::swap(U[k_idx + i], U[pivot_idx + i]);
            });
        }

        const auto remaining_columns = m_side - k;
        const auto remaining_rows = remaining_columns - 1;

        parallel_for(remaining_rows, [&](size_t i) {
            const auto i_idx = (i + k + 1) * m_side;
            L[i_idx + k] = U[i_idx + k] / U[k_idx + k];
        });

        parallel_for(remaining_rows * remaining_columns, [&](size_t i) {
            const auto i_idx = (i / remaining_columns + k + 1) * m_side;
            const auto j_idx = i % remaining_columns + k;
            U[i_idx + j_idx] = U[i_idx + j_idx] - L[i_idx + k] * U[k_idx + j_idx];
        });
    }
}

template <typename T>
void Inverse::lu_solve(T* output, std::vector<T>& L, std::vector<T>& U, std::vector<size_t>& P, size_t b) {
    parallel_for(m_side, [&](size_t column) {
        std::vector<T> X(m_side, T{0});
        std::vector<T> Y(m_side, T{0});

        // Forward substitution: Ly = Pb
        for (size_t i = 0; i < m_side; ++i) {
            if (P[i] == column) {
                Y[i] = T{1};
            }
            const auto i_idx = i * m_side;
            for (size_t j = 0; j < i; ++j) {
                Y[i] = Y[i] - L[i_idx + j] * Y[j];
            }
        }

        // Backward substitution: Ux = y
        for (size_t i = 0; i < m_side; ++i) {
            size_t i_adj = m_side - i - 1;
            size_t i_idx = i_adj * m_side;
            X[i_adj] = Y[i_adj];
            for (size_t j = i_adj + 1; j < m_side; ++j) {
                X[i_adj] = X[i_adj] - U[i_idx + j] * X[j];
            }
            X[i_adj] = X[i_adj] / U[i_idx + i_adj];
        }

        // Substitute back to get result
        const auto batch_column_idx = b * m_side_squared + column;
        for (size_t row = 0; row < m_side; ++row) {
            output[batch_column_idx + row * m_side] = X[row];
        }
    });
}

template <typename T>
void Inverse::to_adjoint(T* output, std::vector<T>& U, bool sign, size_t b) {
    T determinant = sign ? T{1} : T{-1};

    for (size_t i = 0; i < m_side; ++i) {
        determinant = determinant * U[i * m_side + i];
    }

    const auto batch_idx = b * m_side_squared;
    parallel_for(m_side_squared, [&](size_t i) {
        output[batch_idx + i] = output[batch_idx + i] * determinant;
    });
}

template <typename T>
void Inverse::lu_decomposition_(const T* input,
                                std::vector<T>& L,
                                std::vector<T>& U,
                                std::vector<size_t>& P,
                                bool& sign,
                                size_t b,
                                size_t n,
                                size_t n_squared) {
    // Make L identity, U a copy of input and P a range(0, n)
    const auto batch_idx = b * n_squared;

    std::fill(L.begin(), L.end(), T{0});
    memcpy(&U[0], &input[batch_idx], sizeof(T) * n_squared);

    for (size_t i = 0; i < n; ++i) {
        L[i * n + i] = T{1};
        P[i] = i;
    }

    for (size_t k = 0; k < n; ++k) {
        // Partial Pivoting
        auto pivot_row = k;
        auto pivot_idx = pivot_row * n;
        const auto k_idx = k * n;

        for (auto i = (k + 1) * n, j = k + 1; i < n_squared; i += n, ++j) {
            if (std::abs(U[i + k]) > std::abs(U[pivot_idx + k])) {
                pivot_row = j;
                pivot_idx = pivot_row * n;
            }
        }

        if (pivot_row != k) {
            // Swap rows in L, U (A) and P
            sign = !sign;
            std::swap(P[k], P[pivot_row]);
            std::swap_ranges(&U[k_idx], &U[k_idx + n], &U[pivot_idx]);
            std::swap_ranges(&L[k_idx], &L[k_idx + n], &L[pivot_idx]);
        }

        const auto remaining_columns = n - k;
        const auto remaining_rows = remaining_columns - 1;

        for (size_t i = 0; i < remaining_rows; ++i) {
            const auto i_idx = (i + k + 1) * n;
            L[i_idx + k] = U[i_idx + k] / U[k_idx + k];
        }

        for (size_t i = 0; i < remaining_rows * remaining_columns; ++i) {
            const auto i_idx = (i / remaining_columns + k + 1) * n;
            const auto j_idx = i % remaining_columns + k;
            U[i_idx + j_idx] = U[i_idx + j_idx] - L[i_idx + k] * U[k_idx + j_idx];
        }
    }
}

template <typename T>
void Inverse::lu_solve_(T* output,
                        std::vector<T>& L,
                        std::vector<T>& U,
                        std::vector<size_t>& P,
                        size_t b,
                        size_t n,
                        size_t n_squared) {
    std::vector<T> X(n);
    std::vector<T> Y(n);

    for (size_t column = 0; column < n; ++column) {
        std::fill(X.begin(), X.end(), T{0});
        std::fill(Y.begin(), Y.end(), T{0});

        // Forward substitution: Ly = Pb
        for (size_t i = 0; i < n; ++i) {
            if (P[i] == column) {
                Y[i] = T{1};
            }
            const auto i_idx = i * n;
            for (size_t j = 0; j < i; ++j) {
                Y[i] -= L[i_idx + j] * Y[j];
            }
        }

        // Backward substitution: Ux = y
        for (size_t i = 0; i < n; ++i) {
            const auto i_adj = n - i - 1;
            const auto i_idx = i_adj * n;
            X[i_adj] = Y[i_adj];
            for (size_t j = i_adj + 1; j < n; ++j) {
                X[i_adj] = X[i_adj] - U[i_idx + j] * X[j];
            }
            X[i_adj] = X[i_adj] / U[i_idx + i_adj];
        }

        const auto batch_column_idx = b * n_squared + column;
        for (size_t row = 0; row < n; ++row) {
            output[batch_column_idx + row * n] = X[row];
        }
    }
}

template <typename T>
void Inverse::to_adjoint_(T* output, std::vector<T>& U, bool sign, size_t b, size_t n, size_t n_squared) {
    T determinant = sign ? T{1} : T{-1};

    for (size_t i = 0; i < n; ++i) {
        determinant *= U[i * n + i];
    }

    const auto batch_idx = b * n_squared;
    for (size_t i = 0; i < n_squared; ++i) {
        output[batch_idx + i] *= determinant;
    }
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
