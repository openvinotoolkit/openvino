// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>    // std::generate
#include <cmath>        // std::floor, std::min
#include <cstddef>      // std::size_t
#include <cstdint>      // std::int64_t
#include <iterator>     // std::begin, std::end
#include <memory>       // std::shared_ptr, std::make_shared
#include <type_traits>  // std::enable_if
#include <vector>

#include "core/node.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace common {
const ov::element::Type& get_ov_element_type(std::int64_t onnx_type);

/// \brief Function does a default checks for a node. Raise an exception if checks are failed
/// \param[in]  node    Node to check
/// \param[in]  min_inputs_size  Minimal amount of inputs expected
void default_op_checks(const Node& node, size_t min_inputs_size);

/// \brief      Return a monotonic sequence.
///
/// \note       Limitations: this function may not work for very large integer values
///             (near numerical limits).
///
/// \param[in]  start_value  The start value of the sequence.
/// \param[in]  end_value    The end value of the sequence.
/// \param[in]  step         The step value for the sequence.
///
/// \tparam     T            The data value type.
///
/// \return     The vector with monotonic sequence
template <typename T>
std::vector<T> get_monotonic_range(T end_value, T start_value = T{0}, T step = T{1}) {
    auto value_count = static_cast<std::size_t>(std::floor((end_value - start_value) / step));

    std::vector<T> range(value_count);

    // Calculate initial value (one step below starting value)
    size_t n = start_value - step;
    // Generate a vector of values by adding step to previous value
    std::generate(std::begin(range), std::end(range), [&n, &step]() -> T {
        return n += step;
    });

    return range;
}

/// \brief      Return a monotonic sequence which end is determined by value rank.
///
/// \param[in]  value        The value node which rank determines end of the sequence.
/// \param[in]  start_value  The start value of the sequence.
/// \param[in]  step         The step value for the sequence.
///
/// \return     The node which represents monotonic sequence.
std::shared_ptr<ov::Node> get_monotonic_range_along_node_rank(const ov::Output<ov::Node>& value,
                                                              int64_t start_value = 0,
                                                              int64_t step = 1);

/// \brief Creates a shifted square identity matrix.
/// \note Shifting in the context of this operator means that
///       the matrix can be created with elements equal to 1 not only in the main
///       diagonal. Shifting adds an offset and moves the diagonal up or down
///
/// \param[in] output_shape Shape of the resulting matrix.
/// \param[in] output_type Element type of the resulting matrix.
/// \param[in] shift Shifting of diagonal.
///
/// \return A Constant node representing shifted identity matrix.
template <typename T = double>
std::shared_ptr<ov::op::v0::Constant> shifted_square_identity(const ov::Shape output_shape,
                                                              const ov::element::Type& output_type,
                                                              const std::int64_t shift) {
    std::vector<T> identity_matrix(shape_size(output_shape), T{0});
    std::int64_t rows = output_shape[0];
    std::int64_t cols = output_shape[1];
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::int64_t diagonal_element_idx = (row * cols) + row + shift;
        if (row + shift < 0) {
            continue;
        } else if (row + shift >= cols) {
            break;
        }
        identity_matrix.at(diagonal_element_idx) = T{1};
    }

    return std::make_shared<ov::op::v0::Constant>(output_type, output_shape, identity_matrix);
}

/// \brief Creates a square identity matrix.
///
/// \param[in] n Order of the resulting matrix.
///
/// \return A Constant node representing identity matrix with shape (n, n).
template <typename T = double>
std::shared_ptr<ov::op::v0::Constant> square_identity(const size_t n, const ov::element::Type& type) {
    return shifted_square_identity(ov::Shape{n, n}, type, 0);
}

/// \brief Performs validation of an input that is expected to be a scalar.
/// \note  This function throws an exception if any of the validation steps fails.
///
/// \param[in] input_name A human-readable name of an input (used for logging)
/// \param[in] input An input node to be validated
/// \param[in] allowed_types An optional set of allowed element types for this input
void validate_scalar_input(const char* input_name,
                           const std::shared_ptr<ov::Node> input,
                           const std::set<ov::element::Type> allowed_types = {});

/// \brief Temporary replacement for C++14 std::make_unique.
/// \note details: https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique
///
/// \param args List of arguments with which an instance of T will be constructed.
///
/// \return     std::unique_ptr of an instance of type T
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/// \brief Function that handles following ONNX operators: Add, Div, Mul, Sub
///        from opset 6.
///
/// \param node ONNX node
///
/// \return     ov::OutputVector with binary op

template <typename T>
ov::OutputVector handle_opset6_binary_op(const ov::frontend::onnx::Node& node);

/// \brief  Creates a "dummy" constant to be used in place of an invalid initializer
///         encountered in the original model.
/// \return A scalar constant containing a single value of zero
///         marked as "failsafe" in the runtime info object
std::shared_ptr<ov::op::v0::Constant> make_failsafe_constant(const ov::element::Type& dtype);

/// \brief Checks the node's runtime info object and returns true if this node represents
///        a dummy failsafe node created instead of an incorrect node found in the original model
bool is_failsafe_node(const std::shared_ptr<ov::Node>& node);

/// \brief Marks an output of a node as "optimized out" meaning that during the import of an ONNX operation
///        no OV nodes have been created and the ONNX operator returns its inputs as its outputs.
///        This information is later used to add extra names to the tensors associated with such outputs.
void mark_as_optimized_out(ov::Output<ov::Node>& node_output);

/// \brief Checks if a given output was marked as optimized out byt the function above.
bool is_optimized_out(const ov::Output<ov::Node>& node_output);

/// \brief Collect unsupported operators after convert_partially and all exceptions from translation process.
/// \param partially_converted ov::Model which has been partially converted
/// \param telemetry Pointer on a TelemetryExtension if telemetry is enabled
/// \param output_stream Pointer on a stream for printint error messages
/// \param unsupported_operations Set for collecting list of unsupported operations, should be nullptr for
///                               first call (will be created internally)
/// \param failures Set for collecting list of failed conversions, should be nullptr for
///                 first call (will be created internally)
/// \return Returns true in case any issues has been found
bool collect_translation_exceptions(const std::shared_ptr<ov::Model>& partially_converted,
                                    const std::shared_ptr<ov::frontend::TelemetryExtension>& telemetry = nullptr,
                                    std::ostream* output_stream = nullptr,
                                    std::shared_ptr<std::set<std::string>> unsupported_operations = nullptr,
                                    std::shared_ptr<std::set<std::string>> failures = nullptr);

// \brief OpenVINO supports only uint64 seeds with a meaningful 0 value (seed will be auto-generated).
// Because we use a seed as a just meaningful identifier we may
// just interpret its value as a 32-bit value (float zero value is same with
// uint32 zero value).
// Float -0 value will be interpreted as a valid uint32 value.
// \param seed Float value for conversion
// \return Returns a converted uint32_t value
inline uint32_t convert_float_seed(const float seed) {
    const void* seed_ptr = &seed;  // To prevent strict-aliasing error
    return *static_cast<const uint32_t*>(seed_ptr);
}

/// \brief Tries normalize axis against the rank.
///
/// Throws if rank is dynamic or, axis outside rank range [-rank, rank).
///
/// \param description  Additional description added to error message.
/// \param axis         Axis value to be normalized.
/// \param rank         Rank used for axis normalization.
/// \return             Normalized axis value.
int64_t normalize_axis(const std::string& description, const int64_t axis, const Rank& rank);
}  // namespace  common
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
