// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/einsum.hpp"

#include <algorithm>

#include "openvino/op/einsum.hpp"
#include "openvino/reference/broadcast.hpp"
#include "openvino/reference/matmul.hpp"
#include "openvino/reference/multiply.hpp"
#include "openvino/reference/reduce_sum.hpp"
#include "openvino/reference/reshape.hpp"
#include "openvino/reference/transpose.hpp"
#include "openvino/reference/utils/span.hpp"

namespace ov {
namespace reference {
namespace {
/// \brief      Compute einsum_path for a given Einsum node meaning that the
/// (pseudo-)optimal order of operands contraction in terms of performance and
/// memory consumption
///
std::vector<std::pair<size_t, size_t>> compute_einsum_path(size_t num_inputs) {
    // TODO: implement algorithm for finding (pseudo-)optimal einsum_path
    std::vector<std::pair<size_t, size_t>> einsum_path;
    OPENVINO_ASSERT(num_inputs > 0);
    for (size_t input_ind = num_inputs - 1; input_ind > 0; --input_ind) {
        einsum_path.push_back(std::make_pair(0, input_ind));
    }
    return einsum_path;
}

/// \brief    Checks if input vector represents a range [0; n]
///
bool is_range_0_to_n(const std::vector<int64_t>& labels_inds) {
    int64_t check_index = 0;
    for (auto index : labels_inds) {
        if (check_index != index) {
            return false;
        }
        ++check_index;
    }
    return true;
}

/// \brief      Check if the dimension with a given label is reduced. The dimension
/// is reduced if the corresponding label is met in neither the output subscript nor
/// the input subscripts excluding ones specified by a vector excluded_indices
///
bool is_dimension_reduced(const std::vector<std::string>& input_subscripts,
                          const std::string& output_subscript,
                          const std::string& label_to_check,
                          const std::vector<size_t>& excluded_indices) {
    for (size_t input_ind = 0; input_ind < input_subscripts.size(); ++input_ind) {
        const auto& input_subscript = input_subscripts[input_ind];
        // the subscript is checked only if its index is not in excluded indices
        // list
        bool check_subscript =
            (std::find(excluded_indices.begin(), excluded_indices.end(), input_ind) == excluded_indices.end());
        if (check_subscript && input_subscript.find(label_to_check) != std::string::npos) {
            return false;
        }
    }
    return output_subscript.find(label_to_check) == std::string::npos;
}

/// \brief      Generate an input subscript that provides to group dimensions into
/// the common, separate and reduced dimensions after transpose
///
std::string generate_grouping_subscript(const std::string& input_subscript,
                                        const std::vector<int64_t>& common_labels_inds,
                                        const std::vector<int64_t>& separate_labels_inds,
                                        const std::vector<int64_t>& reduced_labels_inds,
                                        bool& is_separate_first) {
    // transpose is not needed if common labels, reduced labels
    // and separate labels indices go concurrently
    std::vector<int64_t> labels_inds = common_labels_inds;
    labels_inds.insert(labels_inds.end(), reduced_labels_inds.begin(), reduced_labels_inds.end());
    labels_inds.insert(labels_inds.end(), separate_labels_inds.begin(), separate_labels_inds.end());
    if (is_range_0_to_n(labels_inds)) {
        is_separate_first = false;
        return input_subscript;
    }

    // transpose is not needed if common labels, separate labels
    // and reduced labels indices go concurrently
    labels_inds = common_labels_inds;
    labels_inds.insert(labels_inds.end(), separate_labels_inds.begin(), separate_labels_inds.end());
    labels_inds.insert(labels_inds.end(), reduced_labels_inds.begin(), reduced_labels_inds.end());
    if (is_range_0_to_n(labels_inds)) {
        is_separate_first = true;
        return input_subscript;
    }

    auto labels = ov::op::v7::Einsum::extract_labels(input_subscript);
    std::string required_subscript = "";
    for (auto index : labels_inds) {
        required_subscript += labels[index];
    }
    is_separate_first = true;
    return required_subscript;
}

std::unordered_map<std::string, std::vector<size_t>> compute_label_dim_map(const Rank& input_rank,
                                                                           const std::string& input_subscript) {
    constexpr char ellipsis[] = "...";
    auto labels = ov::op::v7::Einsum::extract_labels(input_subscript);
    size_t input_rank_length = labels.size();
    OPENVINO_ASSERT(input_rank.is_static() || (std::find(labels.begin(), labels.end(), ellipsis) == labels.end()),
                    "Input rank cannot be dynamic in case of ellipsis in input subscript");
    if (input_rank.is_static()) {
        input_rank_length = input_rank.get_length();
    }
    std::unordered_map<std::string, std::vector<size_t>> resulted_map;
    OPENVINO_ASSERT(input_rank_length >= labels.size());
    size_t num_broadcasted_dims = input_rank_length - labels.size() + 1;

    size_t current_dim = 0;
    for (const auto& label : labels) {
        if (label == ellipsis) {
            std::vector<size_t> label_dims;
            for (size_t ind = 0; ind < num_broadcasted_dims; ++ind) {
                label_dims.push_back(static_cast<size_t>(current_dim + ind));
            }
            resulted_map[label] = std::move(label_dims);
            current_dim += num_broadcasted_dims;
        } else if (resulted_map.find(label) != resulted_map.end()) {
            resulted_map[label].push_back(static_cast<size_t>(current_dim));
            ++current_dim;
        } else {
            std::vector<size_t> label_dims;
            label_dims.push_back(static_cast<size_t>(current_dim));
            resulted_map[label] = std::move(label_dims);
            ++current_dim;
        }
    }

    return resulted_map;
}

/// \brief      Return sub-shape (or a sub-vector) for input shape defined by
/// range [s_begin;s_end)
///
Shape compute_sub_shape(const Shape& input_shape, size_t begin, size_t end, bool is_product = false) {
    OPENVINO_ASSERT(end <= input_shape.size());
    if (end <= begin) {
        // return empty shape
        return Shape();
    }
    Shape sub_shape(input_shape.begin() + begin, input_shape.begin() + end);

    if (is_product) {
        auto prod = shape_size(sub_shape);
        sub_shape = {prod};
    }
    return sub_shape;
}

/// \brief      Compute ranges of dimension indices for each group of labels:
/// common, reduced and separated
///
void compute_ranges(const Rank& input_rank,
                    const std::string& input_subscript,
                    const std::vector<std::string>& common_labels,
                    const std::vector<std::string>& sep_labels,
                    const std::vector<std::string>& reduced_labels,
                    size_t& common_begin,
                    size_t& common_end,
                    size_t& sep_begin,
                    size_t& sep_end,
                    size_t& reduced_begin,
                    size_t& reduced_end,
                    bool is_separated_first) {
    auto label_to_dim_map = compute_label_dim_map(input_rank, input_subscript);
    constexpr char ellipsis[] = "...";
    size_t common_rank = common_labels.size();
    if (std::find(common_labels.begin(), common_labels.end(), ellipsis) != common_labels.end()) {
        OPENVINO_ASSERT(label_to_dim_map.find(ellipsis) != label_to_dim_map.end());
        common_rank += label_to_dim_map[ellipsis].size() - 1;
    }
    size_t sep_rank = sep_labels.size();
    if (std::find(sep_labels.begin(), sep_labels.end(), ellipsis) != sep_labels.end()) {
        OPENVINO_ASSERT(label_to_dim_map.find(ellipsis) != label_to_dim_map.end());
        sep_rank += label_to_dim_map[ellipsis].size() - 1;
    }
    size_t reduced_rank = reduced_labels.size();
    if (std::find(reduced_labels.begin(), reduced_labels.end(), ellipsis) != reduced_labels.end()) {
        OPENVINO_ASSERT(label_to_dim_map.find(ellipsis) != label_to_dim_map.end());
        reduced_rank += label_to_dim_map[ellipsis].size() - 1;
    }

    common_begin = 0;
    common_end = common_begin + common_rank;
    if (is_separated_first) {
        sep_begin = common_end;
        sep_end = sep_begin + sep_rank;
        reduced_begin = sep_end;
        reduced_end = reduced_begin + reduced_rank;
    } else {
        reduced_begin = common_end;
        reduced_end = reduced_begin + reduced_rank;
        sep_begin = reduced_end;
        sep_end = sep_begin + sep_rank;
    }
}

/// \brief      Compute output shape produced by MatMul operation
///
Shape compute_matmul_output_shape(const Shape& common_sub_shape,
                                  const Shape& separate_sub_shape1,
                                  const Shape& separate_sub_shape2) {
    Shape matmul_output_shape;
    matmul_output_shape.insert(matmul_output_shape.end(), common_sub_shape.begin(), common_sub_shape.end());

    // compute a product of a sub-shape for separate labels
    Shape separate_sub_shape_prod1 = separate_sub_shape1;
    if (common_sub_shape.size() > 0 && separate_sub_shape_prod1.size() == 0) {
        // in this case new dimension corresponding to separate labels must be added
        // since MatMul operation is not possible to do without separate dimensions
        // if the common dimension presents
        separate_sub_shape_prod1.push_back(1);
    } else if (separate_sub_shape_prod1.size() > 0) {
        // in this case compute a product of separate dimension sizes since they
        // must be presented with just one dimension for MatMul
        auto prod = shape_size(separate_sub_shape_prod1);
        separate_sub_shape_prod1 = {prod};
    }
    Shape separate_sub_shape_prod2 = separate_sub_shape2;
    if (common_sub_shape.size() > 0 && separate_sub_shape_prod2.size() == 0) {
        // in this case new dimension corresponding to separate labels must be added
        // since MatMul operation is not possible to do without separate dimensions
        // if the common dimension presents
        separate_sub_shape_prod2.push_back(1);
    } else if (separate_sub_shape_prod2.size() > 0) {
        // in this case compute a product of separate dimension sizes since they
        // must be presented with just one dimension for MatMul
        auto prod = shape_size(separate_sub_shape_prod2);
        separate_sub_shape_prod2 = {prod};
    }
    matmul_output_shape.insert(matmul_output_shape.end(),
                               separate_sub_shape_prod1.begin(),
                               separate_sub_shape_prod1.end());
    matmul_output_shape.insert(matmul_output_shape.end(),
                               separate_sub_shape_prod2.begin(),
                               separate_sub_shape_prod2.end());

    return matmul_output_shape;
}

/// \brief      Update a vector of inputs and subscripts by removing items for
/// inputs with indices input_ind1 and input_ind2 and inserted new input and
/// the corresponsing subscript in the tail
///
void update_operands(ov::TensorVector& inputs,
                     std::vector<std::string>& input_subscripts,
                     size_t input_ind1,
                     size_t input_ind2,
                     const ov::Tensor& new_input,
                     const std::string& new_subscript) {
    OPENVINO_ASSERT(input_ind1 < input_ind2);
    OPENVINO_ASSERT(input_ind2 < inputs.size());
    OPENVINO_ASSERT(input_ind2 < input_subscripts.size());
    inputs.erase(inputs.begin() + input_ind2);
    inputs.erase(inputs.begin() + input_ind1);
    inputs.push_back(new_input);
    input_subscripts.erase(input_subscripts.begin() + input_ind2);
    input_subscripts.erase(input_subscripts.begin() + input_ind1);
    input_subscripts.push_back(new_subscript);
}

/// \brief      Unsqueeze input by given dimensions if a vector of unsqueezing
/// dimensions is not empty
template <typename T>
ov::Tensor unsqueeze_input(const ov::Tensor& input, std::vector<int64_t>& unsqueeze_axes) {
    if (unsqueeze_axes.empty()) {
        return input;
    }

    const auto& input_shape = input.get_shape();
    auto output_shape = input_shape;
    std::sort(unsqueeze_axes.begin(), unsqueeze_axes.end());
    for (auto unsqueeze_axis : unsqueeze_axes) {
        OPENVINO_ASSERT(unsqueeze_axis >= 0);
        OPENVINO_ASSERT(static_cast<size_t>(unsqueeze_axis) <= output_shape.size());
        output_shape.insert(output_shape.begin() + unsqueeze_axis, 1);
    }

    auto output = ov::Tensor(input.get_element_type(), output_shape);
    const auto element_type = input.get_element_type();

    reshape(static_cast<const char*>(input.data()),
            static_cast<char*>(output.data()),
            input_shape,
            element_type.size());

    return output;
}

/// \brief      Find labels (in a given input subscript) that are met once in the
/// equation and reduce dimensions corresponding to such labels
///
template <typename T>
void reduce_input(ov::TensorVector& inputs,
                  std::vector<std::string>& input_subscripts,
                  const std::string& output_subscript,
                  size_t input_ind) {
    // perform sanity check for arguments
    auto num_inputs = inputs.size();
    OPENVINO_ASSERT(num_inputs == input_subscripts.size(), "Each input must have own subscript.");
    OPENVINO_ASSERT(input_ind < num_inputs, "Input index is out of range.");

    const auto& input_ptr = inputs[input_ind];
    const auto& input_subscript = input_subscripts[input_ind];
    const auto& input_shape = input_ptr.get_shape();

    // compute output shape and axes to reduce
    ov::Shape output_shape;
    ov::AxisSet reduced_axes;
    auto labels = ov::op::v7::Einsum::extract_labels(input_subscripts[input_ind]);
    auto label_dim_map = compute_label_dim_map(input_ptr.get_shape().size(), input_subscript);
    std::string new_input_subscript = "";
    for (const auto& label : labels) {
        // check if the current label is met in the other input subscripts
        // or the output subscript
        bool is_dim_reduced = is_dimension_reduced(input_subscripts, output_subscript, label, {input_ind});

        OPENVINO_ASSERT(label_dim_map.find(label) != label_dim_map.end());
        auto label_dims = label_dim_map[label];

        // if label is not met, dimension corresponding to the label is to reduce
        if (is_dim_reduced) {
            reduced_axes.insert(label_dims.begin(), label_dims.end());
        } else {
            for (auto label_dim : label_dims) {
                output_shape.push_back(input_shape[label_dim]);
            }
            new_input_subscript += label;
        }
    }

    if (reduced_axes.size() == 0) {
        // there is no axis to reduce
        return;
    }

    auto output_ptr = ov::Tensor(input_ptr.get_element_type(), output_shape);

    reference::reduce_sum(input_ptr.data<T>(), output_ptr.data<T>(), input_shape, reduced_axes);

    // update a vector of inputs and input subscripts
    inputs[input_ind] = std::move(output_ptr);
    input_subscripts[input_ind] = std::move(new_input_subscript);
}

/// \brief      Transpose input to layout specified through the required subscript
///
template <typename T>
void transpose_input(ov::TensorVector& inputs,
                     std::vector<std::string>& input_subscripts,
                     const std::string& required_subscript,
                     size_t input_ind) {
    // perform sanity check for arguments
    auto num_inputs = inputs.size();
    OPENVINO_ASSERT(num_inputs == input_subscripts.size(), "Each input must have own subscript.");
    OPENVINO_ASSERT(input_ind < num_inputs, "Input index is out of range.");

    // generate permutation vector by searching for bijection between
    // input_subscripts and required_subscript
    std::vector<int64_t> permutation;
    const auto& input_ptr = inputs[input_ind];
    const auto& input_subscript = input_subscripts[input_ind];

    // transpose is not needed since the input subscript is not going to be changed
    if (required_subscript == input_subscript) {
        return;
    }

    // find permutation that establishes bijection between the input subscript
    // and the required one
    auto label_dim_map = compute_label_dim_map(input_ptr.get_shape().size(), input_subscript);
    auto labels = ov::op::v7::Einsum::extract_labels(input_subscript);
    auto required_labels = ov::op::v7::Einsum::extract_labels(required_subscript);
    OPENVINO_ASSERT(labels.size() == required_labels.size());
    for (const auto& required_label : required_labels) {
        OPENVINO_ASSERT(label_dim_map.find(required_label) != label_dim_map.end());
        auto label_dims = label_dim_map[required_label];
        permutation.insert(permutation.end(), label_dims.begin(), label_dims.end());
    }

    const auto& input_shape = input_ptr.get_shape();
    const auto& element_type = input_ptr.get_element_type();

    Shape output_shape(input_shape.size());
    std::transform(permutation.begin(), permutation.end(), output_shape.begin(), [&](const int64_t& v) {
        OPENVINO_ASSERT(v >= 0, "Negative values for transpose axes order are not supported.");
        OPENVINO_ASSERT(v < int64_t(input_shape.size()), "Transpose axis ", v, " is out of shape range.");
        return input_shape[v];
    });
    auto output_ptr = ov::Tensor(element_type, output_shape);

    reference::transpose(reinterpret_cast<const char*>(input_ptr.data<T>()),
                         reinterpret_cast<char*>(output_ptr.data<T>()),
                         input_shape,
                         element_type.size(),
                         permutation,
                         output_shape);

    // update a vector of inputs and input subscripts
    inputs[input_ind] = std::move(output_ptr);
    input_subscripts[input_ind] = required_subscript;
}

/// \brief      Broadcast input to a new shape. The MatMul operation requires the
/// same shape of both operands in the common (or batch) dimensions.
///
template <typename T>
void broadcast_input(ov::TensorVector& inputs,
                     size_t input_ind,
                     const Shape& new_common_shape,
                     const Shape& separate_shape,
                     const Shape& reduced_shape,
                     bool is_separate_first) {
    OPENVINO_ASSERT(input_ind < inputs.size());
    ov::Tensor& input = inputs[input_ind];
    const Shape old_shape = input.get_shape();
    PartialShape new_shape;
    new_shape.insert(new_shape.end(), new_common_shape.begin(), new_common_shape.end());
    if (is_separate_first) {
        new_shape.insert(new_shape.end(), separate_shape.begin(), separate_shape.end());
        new_shape.insert(new_shape.end(), reduced_shape.begin(), reduced_shape.end());
    } else {
        new_shape.insert(new_shape.end(), reduced_shape.begin(), reduced_shape.end());
        new_shape.insert(new_shape.end(), separate_shape.begin(), separate_shape.end());
    }

    if (input.get_shape() == new_shape.to_shape()) {
        return;
    }
    OPENVINO_ASSERT(old_shape.size() <= new_shape.size());

    std::vector<size_t> broadcast_axes(old_shape.size());
    std::iota(broadcast_axes.begin(), broadcast_axes.end(), new_shape.size() - old_shape.size());
    OPENVINO_ASSERT(PartialShape::broadcast_merge_into(new_shape, old_shape, ov::op::AutoBroadcastType::NUMPY));
    auto output = ov::Tensor(input.get_element_type(), new_shape.to_shape());

    reference::broadcast(reinterpret_cast<const char*>(input.data<T>()),
                         reinterpret_cast<char*>(output.data<T>()),
                         input.get_shape(),
                         output.get_shape(),
                         broadcast_axes,
                         input.get_element_type().size());

    input = std::move(output);
}

/// \brief      Build identity tensor that will be used to zero non-diagonal tensor
/// elements by element-wise multiplication of the input tensor and the built
/// identity
///
template <typename T>
ov::Tensor build_identity(const ov::Tensor& input, const std::vector<size_t>& repeated_label_dims) {
    // allocate Tensor for building identity tensor
    OPENVINO_ASSERT(repeated_label_dims.size() > 1);
    Shape input_shape = input.get_shape();
    Shape identity_shape(input_shape.size(), 1);
    size_t repeated_label_dim_size = input_shape[repeated_label_dims[0]];
    for (auto dim : repeated_label_dims) {
        OPENVINO_ASSERT(dim < input_shape.size());
        OPENVINO_ASSERT(repeated_label_dim_size == input_shape[dim]);
        identity_shape[dim] = repeated_label_dim_size;
    }
    auto identity = ov::Tensor(input.get_element_type(), identity_shape);

    T* identity_data_ptr = identity.data<T>();
    size_t data_size = shape_size(identity_shape) * identity.get_element_type().size();
    std::memset(identity_data_ptr, 0, data_size);

    // Identity[k,k,...,k] element is placed in k*p^(n-1) + ... + k*p + k position,
    // where p is a size of one Identity dimension,
    // n is occurrence number for the considered label and k in [0; p).
    // Note that k*p^(n-1) + ... + k*p + k = k * (p^n-1)/(p-1) = k * alpha
    size_t p = repeated_label_dim_size;
    if (p == 1) {
        identity_data_ptr[0] = static_cast<T>(1);
        return identity;
    }

    size_t n = repeated_label_dims.size();
    size_t alpha = (static_cast<size_t>(std::pow(p, n)) - 1) / (p - 1);
    size_t offset = 0;
    for (size_t k = 0; k < p; ++k) {
        identity_data_ptr[offset] = static_cast<T>(1);
        offset += alpha;
    }

    return identity;
}

/// \brief      Computes a multiplication of Identity tensors built for each
/// repeated label
///
template <typename T>
ov::Tensor build_multi_identity(const ov::Tensor& input,
                                const std::vector<std::string>& repeated_labels,
                                std::unordered_map<std::string, std::vector<size_t>>& label_dim_map) {
    Shape input_shape = input.get_shape();

    // initially set multi-identity with identity for the first repeated label
    OPENVINO_ASSERT(repeated_labels.size() > 0);
    const auto& first_repeated_label = repeated_labels[0];
    OPENVINO_ASSERT(label_dim_map.find(first_repeated_label) != label_dim_map.end());
    auto repeated_label_dims = label_dim_map[first_repeated_label];
    ov::Tensor multi_identity = build_identity<T>(input, repeated_label_dims);

    for (size_t label_ind = 1; label_ind < repeated_labels.size(); ++label_ind) {
        OPENVINO_ASSERT(label_dim_map.find(repeated_labels[label_ind]) != label_dim_map.end());
        repeated_label_dims = label_dim_map[repeated_labels[label_ind]];
        ov::Tensor identity = build_identity<T>(input, repeated_label_dims);

        PartialShape output_shape = multi_identity.get_shape();
        PartialShape::broadcast_merge_into(output_shape, identity.get_shape(), ov::op::AutoBroadcastType::NUMPY);
        auto mul_output = ov::Tensor(identity.get_element_type(), output_shape.get_shape());
        reference::multiply<T>(multi_identity.data<T>(),
                               identity.data<T>(),
                               mul_output.data<T>(),
                               multi_identity.get_shape(),
                               identity.get_shape(),
                               ov::op::AutoBroadcastType::NUMPY);
        multi_identity = std::move(mul_output);
    }
    return multi_identity;
}

/// \brief      Computes a tensor that along some dimensions (specified by repeated
/// labels) is diagonal
///
template <typename T>
void extract_diagonal(ov::TensorVector& inputs, std::vector<std::string>& input_subscripts, size_t input_ind) {
    // perform sanity check for arguments
    auto num_inputs = inputs.size();
    OPENVINO_ASSERT(num_inputs == input_subscripts.size(), "Each input must have own subscript.");
    OPENVINO_ASSERT(input_ind < num_inputs, "Input index is out of range.");

    const auto& input_ptr = inputs[input_ind];
    const auto& input_subscript = input_subscripts[input_ind];
    const auto& input_shape = input_ptr.get_shape();

    std::string resultant_subscript = "";
    constexpr char ellipsis[] = "...";
    auto labels = ov::op::v7::Einsum::extract_labels(input_subscript);
    auto label_dim_map = compute_label_dim_map(input_ptr.get_shape().size(), input_subscript);
    std::vector<std::string> repeated_labels;
    Shape result_shape;
    AxisSet reduced_axes;
    for (const auto& label : labels) {
        if (resultant_subscript.find(label) == std::string::npos) {
            OPENVINO_ASSERT(label_dim_map.find(label) != label_dim_map.end());
            auto dims = label_dim_map[label];
            OPENVINO_ASSERT(dims.size() > 0);
            if (label != ellipsis && dims.size() > 1) {
                // repeated label is found
                for (size_t dim_ind = 1; dim_ind < dims.size(); ++dim_ind) {
                    reduced_axes.insert(dims[dim_ind]);
                }
                // save only the first dimension corresponding to the repeated label
                dims = {dims[0]};
                repeated_labels.push_back(label);
            }
            resultant_subscript += label;
            for (auto dim : dims) {
                OPENVINO_ASSERT(dim < input_shape.size());
                result_shape.push_back(input_shape[dim]);
            }
        }
    }
    if (input_shape == result_shape) {
        return;
    }

    ov::Tensor multi_identity = build_multi_identity<T>(input_ptr, repeated_labels, label_dim_map);

    ov::Tensor mul_output = input_ptr;
    reference::multiply<T>(input_ptr.data<T>(),
                           multi_identity.data<T>(),
                           mul_output.data<T>(),
                           input_ptr.get_shape(),
                           multi_identity.get_shape(),
                           ov::op::AutoBroadcastType::NUMPY);

    auto result = ov::Tensor(input_ptr.get_element_type(), result_shape);
    reference::reduce_sum(mul_output.data<T>(), result.data<T>(), mul_output.get_shape(), reduced_axes);
    inputs[input_ind] = std::move(result);
    input_subscripts[input_ind] = std::move(resultant_subscript);
}

/// \brief      Reshape input to the new shape specified by sub-shapes of the
/// common, separate and reduced dimensions so that the reshaped input has a format
/// acceptable by MatMul
///
template <typename T>
ov::Tensor reshape_input_for_matmul(const ov::Tensor& input,
                                    const Shape& common_sub_shape,
                                    const Shape& separate_sub_shape,
                                    const Shape& reduced_sub_shape_prod,
                                    bool is_separate_first) {
    Shape new_shape;
    new_shape.insert(new_shape.end(), common_sub_shape.begin(), common_sub_shape.end());

    // compute a product of a sub-shape for separate labels
    Shape separate_sub_shape_prod = separate_sub_shape;
    if (common_sub_shape.size() > 0 && separate_sub_shape_prod.size() == 0) {
        // in this case new dimension corresponding to separate labels must be added
        // since MatMul operation is not possible to do without separate dimensions
        // if the common dimension presents
        separate_sub_shape_prod.push_back(1);
    } else if (separate_sub_shape_prod.size() > 0) {
        // in this case compute a product of separate dimension sizes since they
        // must be presented with just one dimension for MatMul
        auto prod = shape_size(separate_sub_shape_prod);
        separate_sub_shape_prod = {prod};
    }

    // form a new shape for input so that collapsed dimensions corresponding
    // to the common, separate and reduced dimensions are placed in the correct
    // order
    if (is_separate_first) {
        new_shape.insert(new_shape.end(), separate_sub_shape_prod.begin(), separate_sub_shape_prod.end());
        new_shape.insert(new_shape.end(), reduced_sub_shape_prod.begin(), reduced_sub_shape_prod.end());
    } else {
        new_shape.insert(new_shape.end(), reduced_sub_shape_prod.begin(), reduced_sub_shape_prod.end());
        new_shape.insert(new_shape.end(), separate_sub_shape_prod.begin(), separate_sub_shape_prod.end());
    }

    // when new shape is equal to the current one,
    // there is no need in reshape
    if (new_shape == input.get_shape()) {
        return input;
    }

    const auto element_type = input.get_element_type();
    const auto& input_shape = input.get_shape();
    auto output = ov::Tensor(element_type, new_shape);

    reshape(static_cast<const char*>(input.data()),
            static_cast<char*>(output.data()),
            input_shape,
            element_type.size());
    return output;
}

/// \brief Adjusts the rank of two input tensors by unsqueezing ellipses to the same rank.
///
/// This function takes two input tensors and their corresponding subscripts, and ensures that
/// the ellipses ("...") in the subscripts have the same rank by unsqueezing dimensions as needed.
/// It modifies the input tensors in place.
///
/// \param inputs A vector of input tensors.
/// \param input_subscripts A vector of strings representing the subscripts for each input tensor.
/// \param input_ind1 The index of the first input tensor in the inputs vector.
/// \param input_ind2 The index of the second input tensor in the inputs vector.
template <typename T>
void unsqueeze_ellipses_to_same_rank(ov::TensorVector& inputs,
                                     std::vector<std::string>& input_subscripts,
                                     size_t input_ind1,
                                     size_t input_ind2) {
    constexpr char ellipsis[] = "...";
    const auto& input1 = inputs[input_ind1];
    const auto& input2 = inputs[input_ind2];
    auto label_to_dim_map1 = compute_label_dim_map(input1.get_shape().size(), input_subscripts[input_ind1]);
    auto label_to_dim_map2 = compute_label_dim_map(input2.get_shape().size(), input_subscripts[input_ind2]);
    if (label_to_dim_map1.find(ellipsis) != label_to_dim_map1.end() &&
        label_to_dim_map2.find(ellipsis) != label_to_dim_map2.end()) {
        std::vector<int64_t> unsqueeze_axis1, unsqueeze_axis2;
        const auto& ellipsis_dims1 = label_to_dim_map1[ellipsis];
        const auto& ellipsis_dims2 = label_to_dim_map2[ellipsis];
        if (ellipsis_dims2.size() > ellipsis_dims1.size()) {
            for (size_t i = 0; i < ellipsis_dims2.size() - ellipsis_dims1.size(); ++i) {
                unsqueeze_axis1.push_back(ellipsis_dims1[0] + i);
            }
        } else if (ellipsis_dims1.size() > ellipsis_dims2.size()) {
            for (size_t i = 0; i < ellipsis_dims1.size() - ellipsis_dims2.size(); ++i) {
                unsqueeze_axis2.push_back(ellipsis_dims2[0] + i);
            }
        }
        ov::Tensor unsqueeze_output1 = unsqueeze_input<T>(input1, unsqueeze_axis1);
        ov::Tensor unsqueeze_output2 = unsqueeze_input<T>(input2, unsqueeze_axis2);
        inputs[input_ind1] = std::move(unsqueeze_output1);
        inputs[input_ind2] = std::move(unsqueeze_output2);
        return;
    }
}

/// \brief      Contract two inputs of Einsum operation according to equation.
/// The result of the contraction is appended into inputs along with its
/// subscript. The inputs with indices input_ind1 and input_ind2 are removed from
/// inputs along with their input subscripts
///
template <typename T>
void contract_two_inputs(ov::TensorVector& inputs,
                         std::vector<std::string>& input_subscripts,
                         const std::string& output_subscript,
                         size_t input_ind1,
                         size_t input_ind2) {
    // assume that input_ind1 < input_ind2 without loss of generality, otherwise,
    // just swap them
    if (input_ind2 < input_ind1) {
        std::swap(input_ind1, input_ind2);
    }

    // perform sanity check for arguments
    auto num_inputs = inputs.size();
    OPENVINO_ASSERT(num_inputs == input_subscripts.size(), "Each input must have own subscript.");
    OPENVINO_ASSERT(input_ind2 < num_inputs && input_ind1 != input_ind2, "Incorrect input index is specified.");

    const auto& input1 = inputs[input_ind1];
    const auto& input2 = inputs[input_ind2];

    // unsqueeze inputs to have same rank of ellipsis for correct broadcasting
    unsqueeze_ellipses_to_same_rank<T>(inputs, input_subscripts, input_ind1, input_ind2);

    // extract diagonals in case repeated labels in the corresponding input
    // subscripts
    extract_diagonal<T>(inputs, input_subscripts, input_ind1);
    extract_diagonal<T>(inputs, input_subscripts, input_ind2);

    // reduce dimensions for input operands if possible
    reduce_input<T>(inputs, input_subscripts, output_subscript, input_ind1);
    reduce_input<T>(inputs, input_subscripts, output_subscript, input_ind2);

    // step 0. split dimensions of both operands into three groups:
    // 1. dimension indices with the same labels (in both subscripts) that are NOT
    // reduced - common labels (dimensions)
    // 2. dimension indices with labels that are met only in one of two subscripts -
    // separate labels (dimensions)
    // 3. dimension indices with the same labels (in both subscripts) that are
    // reduced - reduced labels (dimensions) NOTE: dimension is reduced if the
    // corresponding label are met in neither the output subscript nor the input
    // subscripts for other Einsum inputs excluding two given inputs
    auto& input_subscript1 = input_subscripts[input_ind1];
    auto labels1 = ov::op::v7::Einsum::extract_labels(input_subscript1);
    auto& input_subscript2 = input_subscripts[input_ind2];
    auto labels2 = ov::op::v7::Einsum::extract_labels(input_subscript2);
    std::string common_part = "";
    std::string separate_part1 = "";
    std::string separate_part2 = "";
    std::vector<int64_t> common_labels_inds1, common_labels_inds2;
    std::vector<int64_t> separate_labels_inds1, separate_labels_inds2;
    std::vector<int64_t> reduced_labels_inds1, reduced_labels_inds2;
    std::vector<std::string> common_labels, sep_labels1, sep_labels2, reduced_labels;
    for (size_t label_ind = 0; label_ind < labels1.size(); ++label_ind) {
        const auto& label = labels1[label_ind];
        auto iter = std::find(labels2.begin(), labels2.end(), label);
        if (iter != labels2.end()) {
            bool is_dim_reduced =
                is_dimension_reduced(input_subscripts, output_subscript, label, {input_ind1, input_ind2});
            common_part += label;
            if (is_dim_reduced) {
                reduced_labels_inds1.push_back(static_cast<int64_t>(label_ind));
                reduced_labels_inds2.push_back(static_cast<int64_t>(iter - labels2.begin()));
                reduced_labels.push_back(label);
            } else {
                common_labels_inds1.push_back(static_cast<int64_t>(label_ind));
                common_labels_inds2.push_back(static_cast<int64_t>(iter - labels2.begin()));
                common_labels.push_back(label);
            }
        } else {
            separate_part1 += label;
            separate_labels_inds1.push_back(static_cast<int64_t>(label_ind));
            sep_labels1.push_back(label);
        }
    }
    for (size_t label_ind = 0; label_ind < labels2.size(); ++label_ind) {
        const auto& label = labels2[label_ind];
        auto iter = std::find(labels1.begin(), labels1.end(), label);
        if (iter == labels1.end()) {
            separate_part2 += label;
            separate_labels_inds2.push_back(static_cast<int64_t>(label_ind));
            sep_labels2.push_back(label);
        }
    }

    // if there is no common dimension to reduce, apply eltwise multiplication
    if (reduced_labels_inds1.empty()) {
        std::string convenient_subscript = common_part + separate_part2;
        std::string resultant_subscript = input_subscript1 + separate_part2;

        // transpose the second operand in order to get the convenient layout
        // for further unsqueezing
        transpose_input<T>(inputs, input_subscripts, convenient_subscript, input_ind2);

        auto separate_labels1 = ov::op::v7::Einsum::extract_labels(separate_part1);
        auto separate_labels2 = ov::op::v7::Einsum::extract_labels(separate_part2);
        auto label_to_dim_map1 = compute_label_dim_map(input1.get_shape().size(), input_subscript1);
        auto label_to_dim_map2 = compute_label_dim_map(input2.get_shape().size(), input_subscript2);

        // unsqueeze the first operand with new dimensions in the tail
        // and the number of them is equal to the number of separate labels in the
        // second subscript
        int64_t input_rank1 = input1.get_shape().size();
        int64_t unsqueeze_dim = input_rank1;
        std::vector<int64_t> unsqueeze_axis1;
        std::vector<int64_t> unsqueeze_axis2;
        for (const auto& sep_label2 : separate_labels2) {
            OPENVINO_ASSERT(label_to_dim_map2.find(sep_label2) != label_to_dim_map2.end());
            const auto& label_dims = label_to_dim_map2[sep_label2];
            for (size_t dim_ind = 0; dim_ind < label_dims.size(); ++dim_ind) {
                unsqueeze_axis1.push_back(unsqueeze_dim + static_cast<int64_t>(dim_ind));
            }
        }
        for (const auto& sep_label1 : separate_labels1) {
            OPENVINO_ASSERT(label_to_dim_map1.find(sep_label1) != label_to_dim_map1.end());
            auto label_dims = label_to_dim_map1[sep_label1];
            for (auto label_dim : label_dims) {
                unsqueeze_axis2.push_back(label_dim);
            }
        }

        // unsqueeze input operands for elementwise-multiplication with broadcasting
        ov::Tensor unsqueeze_output1 = unsqueeze_input<T>(input1, unsqueeze_axis1);
        ov::Tensor unsqueeze_output2 = unsqueeze_input<T>(input2, unsqueeze_axis2);

        // multiply both operands with broadcasting
        PartialShape output_shape = unsqueeze_output1.get_shape();
        PartialShape::broadcast_merge_into(output_shape,
                                           unsqueeze_output2.get_shape(),
                                           ov::op::AutoBroadcastType::NUMPY);
        auto mul_output = ov::Tensor(unsqueeze_output1.get_element_type(), output_shape.get_shape());
        reference::multiply<T>(unsqueeze_output1.data<T>(),
                               unsqueeze_output2.data<T>(),
                               mul_output.data<T>(),
                               unsqueeze_output1.get_shape(),
                               unsqueeze_output2.get_shape(),
                               ov::op::AutoBroadcastType::NUMPY);

        // update input operand and input subscript for Einsum operation
        update_operands(inputs, input_subscripts, input_ind1, input_ind2, mul_output, resultant_subscript);
        return;
    }

    // in this case a set of reduced labels is not empty and it can apply MatMul
    // operation step 1. transpose both operands so that common labels, separated
    // and reduced labels are grouped for both operands
    bool is_separate_first1 = false;
    auto int_subscript1 = generate_grouping_subscript(input_subscript1,
                                                      common_labels_inds1,
                                                      separate_labels_inds1,
                                                      reduced_labels_inds1,
                                                      is_separate_first1);
    transpose_input<T>(inputs, input_subscripts, int_subscript1, input_ind1);
    bool is_separate_first2 = false;
    auto int_subscript2 = generate_grouping_subscript(input_subscript2,
                                                      common_labels_inds2,
                                                      separate_labels_inds2,
                                                      reduced_labels_inds2,
                                                      is_separate_first2);
    transpose_input<T>(inputs, input_subscripts, int_subscript2, input_ind2);

    // step 2. reshape both operands so that separate labels and reduced labels are
    // represented with just one dimension this is needed by MatMul operation
    // requirement to operands format. For example, the shape must be in a format
    // [B1, ..., Bm, X1, Y] or [B1, ..., Bm, Y, X2], where B1, ..., Bm are common
    // dimensions, X1 and X2 are collapsed dimensions for separate labels and Y is
    // collapsed dimension for reduced labels
    Shape input_shape1 = input1.get_shape();
    Shape input_shape2 = input2.get_shape();
    size_t common_dims_begin, common_dims_end, reduced_dims_begin, reduced_dims_end, separate1_dims_begin,
        separate1_dims_end;
    compute_ranges(input1.get_shape().size(),
                   input_subscript1,
                   common_labels,
                   sep_labels1,
                   reduced_labels,
                   common_dims_begin,
                   common_dims_end,
                   separate1_dims_begin,
                   separate1_dims_end,
                   reduced_dims_begin,
                   reduced_dims_end,
                   is_separate_first1);

    size_t common_dims_begin2, common_dims_end2, reduced_dims_begin2, reduced_dims_end2, separate2_dims_begin,
        separate2_dims_end;
    compute_ranges(input2.get_shape().size(),
                   input_subscript2,
                   common_labels,
                   sep_labels2,
                   reduced_labels,
                   common_dims_begin2,
                   common_dims_end2,
                   separate2_dims_begin,
                   separate2_dims_end,
                   reduced_dims_begin2,
                   reduced_dims_end2,
                   is_separate_first2);

    PartialShape common_sub_shape1 = compute_sub_shape(input_shape1, common_dims_begin, common_dims_end);
    PartialShape common_sub_shape2 = compute_sub_shape(input_shape2, common_dims_begin2, common_dims_end2);

    PartialShape reduced_sub_shape = compute_sub_shape(input_shape1, reduced_dims_begin, reduced_dims_end);
    Shape reduced_sub_shape2 = compute_sub_shape(input_shape2, reduced_dims_begin2, reduced_dims_end2);
    Shape separate1_sub_shape = compute_sub_shape(input_shape1, separate1_dims_begin, separate1_dims_end);
    Shape separate2_sub_shape = compute_sub_shape(input_shape2, separate2_dims_begin, separate2_dims_end);

    // broadcast both inputs to have common sub-shape broadcasted that is needed
    // in case of ellipsis among the common labels
    // reference::broadcast()
    PartialShape::broadcast_merge_into(common_sub_shape1, common_sub_shape2, op::AutoBroadcastType::NUMPY);
    PartialShape::broadcast_merge_into(reduced_sub_shape, reduced_sub_shape2, op::AutoBroadcastType::NUMPY);
    Shape reduced_sub_shape_prod = {shape_size(reduced_sub_shape.get_shape())};
    Shape common_sub_shape = common_sub_shape1.get_shape();
    broadcast_input<T>(inputs,
                       input_ind1,
                       common_sub_shape,
                       separate1_sub_shape,
                       reduced_sub_shape.get_shape(),
                       is_separate_first1);
    broadcast_input<T>(inputs,
                       input_ind2,
                       common_sub_shape,
                       separate2_sub_shape,
                       reduced_sub_shape.get_shape(),
                       is_separate_first2);

    ov::Tensor matmul_operand1 = reshape_input_for_matmul<T>(input1,
                                                             common_sub_shape,
                                                             separate1_sub_shape,
                                                             reduced_sub_shape_prod,
                                                             is_separate_first1);

    ov::Tensor matmul_operand2 = reshape_input_for_matmul<T>(input2,
                                                             common_sub_shape,
                                                             separate2_sub_shape,
                                                             reduced_sub_shape_prod,
                                                             is_separate_first2);

    // step 3. apply MatMul operation for formatted inputs
    Shape matmul_output_shape = compute_matmul_output_shape(common_sub_shape, separate1_sub_shape, separate2_sub_shape);
    auto matmul_output = ov::Tensor(matmul_operand1.get_element_type(), matmul_output_shape);

    bool transpose_a = (is_separate_first1 ? false : true);
    bool transpose_b = (is_separate_first2 ? true : false);
    reference::matmul(matmul_operand1.data<T>(),
                      matmul_operand2.data<T>(),
                      matmul_output.data<T>(),
                      matmul_operand1.get_shape(),
                      matmul_operand2.get_shape(),
                      matmul_output_shape,
                      transpose_a,
                      transpose_b);

    // step 4. reshape back by unrolling dimensions corresponding to separate labels
    // if needed now dimensions corresponding to reduced labels are reduced by the
    // MatMul operation
    common_part = "";
    for (const auto& common_label : common_labels) {
        common_part += common_label;
    }
    std::string resultant_subscript = common_part + separate_part1 + separate_part2;
    Shape back_shape;
    back_shape.insert(back_shape.end(), common_sub_shape.begin(), common_sub_shape.end());
    back_shape.insert(back_shape.end(), separate1_sub_shape.begin(), separate1_sub_shape.end());
    back_shape.insert(back_shape.end(), separate2_sub_shape.begin(), separate2_sub_shape.end());

    auto contract_output = ov::Tensor(matmul_output.get_element_type(), back_shape);
    reshape(static_cast<const char*>(matmul_output.data()),
            static_cast<char*>(contract_output.data()),
            matmul_output.get_shape(),
            matmul_output.get_element_type().size());

    update_operands(inputs, input_subscripts, input_ind1, input_ind2, contract_output, resultant_subscript);
}

/// \brief Adjusts input subscripts and nodes to handle 0-dimensional ellipsis in Einsum operations.
///
/// Handle ellipses labels that do not represent any dimensions:
/// 1. If there is no ellipsis in the input subscripts, remove ellipsis from the output subscript.
/// 2. If all ellipses in the input subscripts do not represent any dimensions, remove ellipses from all subscripts.
/// 3. If there is at least one ellipsis that represents dimension, unsqueeze ellipses that do not represent any,
///
/// \param input_nodes A vector of input tensors for the Einsum operation.
/// \param input_subscripts A vector of input subscripts corresponding to the input nodes.
/// \param output_subscript The output subscript for the Einsum operation.
template <typename T>
void fix_inputs_with_0d_ellipsis(ov::TensorVector& input_nodes,
                                 std::vector<std::string>& input_subscripts,
                                 std::string& output_subscript) {
    static const std::string ellipsis = "...";
    bool has_ellipsis = false;
    bool all_no_ellipsis_or_empty = true;

    for (size_t i = 0; i < input_nodes.size(); ++i) {
        const auto& labels = ov::op::v7::Einsum::extract_labels(input_subscripts[i]);
        bool has_ellipsis_in_input = std::find(labels.begin(), labels.end(), ellipsis) != labels.end();
        has_ellipsis |= has_ellipsis_in_input;
        all_no_ellipsis_or_empty &=
            !has_ellipsis_in_input || (input_nodes[i].get_shape().size() == (labels.size() - 1));
    }

    if (!has_ellipsis) {
        if (output_subscript.find(ellipsis) != std::string::npos) {
            output_subscript.erase(output_subscript.find(ellipsis), ellipsis.size());
        }
    } else if (all_no_ellipsis_or_empty) {
        for (auto& subscript : input_subscripts) {
            if (subscript.find(ellipsis) != std::string::npos) {
                subscript.erase(subscript.find(ellipsis), ellipsis.size());
            }
        }
        if (output_subscript.find(ellipsis) != std::string::npos) {
            output_subscript.erase(output_subscript.find(ellipsis), ellipsis.size());
        }
    } else {
        for (size_t i = 0; i < input_nodes.size(); ++i) {
            const auto& labels = ov::op::v7::Einsum::extract_labels(input_subscripts[i]);
            if (std::find(labels.begin(), labels.end(), ellipsis) != labels.end() &&
                input_nodes[i].get_shape().size() == (labels.size() - 1)) {
                std::vector<int64_t> ellipsis_idx{
                    std::distance(labels.begin(), std::find(labels.begin(), labels.end(), ellipsis))};
                input_nodes[i] = unsqueeze_input<T>(input_nodes[i], ellipsis_idx);
            }
        }
    }
}

template <typename T>
void einsum_impl(const ov::TensorVector& inputs, ov::TensorVector& outputs, const std::string& equation) {
    std::vector<std::string> input_subscripts;
    std::string output_subscript;
    ov::op::v7::Einsum::parse_equation(equation, input_subscripts, output_subscript);

    // compute einsum path that is used to contract a pair of operands
    // in more optimal order
    size_t num_inputs = inputs.size();
    auto einsum_path = compute_einsum_path(num_inputs);
    ov::TensorVector int_inputs = inputs;

    // fix inputs where ellipsis does not contain any dimensions
    fix_inputs_with_0d_ellipsis<T>(int_inputs, input_subscripts, output_subscript);

    // contract inputs by Einsum until just one is remained
    for (auto const& inds_pair : einsum_path) {
        contract_two_inputs<T>(int_inputs, input_subscripts, output_subscript, inds_pair.first, inds_pair.second);
    }

    OPENVINO_ASSERT(int_inputs.size() == 1);

    // extract diagonal for the single operand
    extract_diagonal<T>(int_inputs, input_subscripts, 0);

    // reduce dimensions for the remained input node
    reduce_input<T>(int_inputs, input_subscripts, output_subscript, 0);

    // transpose dimensions to layout required by the output subscript
    transpose_input<T>(int_inputs, input_subscripts, output_subscript, 0);

    int_inputs[0].copy_to(outputs[0]);
}

}  // namespace

void einsum(ov::TensorVector& outputs, const ov::TensorVector& inputs, const std::string& equation) {
    OPENVINO_ASSERT(inputs.size() > 0, "Einsum must accept at least one input.");
    auto input_type = inputs[0].get_element_type();
    for (size_t input_ind = 1; input_ind < inputs.size(); ++input_ind) {
        OPENVINO_ASSERT(inputs[input_ind].get_element_type() == input_type, "Input types must be the same.");
    }
    if (input_type == element::Type_t::f32) {
        einsum_impl<float>(inputs, outputs, equation);
    } else if (input_type == element::Type_t::i32) {
        einsum_impl<int>(inputs, outputs, equation);
    } else {
        OPENVINO_ASSERT(false, "Unsupported input type for Einsum operation.");
    }
}

}  // namespace reference
}  // namespace ov
