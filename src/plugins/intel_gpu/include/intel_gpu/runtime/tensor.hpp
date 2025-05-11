// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "format.hpp"
#include "compounds.hpp"
#include "utils.hpp"

#include <openvino/core/partial_shape.hpp>

#include <map>
#include <list>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <vector>
#include <string>
#include <utility>
#include <functional>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_memory Memory description and management
/// @{

constexpr int32_t tensor_batch_dim_max = 1;
constexpr int32_t tensor_feature_dim_max = 1;
constexpr int32_t tensor_spatial_dim_max = 6;
constexpr int32_t tensor_group_dim_max = 1;
constexpr int32_t tensor_dim_max = tensor_batch_dim_max + tensor_feature_dim_max + tensor_spatial_dim_max + tensor_group_dim_max;

struct tensor;

/// @brief Helper structs used in tensor constructor with dim_vec_kinds
namespace details {
/// @brief enum class that represent dimension kinds
enum class dim_vec_kind {
    batch,
    feature,
    spatial,
    group
};

/// @brief template class with max_dimensionalities and dimension offset for dimension kinds
template <dim_vec_kind Kind>
struct dim_vec_limits {
    static_assert(meta::always_false_ty_val<dim_vec_kind, Kind>::value, "Limits are undefined for selected value of dim_vec_kind.");
};

template <>
struct dim_vec_limits<dim_vec_kind::batch> {
    static constexpr int32_t max_dimentionality = tensor_batch_dim_max;
    static constexpr int32_t dim_offset = 0;
};

template <>
struct dim_vec_limits<dim_vec_kind::feature> {
    static constexpr int32_t max_dimentionality = tensor_feature_dim_max;
    static constexpr int32_t dim_offset = tensor_batch_dim_max;
};

template <>
struct dim_vec_limits<dim_vec_kind::spatial> {
    static constexpr int32_t max_dimentionality = tensor_spatial_dim_max;
    static constexpr int32_t dim_offset = tensor_batch_dim_max + tensor_feature_dim_max;
};

template <>
struct dim_vec_limits<dim_vec_kind::group> {
    static constexpr int32_t max_dimentionality = tensor_group_dim_max;
    static constexpr int32_t dim_offset = tensor_batch_dim_max + tensor_feature_dim_max + tensor_spatial_dim_max;
};

/// @brief Template class used in tensor constructor using dim_vec_kinds
template <dim_vec_kind Kind>
class dim_vec_kind_init {
public:
    static constexpr auto _max_dimensionality = dim_vec_limits<Kind>::max_dimentionality;
    static constexpr auto _dimOffset = dim_vec_limits<Kind>::dim_offset;

    template <typename... DimTys>
    explicit dim_vec_kind_init(DimTys&&... values)
        : _sizes{int32_t(std::forward<DimTys>(values))...}, _dimSize(sizeof...(DimTys)) {
    }

    void init_tensor_values(cldnn::tensor& t);

    int32_t _sizes[_max_dimensionality];
    int32_t _dimSize;
};
}  // namespace details

template <typename... InitTys>
details::dim_vec_kind_init<details::dim_vec_kind::batch> batch(InitTys&&... inits) {
    return details::dim_vec_kind_init<details::dim_vec_kind::batch>(std::forward<InitTys>(inits)...);
}

template <typename... InitTys>
details::dim_vec_kind_init<details::dim_vec_kind::feature> feature(InitTys&&... inits) {
    return details::dim_vec_kind_init<details::dim_vec_kind::feature>(std::forward<InitTys>(inits)...);
}

template <typename... InitTys>
details::dim_vec_kind_init<details::dim_vec_kind::spatial> spatial(InitTys&&... inits) {
    return details::dim_vec_kind_init<details::dim_vec_kind::spatial>(std::forward<InitTys>(inits)...);
}

template <typename... InitTys>
details::dim_vec_kind_init<details::dim_vec_kind::group> group(InitTys&&... inits) {
    return details::dim_vec_kind_init<details::dim_vec_kind::group>(std::forward<InitTys>(inits)...);
}

/// @brief N-dimensional vector. Mostly used to represent memory size.
struct tensor {
    friend class details::dim_vec_kind_init<details::dim_vec_kind::batch>;
    friend class details::dim_vec_kind_init<details::dim_vec_kind::feature>;
    friend class details::dim_vec_kind_init<details::dim_vec_kind::spatial>;
    friend class details::dim_vec_kind_init<details::dim_vec_kind::group>;

    typedef int32_t value_type;  ///< Values type stored in tensor.
    // TODO find the way to prevent direct change of following fields.
    mutable_array_ref<value_type> raw;      ///< Raw representation of all dimensions.
    mutable_array_ref<value_type> batch;    ///< Batch dimensions.
    mutable_array_ref<value_type> feature;  ///< Feature maps.
    mutable_array_ref<value_type> spatial;  ///< Spatial dimensions.
    mutable_array_ref<value_type> group;    ///< Group dimensions.

private:
    value_type _sizes[tensor_dim_max];

public:
    explicit tensor(value_type default_size = 0) :
        raw(_sizes, tensor_dim_max),
        batch(_sizes, tensor_batch_dim_max),
        feature(_sizes + tensor_batch_dim_max, tensor_feature_dim_max),
        spatial(_sizes + tensor_batch_dim_max + tensor_feature_dim_max, tensor_spatial_dim_max),
        group(_sizes + tensor_batch_dim_max + tensor_feature_dim_max + tensor_spatial_dim_max, tensor_group_dim_max) {
        std::fill_n(_sizes, tensor_dim_max, default_size);
    }

    /// @brief Constructs tensor.
    /// @param[in] kind_inits Dimensions defined using dim_vec_kind. If dimension is not provided it is set to 1.
    /// @details Example:
    /*! @code
    *
    tensor my_tensor(batch(2), spatial(5, 6));   // y=6, x=5, b=2, f - not set
    cout << my_tensor.batch[0] << endl;           // 2
    cout << my_tensor.feature[0] << endl;         // 1 - default_size
    cout << "x=" << my_tensor.spatial[0] << endl; // x=5
    cout << "y=" << my_tensor.spatial[1] << endl; // y=6
    *
    * @endcode
    */
    template <typename... KindInitTys,
              typename = typename std::enable_if<
                  meta::all<
                      meta::is_any_of<KindInitTys,
                                      cldnn::details::dim_vec_kind_init<details::dim_vec_kind::batch>,
                                      cldnn::details::dim_vec_kind_init<details::dim_vec_kind::feature>,
                                      cldnn::details::dim_vec_kind_init<details::dim_vec_kind::spatial>,
                                      cldnn::details::dim_vec_kind_init<details::dim_vec_kind::group>>::value...>::value,
                  void>::type>
    explicit tensor(KindInitTys&&... kind_inits)
        : tensor(1) {
        assign_inits(std::forward<KindInitTys>(kind_inits)...);
    }

    /// @brief Constructs @p tensor.
    /// @details Example:
    /*! @code
     *
       tensor my_tensor( 2, 3, 4, 5 );   // b=2, f=3, x=4, y=5
       cout << my_tensor.batch[0] << endl;           // 2
       cout << my_tensor.feature[0] << endl;         // 3
       cout << "x=" << my_tensor.spatial[0] << endl; // x=4
       cout << "y=" << my_tensor.spatial[1] << endl; // y=5
     *
     * @endcode
     */
    tensor(value_type batch_num, value_type feature_num, value_type x, value_type y)
        : tensor(1) {
        _sizes[0] = batch_num;
        _sizes[tensor_batch_dim_max] = feature_num;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max] = x;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 1] = y;
        if (batch_num == 0 && feature_num == 0 && x == 0 && y == 0)
            _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 2] = 0;
    }

    /// @brief Constructs @p tensor.
    /// @details Example:
    /*! @code
    *
    tensor my_tensor( 2, 3, 4, 5, 6 );   // b=2, f=3, x=4, y=5, z=6
    cout << my_tensor.batch[0] << endl;           // 2
    cout << my_tensor.feature[0] << endl;         // 3
    cout << "x=" << my_tensor.spatial[0] << endl; // x=4
    cout << "y=" << my_tensor.spatial[1] << endl; // y=5
    cout << "z=" << my_tensor.spatial[2] << endl; // z=6
    *
    * @endcode
    */
    tensor(value_type batch_num, value_type feature_num, value_type x, value_type y, value_type z)
        : tensor(1) {
        _sizes[0] = batch_num;
        _sizes[tensor_batch_dim_max] = feature_num;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max] = x;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 1] = y;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 2] = z;
    }

    /// @brief Constructs @p tensor.
    /// @details Example:
    /*! @code
    *
    tensor my_tensor( 2, 3, 4, 5, 6, 7 );   // b=2, f=3, x=4, y=5, z=6, w=7
    cout << my_tensor.batch[0] << endl;           // 2
    cout << my_tensor.feature[0] << endl;         // 3
    cout << "x=" << my_tensor.spatial[0] << endl; // x=4
    cout << "y=" << my_tensor.spatial[1] << endl; // y=5
    cout << "z=" << my_tensor.spatial[2] << endl; // z=6
    cout << "w=" << my_tensor.spatial[3] << endl; // w=7
    *
    * @endcode
    */
    tensor(value_type batch_num, value_type feature_num, value_type x, value_type y, value_type z, value_type w)
        : tensor(1) {
        _sizes[0] = batch_num;
        _sizes[tensor_batch_dim_max] = feature_num;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max] = x;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 1] = y;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 2] = z;
        _sizes[tensor_batch_dim_max + tensor_feature_dim_max + 3] = w;
    }

    /// @brief Constructs @p tensor using vector of sizes.
    /// @param[in] sizes dimensions need to be provided in the following order {batch, feature, spatial_x, spatial_y [, spatial_z] }.
    /// @param[in] default_size default_size for tensor dimensions.
    /// @details Example:
    /*! @code
     *
       tensor my_tensor = { 2, 3, 4, 5 };   // b=2, f=3, x=4, y=5
       cout << my_tensor.batch[0] << endl;           // 2
       cout << my_tensor.feature[0] << endl;         // 3
       cout << "x=" << my_tensor.spatial[0] << endl; // x=4
       cout << "y=" << my_tensor.spatial[1] << endl; // y=5
     *
     * @endcode
     */
    explicit tensor(const std::vector<value_type>& sizes, value_type default_size = 1)
        : tensor(default_size) {
        int max_size = std::min(static_cast<int>(sizes.size()), tensor_dim_max);
        for (int i = 0; i < max_size; i++)
            _sizes[i] = sizes[i];
    }

    tensor(format fmt, const std::vector<value_type>& sizes, value_type default_size = 1)
        : tensor(default_size) {
        const auto& in_order = fmt.order();
        const auto& out_order = fmt.internal_order();
        if (in_order.size() != sizes.size())
            throw std::invalid_argument("The count of values passed to initialize tensor does not match passed format.");

        for (size_t out_idx = 0; out_idx < out_order.size(); ++out_idx) {
            auto channel = out_order[out_idx];
            if (channel == '?')
                continue;

            auto in_idx = in_order.find(channel);
            if (in_idx == in_order.npos)
                throw std::runtime_error("Internal order of a format contains channel which does not appear in external order.");

            _sizes[out_idx] = sizes[in_idx];
        }
    }

    /// @brief Copy construction.
    tensor(const tensor& other)
        : tensor(0) {
        std::copy_n(other._sizes, tensor_dim_max, _sizes);
    }

    /// @brief Copy assignment.
    tensor& operator=(const tensor& other) {
        if (this == &other)
            return *this;
        std::copy_n(other._sizes, tensor_dim_max, _sizes);
        return *this;
    }

    friend bool operator==(const tensor& lhs, const tensor& rhs) {
        return lhs.raw.size() == rhs.raw.size() && std::equal(lhs.raw.begin(), lhs.raw.end(), rhs.raw.begin());
    }

    friend bool operator!=(const tensor& lhs, const tensor& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator<(const tensor& lhs, const tensor& rhs) {
        if (lhs.raw.size() != rhs.raw.size())
            return lhs.raw.size() < rhs.raw.size();
        for (size_t i = 0; i < lhs.raw.size(); ++i) {
            if (lhs.raw[i] < rhs.raw[i])
                return true;
            if (rhs.raw[i] < lhs.raw[i])
                return false;
        }

        return false;
    }

    friend std::ostream& operator<<(std::ostream& os, const tensor& tensor) {
        os << tensor.to_string();
        return os;
    }

    size_t hash() const {
        size_t seed = 0;
        seed = hash_range(seed, batch.begin(),      batch.end());
        seed = hash_range(seed, feature.begin(),    feature.end());
        seed = hash_range(seed, spatial.begin(),    spatial.end());
        seed = hash_range(seed, group.begin(),      group.end());
        return seed;
    }

    std::string to_string() const {
        std::stringstream out;
        const char* delim = "";

        out << "[b:";
        for (size_t i = 0; i < batch.size(); ++i) {
            out << delim << batch[i];
            delim = ",";
        }
        delim = "";

        out << ", f:";
        for (size_t i = 0; i < feature.size(); ++i) {
            out << delim << feature[i];
            delim = ",";
        }

        std::vector<std::string> spatial_dim_names = {", x", ", y", ", z", ", w", ", u", ", v"};
        for (size_t i = 0; i < spatial.size(); ++i) {
            out << spatial_dim_names[i] << ":" << spatial[i];
        }

        out << ", g:";
        for (size_t i = 0; i < group.size(); ++i) {
            out << group[i];
        }
        out << "]";

        return out.str();
    }

    /// @brief Returns a tensor with all negated elements.
    tensor negate() const {
        auto result = *this;
        for (size_t i = 0; i < tensor_dim_max; i++) {
            result._sizes[i] = -_sizes[i];
        }
        return result;
    }

    /// @brief Returns a tensor with all elements multilied to @p multiplier.
    tensor mul(value_type multiplier) const {
        auto result = *this;
        for (size_t i = 0; i < tensor_dim_max; i++) {
            result._sizes[i] *= multiplier;
        }
        return result;
    }

    /// @brief Returns a tensor with all elements divided by @p divider.
    tensor div(value_type divider) const {
        auto result = *this;
        for (size_t i = 0; i < tensor_dim_max; i++) {
            result._sizes[i] /= divider;
        }
        return result;
    }

    /// @brief Returns a tensor with all elements added by appropriate elements of @p rhs
    tensor add(const tensor& rhs) const {
        auto result = *this;
        for (size_t i = 0; i < tensor_dim_max; i++) {
            result._sizes[i] += rhs._sizes[i];
        }
        return result;
    }

    /// @brief Returns a tensor with all elements subtracted by appropriate elements of @p rhs
    tensor sub(const tensor& rhs) const {
        return add(rhs.negate());
    }

    /// @brief Assign and add
    tensor& operator+=(const tensor& rhs) {
        for (size_t i = 0; i < tensor_dim_max; i++)
            _sizes[i] += rhs._sizes[i];
        return *this;
    }

    /// @brief Assign and subtract
    tensor& operator-=(const tensor& rhs) {
        for (size_t i = 0; i < tensor_dim_max; i++)
            _sizes[i] -= rhs._sizes[i];
        return *this;
    }

    /// @brief Returns a vector of tensors values, ordered regarding to @p format.
    std::vector<value_type> sizes(cldnn::format fmt) const {
        const auto& output_order = fmt.order();
        const auto& internal_order = fmt.internal_order();
        std::vector<value_type> sizes(output_order.size(), 0);

        for (size_t i = 0; i < sizes.size(); ++i) {
            auto c = output_order[i];
            auto pos = internal_order.find(c);
            if (pos == std::string::npos)
                throw std::domain_error(std::string("Unknown coord type: ") + c);

            sizes[i] = _sizes[pos];
        }

        return sizes;
    }

    /// @brief Returns a vector of tensors values, ordered batch, feature, spatial_x, spatial_y.
    std::vector<value_type> sizes() const {
        std::vector<value_type> sizes(sizeof(_sizes) / sizeof(_sizes[0]), 0);
        for (size_t i = 0; i < sizes.size(); ++i)
            sizes[i] = _sizes[i];
        return sizes;
    }

    /// @brief Returns tensor elements count calculated as multiplication of all elements.
    size_t count() const {
        return std::accumulate(
            raw.begin(),
            raw.end(),
            static_cast<size_t>(1),
            std::multiplies<size_t>());
    }

    /// @brief Returns new tensor based on current but transformed to new @p format.
    /// @param[in] new_fmt Format of new tensor.
    /// @param[in] default_size Default element values for positions not defined by current format.
    /// @details Example:
    /*!
     * @code
       tensor my_tensor({ 2, 3, 4, 5 });
       auto my_sizes = my_tensor.sizes();
       cout << "dims_num=" << my_sizes.size() << endl; // dims_num=2
       cout << "b=" << my_sizes[0] << endl;            // b=2
       cout << "f=" << my_sizes[1] << endl;            // f=3
       cout << "x=" << my_sizes[2] << endl;            // x=5
       cout << "y=" << my_sizes[3] << endl;            // y=4
       auto new_tensor = my_tensor.transform(format::yxfb, 10);
       auto new_sizes = new_tensor.sizes();
       cout << "new_num=" << new_sizes.size() << endl;   // new_num=4
       for(auto dim : new_sizes) cout << " " << dim;     //  5 4 3 2
       cout << endl;
       * @endcode
     */
    tensor transform(cldnn::format new_fmt, value_type default_size) const {
        cldnn::format default_fmt = cldnn::format::bfvuwzyx;
        const auto& val_order = default_fmt.internal_order();
        const auto& new_order = new_fmt.internal_order();
        const std::vector<value_type>& old_sizes = sizes();
        std::vector<value_type> new_sizes(old_sizes.size(), default_size);
        const auto& new_traits = new_fmt.traits();
        static const std::map<char, char> flatten_mapping = {
            { 'v', 'u'},
            { 'u', 'w'},
            { 'w', 'z'},
            { 'z', 'y'}
        };

        for (size_t i = 0; i < default_fmt.order().size(); i++) {
            auto target_dim = val_order[i]; //bfxywzuv
            while (!new_traits.has_dimension(target_dim)) {
                if (flatten_mapping.find(target_dim) != flatten_mapping.end()) {
                    target_dim = flatten_mapping.at(target_dim);
                } else {
                    target_dim = new_fmt.order().back();
                }
            }

            auto new_pos = new_order.find(target_dim);
            if (new_pos != std::string::npos) {
                if (new_sizes[new_pos] == -1) {
                    new_sizes[new_pos] = old_sizes[i];
                } else {
                    new_sizes[new_pos] *= old_sizes[i];
                }
            }
        }

        for (size_t i = 0; i < new_order.size(); i++) {
            auto c = new_order[i]; //bfxywz
            if (c == '?')
                continue;
            if (new_sizes[i] == -1) {
                new_sizes[i] = 1;
            }
        }

        tensor sizes { new_sizes };
        return sizes;
    }

    /// @brief Calculates linear offset for given @p coord within current tensor.
    /// @param coord The coordinate within current tensor.
    size_t get_linear_offset(const tensor& coord, const cldnn::format& fmt) const {
        auto my_sizes = this->sizes(fmt);
        auto adjusted_coords = coord.sizes(fmt);

        // Extend N-dimensional format with B blocked dimensions to (N+B) sizes
        for (const auto& block : fmt.block_sizes()) {
            auto block_axis = block.first;
            auto block_size = block.second;
            auto external_axis = fmt.internal_to_external(block_axis);

            my_sizes.push_back(block_size);
            my_sizes[external_axis] = ceil_div(my_sizes[external_axis], block_size);

            adjusted_coords.push_back(adjusted_coords[external_axis] % block_size);
            adjusted_coords[external_axis] /= block_size;
        }

        if (fmt == cldnn::format::os_is_yx_isa8_osv8_isv4 &&  // TODO Fix offsets calculation for formats below
                   !(is_aligned_to(my_sizes[0], 8)) &&
                   !(is_aligned_to(my_sizes[1], 32))) {
            my_sizes[0] = align_to(my_sizes[0], 8);
            my_sizes[1] = align_to(my_sizes[1], 32);
            adjusted_coords[0] = align_to(adjusted_coords[0], 8);
            adjusted_coords[1] = align_to(adjusted_coords[1], 32);
        } else if (fmt == cldnn::format::os_is_yx_isa8_osv16_isv4 &&
                   !(is_aligned_to(my_sizes[0], 16)) &&
                   !(is_aligned_to(my_sizes[1], 32))) {
            my_sizes[0] = align_to(my_sizes[0], 16);
            my_sizes[1] = align_to(my_sizes[1], 32);
            adjusted_coords[0] = align_to(adjusted_coords[0], 16);
            adjusted_coords[1] = align_to(adjusted_coords[1], 32);
        } else if (fmt == cldnn::format::gs_oi_yxs_gsv4_yxsv4 || fmt == cldnn::format::gs_oi_yxs_gsv16_yxsv4 || fmt == cldnn::format::gs_oi_yxs_gsv32_yxsv4) {
            const auto yxsv = 4;
            const auto flat_xy = adjusted_coords[4] + adjusted_coords[3] * my_sizes[4];

            my_sizes.push_back(yxsv);
            my_sizes[4] = ceil_div(my_sizes[3] * my_sizes[4], yxsv);
            my_sizes[3] = 1;

            adjusted_coords.push_back(flat_xy % yxsv);
            adjusted_coords[4] = flat_xy / yxsv;
            adjusted_coords[3] = 0;
        } else if (fmt == cldnn::format::os_iyx_osv32__ai32 && !is_aligned_to(my_sizes[1], 32)) {
            my_sizes[1] = align_to(my_sizes[1], 32);
        } else if ((fmt == cldnn::format::iy_xs_os_xsv2_osv8__ao32 || fmt == cldnn::format::iy_xs_os_xsv2_osv16__ao32) && !is_aligned_to(my_sizes[3], 32)) {
            my_sizes[3] = align_to(my_sizes[3], 32);
        } else if (fmt == cldnn::format::i_yxs_os_yxsv2_osv16 || fmt == cldnn::format::gi_yxs_os_yxsv2_osv16) {
            const auto yxsv = 2;
            auto flat_xy = adjusted_coords[2] + adjusted_coords[1] * my_sizes[2];

            my_sizes.insert(std::prev(my_sizes.end()), yxsv);
            my_sizes[2] = ceil_div(my_sizes[1] * my_sizes[2], yxsv);
            my_sizes[1] = 1;

            adjusted_coords.insert(std::prev(adjusted_coords.end()), flat_xy % yxsv);
            adjusted_coords[2] = flat_xy / yxsv;
            adjusted_coords[1] = 0;
        } else if ((fmt == cldnn::format::giy_xs_os_xsv2_osv8__ao32 || fmt == cldnn::format::giy_xs_os_xsv2_osv16__ao32) && !is_aligned_to(my_sizes[3], 32)) {
            my_sizes[4] = align_to(my_sizes[4], 32);
        }

        assert(my_sizes.size() == adjusted_coords.size());
        assert(adjusted_coords.size() > 0);

        size_t offset = adjusted_coords[0];
        for (size_t i = 1; i < adjusted_coords.size(); i++) {
            offset = offset * my_sizes[i] + adjusted_coords[i];
        }
        return offset;
    }

    /// @brief Returns partial shape of the requested rank
    /// @param rank The requested rank of partial shape
    /// @param format_dims Number of actual dimensions for layout's format
    ov::PartialShape get_partial_shape(size_t rank, size_t format_dims) const {
        ov::Shape shape;
        size_t i = 0;
        for (; i < std::min(static_cast<size_t>(2), rank); ++i) {
            shape.push_back(_sizes[i]);
        }
        for (; i < rank; ++i) {
            shape.push_back(_sizes[format_dims - (i - 2) - 1]);
        }
        return ov::PartialShape(shape);
    }

    /// @brief Returns a tensor containing values maximum from @p lhs and @p rhs.
    static tensor max(tensor const& lhs, tensor const& rhs) {
        auto ret = lhs;
        for (size_t i = 0; i < tensor_dim_max; ++i)
            ret._sizes[i] = std::max(ret._sizes[i], rhs._sizes[i]);

        return ret;
    }

    /// @brief Returns a tensor containing values minimum from @p lhs and @p rhs.
    static tensor min(tensor const& lhs, tensor const& rhs) {
        auto ret = lhs;
        for (size_t i = 0; i < tensor_dim_max; ++i)
            ret._sizes[i] = std::min(ret._sizes[i], rhs._sizes[i]);

        return ret;
    }

private:
    /// @brief Helper functions for tensor constructor using dim_vec_kinds
    template <typename KindInitT>
    void assign_inits(KindInitT&& init) {
        init.init_tensor_values(*this);
    }

    template <typename KindInitT, typename... KindInitTys>
    void assign_inits(KindInitT&& init, KindInitTys&&... kind_inits) {
        init.init_tensor_values(*this);
        assign_inits(std::forward<KindInitTys>(kind_inits)...);
    }
};

#define TensorValue(val) static_cast<cldnn::tensor::value_type>(val)

template <details::dim_vec_kind Kind>
inline void details::dim_vec_kind_init<Kind>::init_tensor_values(cldnn::tensor& t) {
    for (size_t i = _dimOffset; i < (size_t)(_dimOffset + _dimSize); i++)
        t._sizes[i] = _sizes[i - _dimOffset];
}

/// @brief Adds two @p tensors
inline tensor operator+(const tensor& lhs, const tensor& rhs) { return lhs.add(rhs); }
/// @brief Subtracts two @p tensors
inline tensor operator-(const tensor& lhs, const tensor& rhs) { return lhs.sub(rhs); }
/// @brief Multiplies a @p tensor to a @p scalar
inline tensor operator*(const tensor& lhs, tensor::value_type rhs) { return lhs.mul(rhs); }
/// @brief Divides a @p tensor by a @p scalar
inline tensor operator/(const tensor& lhs, tensor::value_type rhs) { return lhs.div(rhs); }

/// @}
/// @}
}  // namespace cldnn
