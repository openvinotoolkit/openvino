// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstring>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/shared_buffer.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Class for constants.
            class NGRAPH_API Constant : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Constant", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Constant() = default;

                /// \brief Initialize a constant from tensor
                /// \param tensor The tensor with data
                Constant(const std::shared_ptr<runtime::Tensor>& tensor);

                /// \brief Constructs a tensor constant.
                ///
                /// \param type The element type of the tensor constant.
                /// \param shape The shape of the tensor constant.
                /// \param values A vector of literals for initializing the tensor constant. The
                ///               size of values must match the size of the shape.
                template <typename T>
                Constant(const element::Type& type,
                         const Shape& shape,
                         const std::vector<T>& values)
                    : Constant(type, shape)
                {
                    NODE_VALIDATION_CHECK(
                        this,
                        values.size() == 1 || values.size() == shape_size(m_shape),
                        "Did not get the expected number of literals for a constant of shape ",
                        m_shape,
                        " (got ",
                        values.size(),
                        ", expected ",
                        (shape_size(m_shape) == 1 ? "" : "1 or "),
                        shape_size(m_shape),
                        ").");

                    if (values.size() == 1)
                    {
                        fill_data(type, values.front());
                    }
                    else
                    {
                        write_values(values);
                    }
                    m_all_elements_bitwise_identical = are_all_data_elements_bitwise_identical();
                }

                /// \brief Create uninitialized constant
                Constant(const element::Type& type, const Shape& shape);
                /// \brief Constructs a uniform tensor constant.
                ///
                /// \param type The element type of the tensor constant.
                /// \param shape The shape of the tensor constant.
                /// \param value A scalar for initializing the uniform tensor constant. The
                ///               value is broadcast to the specified shape.
                template <class T,
                          class = typename std::enable_if<std::is_fundamental<T>::value>::type>
                Constant(const element::Type& type, const Shape& shape, T value)
                    : Constant(type, shape)
                {
                    fill_data(type, value);
                    m_all_elements_bitwise_identical = true;
                }

                template <typename T>
                void fill_data(const element::Type& type, T value)
                {
                    using Type_t = element::Type_t;
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
                    switch (type)
                    {
                    case Type_t::boolean: fill_data<Type_t::boolean>(value); break;
                    case Type_t::bf16: fill_data<Type_t::bf16>(value); break;
                    case Type_t::f16: fill_data<Type_t::f16>(value); break;
                    case Type_t::f32: fill_data<Type_t::f32>(value); break;
                    case Type_t::f64: fill_data<Type_t::f64>(value); break;
                    case Type_t::i4: fill_data<Type_t::i4>(value); break;
                    case Type_t::i8: fill_data<Type_t::i8>(value); break;
                    case Type_t::i16: fill_data<Type_t::i16>(value); break;
                    case Type_t::i32: fill_data<Type_t::i32>(value); break;
                    case Type_t::i64: fill_data<Type_t::i64>(value); break;
                    case Type_t::u1: fill_data<Type_t::u1>(value); break;
                    case Type_t::u4: fill_data<Type_t::u4>(value); break;
                    case Type_t::u8: fill_data<Type_t::u8>(value); break;
                    case Type_t::u16: fill_data<Type_t::u16>(value); break;
                    case Type_t::u32: fill_data<Type_t::u32>(value); break;
                    case Type_t::u64: fill_data<Type_t::u64>(value); break;
                    case Type_t::undefined:
                    case Type_t::dynamic: throw std::runtime_error("unsupported type");
                    }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
                }

                /// \brief Constructs a tensor constant
                ///        This constructor is mainly to support deserialization of constants.
                ///
                /// \param type The element type of the tensor constant.
                /// \param shape The shape of the tensor constant.
                /// \param values A list of string values to use as the constant data.
                Constant(const element::Type& type,
                         const Shape& shape,
                         const std::vector<std::string>& values);

                /// \brief Constructs a tensor constant with the supplied data
                ///
                /// \param type The element type of the tensor constant.
                /// \param shape The shape of the tensor constant.
                /// \param data A void* to constant data.
                Constant(const element::Type& type, const Shape& shape, const void* data);

                /// \brief Constructs a tensor constant with the supplied data
                ///
                /// \param type The element type of the tensor constant.
                /// \param shape The shape of the tensor constant.
                /// \param data A pointer to pre-allocated shared data.
                template <typename T>
                Constant(const element::Type& type,
                         const Shape& shape,
                         std::shared_ptr<runtime::SharedBuffer<T>> data)
                    : m_element_type(type)
                    , m_shape(shape)
                {
                    m_data = data;
                    constructor_validate_and_infer_types();
                }

                Constant(const Constant& other);
                Constant& operator=(const Constant&) = delete;

                virtual ~Constant() override;

                void validate_and_infer_types() override
                {
                    infer_element_type();
                    set_output_type(0, m_element_type, m_shape);
                }

                bool visit_attributes(AttributeVisitor& visitor) override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
                bool evaluate_lower(const HostTensorVector& outputs) const override;
                bool evaluate_upper(const HostTensorVector& outputs) const override;

                // Don't constant fold a constant; it would make a copy
                bool constant_fold(OutputVector& outputs, const OutputVector& inputs) override
                {
                    return false;
                }

                /// \brief Returns the value of the constant node as a Shape object
                ///        Can only be used on element::i64 nodes and interprets
                ///        negative values as zeros.
                Shape get_shape_val() const;
                /// \brief Returns the value of the constant node as a Strides
                ///        object
                ///        Can only be used on element::i64 nodes and interprets
                ///        negative values as zeros.
                Strides get_strides_val() const;
                /// \brief Returns the value of the constant node as a Coordinate
                ///        object
                ///        Can only be used on element::i64 nodes and interprets
                ///        negative values as zeros.
                Coordinate get_coordinate_val() const;
                /// \brief Returns the value of the constant node as a
                ///        CoordinateDiff object
                ///        Can only be used on element::i64 nodes.
                CoordinateDiff get_coordinate_diff_val() const;
                /// \brief Returns the value of the constant node as an AxisVector
                ///        object
                ///        Can only be used on element::i64 nodes and interprets
                ///        negative values as zeros.
                AxisVector get_axis_vector_val() const;
                /// \brief Returns the value of the constant node as an AxisSet
                ///        object
                ///        Can only be used on element::i64 nodes and interprets
                ///        negative values as zeros.
                ///        Repeated values are allowed.
                AxisSet get_axis_set_val() const;

                /// \brief Update Constant shape. New shape size must equal to the data elements
                /// count
                ///
                /// \param shape The shape of the tensor constant.
                void set_data_shape(const Shape& shape);

                /// \brief Wrapper around constructing a shared_ptr of a Constant
                ///
                /// \param type The element type of the tensor constant.
                /// \param shape The shape of the tensor constant.
                /// \param values A vector of values to use as the constant data.
                template <typename T>
                static std::shared_ptr<Constant> create(const element::Type& type,
                                                        const Shape& shape,
                                                        const std::vector<T>& values)
                {
                    return std::make_shared<Constant>(type, shape, values);
                }

                /// \brief Wrapper around constructing a shared_ptr of a Constant
                ///
                /// \param type The element type of the tensor constant.
                /// \param shape The shape of the tensor constant.
                /// \param values An initializer_list of values to use as the constant data.
                template <typename T>
                static std::shared_ptr<Constant> create(const element::Type& type,
                                                        const Shape& shape,
                                                        std::initializer_list<T> values)
                {
                    return std::make_shared<Constant>(type, shape, std::vector<T>{values});
                }

                /// \brief Wrapper around constructing a shared_ptr of a Constant
                ///
                /// \param type The element type of the tensor constant.
                /// \param shape The shape of the tensor constant.
                /// \param memory An continues memory chunk which contains the constant data.
                static std::shared_ptr<Constant>
                    create(const element::Type& type, const Shape& shape, const void* memory)
                {
                    return std::make_shared<Constant>(type, shape, memory);
                }

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The initialization literals for the tensor constant.
                std::vector<std::string> get_value_strings() const;

                template <typename T>
                std::vector<T> get_vector() const
                {
                    const T* p = get_data_ptr<T>();
                    if (p == nullptr)
                        throw std::runtime_error("Cannot create vector! Buffer is not allocated.");
                    return std::vector<T>(p, p + shape_size(m_shape));
                }

                /// \brief Return the Constant's value as a vector cast to type T
                ///
                /// \tparam T  Type to which data vector's entries will be cast.
                /// \return    Constant's data vector.
                template <typename T>
                std::vector<T> cast_vector() const
                {
                    auto source_type = get_element_type();
                    std::vector<T> rc;
                    using Type_t = element::Type_t;
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)
#endif
                    switch (source_type)
                    {
                    case Type_t::boolean: cast_vector<Type_t::boolean>(rc); break;
                    case Type_t::bf16: cast_vector<Type_t::bf16>(rc); break;
                    case Type_t::f16: cast_vector<Type_t::f16>(rc); break;
                    case Type_t::f32: cast_vector<Type_t::f32>(rc); break;
                    case Type_t::f64: cast_vector<Type_t::f64>(rc); break;
                    case Type_t::i4: cast_vector<Type_t::i4>(rc); break;
                    case Type_t::i8: cast_vector<Type_t::i8>(rc); break;
                    case Type_t::i16: cast_vector<Type_t::i16>(rc); break;
                    case Type_t::i32: cast_vector<Type_t::i32>(rc); break;
                    case Type_t::i64: cast_vector<Type_t::i64>(rc); break;
                    case Type_t::u1: cast_vector<Type_t::u1>(rc); break;
                    case Type_t::u4: cast_vector<Type_t::u4>(rc); break;
                    case Type_t::u8: cast_vector<Type_t::u8>(rc); break;
                    case Type_t::u16: cast_vector<Type_t::u16>(rc); break;
                    case Type_t::u32: cast_vector<Type_t::u32>(rc); break;
                    case Type_t::u64: cast_vector<Type_t::u64>(rc); break;
                    default: throw std::runtime_error("unsupported type");
                    }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
                    return rc;
                }

                const void* get_data_ptr() const { return (m_data ? m_data->get_ptr() : nullptr); }
                template <typename T>
                const T* get_data_ptr() const
                {
                    if (sizeof(T) > m_element_type.size() && shape_size(m_shape) > 0)
                    {
                        throw ngraph_error("Buffer over-read");
                    }

                    return static_cast<const T*>(get_data_ptr());
                }

                template <element::Type_t ET>
                const typename element_type_traits<ET>::value_type* get_data_ptr() const
                {
                    NGRAPH_CHECK(ET == get_element_type(),
                                 "get_data_ptr() called for incorrect element type.");
                    return static_cast<const typename element_type_traits<ET>::value_type*>(
                        get_data_ptr());
                }

                bool get_all_data_elements_bitwise_identical() const
                {
                    return m_all_elements_bitwise_identical;
                }
                std::string convert_value_to_string(size_t index) const;

                /**
                 * \brief Allows to avoid buffer allocation on the visit_attributes call
                 */
                void alloc_buffer_on_visit_attributes(bool val)
                {
                    m_alloc_buffer_on_visit_attributes = val;
                }

            private:
                template <element::Type_t Type,
                          typename StorageDataType = fundamental_type_for<Type>,
                          typename std::enable_if<Type != element::Type_t::u1 &&
                                                      Type != element::Type_t::u4 &&
                                                      Type != element::Type_t::i4,
                                                  bool>::type = true>
                StorageDataType get_element_value(size_t index) const
                {
                    return get_data_ptr<Type>()[index];
                }

                template <element::Type_t Type,
                          typename StorageDataType = fundamental_type_for<Type>,
                          typename std::enable_if<Type == element::Type_t::u1, bool>::type = true>
                StorageDataType get_element_value(size_t index) const
                {
                    return (get_data_ptr<uint8_t>()[index / 8] >> (7 - (index % 8))) & 1;
                }

                template <element::Type_t Type,
                          typename StorageDataType = fundamental_type_for<Type>,
                          typename std::enable_if<Type == element::Type_t::u4, bool>::type = true>
                StorageDataType get_element_value(size_t index) const
                {
                    return (get_data_ptr<uint8_t>()[index / 2] >> (index % 2 ? 0 : 4)) & 0x0F;
                }

                template <element::Type_t Type,
                          typename StorageDataType = fundamental_type_for<Type>,
                          typename std::enable_if<Type == element::Type_t::i4, bool>::type = true>
                StorageDataType get_element_value(size_t index) const
                {
                    const uint8_t i4data =
                        (get_data_ptr<uint8_t>()[index / 2] >> (index % 2 ? 0 : 4)) & 0x0F;
                    const bool is_negative_number = (i4data >> 3) & 0b1;
                    const int8_t data = is_negative_number ? i4data | 0xF0 : i4data;
                    return data;
                }

                template <element::Type_t Type,
                          typename OUT_T,
                          typename std::enable_if<Type != element::Type_t::u1 &&
                                                      Type != element::Type_t::u4 &&
                                                      Type != element::Type_t::i4,
                                                  bool>::type = true>
                void cast_vector(std::vector<OUT_T>& output_vector) const
                {
                    // this function is workaround for waring during windows building
                    // build complains for vector creation based on iterators
                    // which point on different type than destination vector::value_type
                    using IN_T = fundamental_type_for<Type>;
                    auto source_vector = get_vector<IN_T>();
                    output_vector.reserve(source_vector.size());

                    std::transform(source_vector.begin(),
                                   source_vector.end(),
                                   std::back_inserter(output_vector),
                                   [](IN_T c) { return static_cast<OUT_T>(c); });
                }

                template <element::Type_t Type,
                          typename OUT_T,
                          typename std::enable_if<Type == element::Type_t::u1, bool>::type = true>
                void cast_vector(std::vector<OUT_T>& output) const
                {
                    using IN_T = fundamental_type_for<Type>;
                    const auto element_number = shape_size(m_shape);
                    const auto source_begin = get_data_ptr<uint8_t>();
                    const auto source_end = std::next(source_begin, (element_number + 7) / 8);
                    const auto round_element_no = element_number % 8
                                                      ? element_number - element_number % 8 + 8
                                                      : element_number;
                    output.reserve(round_element_no); // adds 7 more elements here?
                    std::for_each(source_begin, source_end, [&](IN_T c) {
                        for (const auto i : {7, 6, 5, 4, 3, 2, 1, 0})
                        {
                            const uint8_t data = (c >> i) & 0x01;
                            output.push_back(data);
                        }
                    });
                    output.resize(element_number);
                }

                template <element::Type_t Type,
                          typename OUT_T,
                          typename std::enable_if<Type == element::Type_t::u4, bool>::type = true>
                void cast_vector(std::vector<OUT_T>& output) const
                {
                    using IN_T = fundamental_type_for<Type>;
                    const auto element_number = shape_size(m_shape);
                    const auto source_begin = get_data_ptr<uint8_t>();
                    const auto source_end = std::next(source_begin, (element_number + 1) / 2);
                    const auto round_element_no =
                        element_number % 2 ? element_number + 1 : element_number;
                    output.reserve(round_element_no); // adds 1 more elements here?
                    std::for_each(source_begin, source_end, [&](IN_T c) {
                        for (const auto i : {4, 0})
                        {
                            const uint8_t data = (c >> i) & 0x0F;
                            output.push_back(data);
                        }
                    });
                    output.resize(element_number);
                }
                template <element::Type_t Type,
                          typename OUT_T,
                          typename std::enable_if<Type == element::Type_t::i4, bool>::type = true>
                void cast_vector(std::vector<OUT_T>& output) const
                {
                    using IN_T = fundamental_type_for<Type>;
                    const auto element_number = shape_size(m_shape);
                    const auto source_begin = get_data_ptr<uint8_t>();
                    const auto source_end = std::next(source_begin, (element_number + 1) / 2);
                    const auto round_element_no =
                        element_number % 2 ? element_number + 1 : element_number;
                    output.reserve(round_element_no); // adds 1 more elements here?
                    std::for_each(source_begin, source_end, [&](IN_T c) {
                        for (const auto i : {4, 0})
                        {
                            const uint8_t i4data = (c >> i) & 0x0F;
                            const bool is_negative_number = (i4data >> 3) & 0b1;
                            const int8_t data = is_negative_number ? i4data | 0xF0 : i4data;
                            output.push_back(data);
                        }
                    });
                    output.resize(element_number);
                }

                template <element::Type_t Type,
                          typename T,
                          typename StorageDataType = fundamental_type_for<Type>,
                          typename std::enable_if<Type != element::Type_t::u1 &&
                                                      Type != element::Type_t::u4 &&
                                                      Type != element::Type_t::i4,
                                                  bool>::type = true>
                void fill_data(const T& value)
                {
                    const auto size = shape_size(m_shape);
                    const auto v = static_cast<StorageDataType>(value);
                    std::fill_n(get_data_ptr_nc<Type>(), size, v);
                }

                template <element::Type_t Type,
                          typename T,
                          typename StorageDataType = fundamental_type_for<Type>,
                          typename std::enable_if<Type == element::Type_t::u1, bool>::type = true>
                void fill_data(const T& value)
                {
                    const StorageDataType v = value ? 0xFF : 0x00;
                    std::fill_n(get_data_ptr_nc<Type>(), mem_size(), v);
                }

                template <element::Type_t Type,
                          typename T,
                          typename StorageDataType = fundamental_type_for<Type>,
                          typename std::enable_if<Type == element::Type_t::u4 ||
                                                      Type == element::Type_t::i4,
                                                  bool>::type = true>
                void fill_data(const T& value)
                {
                    uint8_t v = value_in_range<Type>(value);
                    v &= 0x0F;
                    v += v << 4;
                    std::fill_n(get_data_ptr_nc<Type>(), mem_size(), v);
                }

                void allocate_buffer();

                void* get_data_ptr_nc() { return (m_data ? m_data->get_ptr() : nullptr); }

                template <element::Type_t ET>
                typename element_type_traits<ET>::value_type* get_data_ptr_nc()
                {
                    NGRAPH_CHECK(ET == get_element_type(),
                                 "get_data_ptr_nc() called for incorrect element type.");
                    return static_cast<typename element_type_traits<ET>::value_type*>(
                        get_data_ptr_nc());
                }

                Constant(const OutputVector& args)
                    : Op(args)
                    , m_shape({})
                {
                }

                virtual void infer_element_type() {}
                template <typename T>
                void write_values(const std::vector<T>& values)
                {
                    write_to_buffer(values);
                }

                template <element::Type_t Type,
                          typename T,
                          typename StorageDataType = fundamental_type_for<Type>,
                          typename std::enable_if<Type != element::Type_t::u1 &&
                                                      Type != element::Type_t::u4 &&
                                                      Type != element::Type_t::i4,
                                                  bool>::type = true>
                void write_buffer(const std::vector<T>& source)
                {
                    auto p = get_data_ptr_nc<Type>();
                    for (size_t i = 0; i < source.size(); i++)
                    {
                        p[i] = static_cast<StorageDataType>(source[i]);
                    }
                }

                template <element::Type_t Type,
                          typename T,
                          typename StorageDataType = fundamental_type_for<Type>,
                          typename std::enable_if<Type == element::Type_t::u4 ||
                                                      Type == element::Type_t::i4,
                                                  bool>::type = true>
                void write_buffer(const std::vector<T>& source)
                {
                    auto p = get_data_ptr_nc<Type>();
                    size_t i = 0;
                    for (; i < source.size() / 2; i++)
                    {
                        const auto v1 = value_in_range<Type>(source[i * 2]) & 0x0F;
                        const auto v2 = value_in_range<Type>(source[i * 2 + 1]) & 0x0F;
                        const auto v = (v1 << 4) | v2;
                        p[i] = static_cast<StorageDataType>(v);
                    }
                    if (source.size() % 2)
                    {
                        const auto v1 = value_in_range<Type>(source[i * 2]) & 0x0F;
                        const auto v = v1 << 4;
                        p[i] = static_cast<StorageDataType>(v);
                    }
                }

                template <element::Type_t Type,
                          typename T,
                          typename StorageDataType = fundamental_type_for<Type>,
                          typename std::enable_if<Type == element::Type_t::u1, bool>::type = true>
                void write_buffer(const std::vector<T>& source)
                {
                    auto p = get_data_ptr_nc<Type>();
                    size_t i = 0;
                    for (; i < source.size() / 8; i++)
                    {
                        uint8_t v{};
                        for (int j = 0; j != 8; j++)
                        {
                            const uint8_t b = source[i * 8 + j] ? 0x01 << (7 - j) : 0;
                            v |= b;
                        }
                        p[i] = static_cast<StorageDataType>(v);
                    }
                    uint8_t v{};
                    for (unsigned j = 0; j != source.size() % 8; j++)
                    {
                        const uint8_t b = source[i * 8 + j] ? 0x01 << (7 - j) : 0;
                        v |= b;
                    }
                    p[i] = static_cast<StorageDataType>(v);
                }

                template <typename T>
                void write_to_buffer(const std::vector<T>& source)
                {
                    const auto& target_type = m_element_type;
                    size_t target_element_count = shape_size(m_shape);
                    if (source.size() != target_element_count)
                    {
                        throw std::runtime_error("Constant initializer does not match shape");
                    }
                    using Type_t = element::Type_t;
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
                    switch (target_type)
                    {
                    case Type_t::boolean: write_buffer<Type_t::boolean>(source); break;
                    case Type_t::bf16: write_buffer<Type_t::bf16>(source); break;
                    case Type_t::f16: write_buffer<Type_t::f16>(source); break;
                    case Type_t::f32: write_buffer<Type_t::f32>(source); break;
                    case Type_t::f64: write_buffer<Type_t::f64>(source); break;
                    case Type_t::i4: write_buffer<Type_t::i4>(source); break;
                    case Type_t::i8: write_buffer<Type_t::i8>(source); break;
                    case Type_t::i16: write_buffer<Type_t::i16>(source); break;
                    case Type_t::i32: write_buffer<Type_t::i32>(source); break;
                    case Type_t::i64: write_buffer<Type_t::i64>(source); break;
                    case Type_t::u1: write_buffer<Type_t::u1>(source); break;
                    case Type_t::u4: write_buffer<Type_t::u4>(source); break;
                    case Type_t::u8: write_buffer<Type_t::u8>(source); break;
                    case Type_t::u16: write_buffer<Type_t::u16>(source); break;
                    case Type_t::u32: write_buffer<Type_t::u32>(source); break;
                    case Type_t::u64: write_buffer<Type_t::u64>(source); break;
                    case element::Type_t::undefined:
                    case element::Type_t::dynamic: throw std::runtime_error("unsupported type");
                    }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
                }
                template <
                    ngraph::element::Type_t Type,
                    typename ValueT,
                    typename std::enable_if<Type == ngraph::element::Type_t::u4, bool>::type = true>
                static ngraph::fundamental_type_for<Type> value_in_range(const ValueT& value)
                {
                    const auto result = ngraph::fundamental_type_for<Type>(value);
                    NGRAPH_CHECK(0 <= result && result <= 15,
                                 "assigned value out of range u4 values");
                    return result;
                }

                template <
                    ngraph::element::Type_t Type,
                    typename ValueT,
                    typename std::enable_if<Type == ngraph::element::Type_t::i4, bool>::type = true>
                static ngraph::fundamental_type_for<Type> value_in_range(const ValueT& value)
                {
                    const auto result = ngraph::fundamental_type_for<Type>(value);
                    NGRAPH_CHECK(-8 <= result && result <= 7,
                                 "assigned value out of range i4 values");
                    return result;
                }

                bool are_all_data_elements_bitwise_identical() const;
                static constexpr size_t host_alignment() { return 64; }

                size_t mem_size() const
                {
                    const bool bitwidth_less_than_byte = m_element_type.bitwidth() < 8;
                    if (bitwidth_less_than_byte)
                    {
                        const auto size = shape_size(m_shape);
                        const auto bitwidth = size * m_element_type.bitwidth();
                        // for rounding by `(bitwidth + 7) / 8` will work for
                        // `bitwidth < numeric_limits<size_t>::max() - 7`
                        return bitwidth / 8 + (bitwidth % 8 ? 1 : 0);
                    }
                    return shape_size(m_shape) * m_element_type.size();
                }

                element::Type m_element_type;
                Shape m_shape{};
                std::shared_ptr<runtime::AlignedBuffer> m_data;
                bool m_all_elements_bitwise_identical;
                bool m_alloc_buffer_on_visit_attributes = true;
            };
        } // namespace v0
        using v0::Constant;
    } // namespace op
} // namespace ngraph
