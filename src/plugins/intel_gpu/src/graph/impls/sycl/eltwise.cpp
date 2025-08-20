// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"
#include "eltwise_inst.h"
#include "intel_gpu/primitives/reorder.hpp"
#include "sycl/sycl_stream.hpp"
#include "sycl/sycl_memory.hpp"
#include "openvino/core/type/element_type.hpp"
#include "primitive_sycl_base.h"
#include "registry/implementation_map.hpp"

#include "impls/ocl/kernel_selector_helper.h"

#include "sycl/sycl.hpp"

#include <memory>
#include <variant>

namespace {

/**
 * @brief Base class for eltwise operation function objects using CRTP (Curiously Recurring Template Pattern)
 *
 * This base class provides a common interface for all eltwise operations while allowing
 * compile-time polymorphism. Each derived class must implement the 'apply' method.
 * Supports heterogeneous input types and flexible output types.
 *
 * @tparam Derived The derived class implementing the specific operation
 */
template<typename Derived>
struct EltwiseOpBase {
    /**
     * @brief Function call operator that delegates to the derived class's apply method
     * @tparam T1 The data type for the first operand
     * @tparam T2 The data type for the second operand
     * @param a First operand
     * @param b Second operand
     * @return Result of the operation (type determined by derived class implementation)
     */
    template<typename T1, typename T2>
    constexpr auto operator()(const T1& a, const T2& b) const noexcept {
        return static_cast<const Derived*>(this)->apply(a, b);
    }
};

// ============================================================================
// Arithmetic Operations
// ============================================================================

/**
 * @brief Addition operation (a + b)
 * Commutative operation that returns the sum of two operands
 */
struct AddOp : public EltwiseOpBase<AddOp> {
    template<typename T1, typename T2>
    constexpr auto apply(const T1& a, const T2& b) const noexcept {
        return a + b;
    }
};

/**
 * @brief Subtraction operation (a - b)
 * Non-commutative operation that returns the difference of two operands
 */
struct SubOp : public EltwiseOpBase<SubOp> {
    template<typename T1, typename T2>
    constexpr auto apply(const T1& a, const T2& b) const noexcept {
        return a - b;
    }
};

/**
 * @brief Multiplication operation (a * b)
 * Commutative operation that returns the product of two operands
 */
struct MulOp : public EltwiseOpBase<MulOp> {
    template<typename T1, typename T2>
    constexpr auto apply(const T1& a, const T2& b) const noexcept {
        return a * b;
    }
};

/**
 * @brief Division operation (a / b)
 * Non-commutative operation that returns the quotient of two operands
 */
struct DivOp : public EltwiseOpBase<DivOp> {
    template<typename T1, typename T2>
    constexpr auto apply(const T1& a, const T2& b) const noexcept {
        return a / b;
    }
};

/**
 * @brief Power operation (pow(a, b))
 * Non-commutative operation that returns a raised to the power of b
 */
struct PowOp : public EltwiseOpBase<PowOp> {
    template<typename T1, typename T2>
    inline auto apply(const T1& a, const T2& b) const noexcept {
        return ::sycl::pow(a, b);
    }
};

// ============================================================================
// Min/Max Operations
// ============================================================================

/**
 * @brief Maximum operation (max(a, b))
 * Commutative operation that returns the maximum of two operands
 */
struct MaxOp : public EltwiseOpBase<MaxOp> {
    template<typename T1, typename T2>
    inline auto apply(const T1& a, const T2& b) const {
        return ::sycl::max(a, b);
    }
};

/**
 * @brief Minimum operation (min(a, b))
 * Commutative operation that returns the minimum of two operands
 */
struct MinOp : public EltwiseOpBase<MinOp> {
    template<typename T1, typename T2>
    inline auto apply(const T1& a, const T2& b) const {
        return ::sycl::min(a, b);
    }
};

// ============================================================================
// Special Operations
// ============================================================================

/**
 * @brief Squared difference operation ((a - b)^2)
 * Commutative operation that returns the square of the difference
 */
struct SquaredDiffOp : public EltwiseOpBase<SquaredDiffOp> {
    template<typename T1, typename T2>
    inline auto apply(const T1& a, const T2& b) const {
        const auto diff = a - b;
        return diff * diff;
    }
};

// ============================================================================
// Comparison Operations (return 0 or 1)
// ============================================================================

/**
 * @brief Equality comparison (a == b)
 * Commutative boolean operation that returns 1 if equal, 0 otherwise
 */
struct EqOp : public EltwiseOpBase<EqOp> {
    template<typename T1, typename T2>
    inline auto apply(const T1& a, const T2& b) const {
        return (a == b) ? 1 : 0;
    }
};

/**
 * @brief Not equal comparison (a != b)
 * Commutative boolean operation that returns 1 if not equal, 0 otherwise
 */
struct NeOp : public EltwiseOpBase<NeOp> {
    template<typename T1, typename T2>
    inline auto apply(const T1& a, const T2& b) const {
        return (a != b) ? 1 : 0;
    }
};

/**
 * @brief Less than comparison (a < b)
 * Non-commutative boolean operation that returns 1 if a < b, 0 otherwise
 */
struct LtOp : public EltwiseOpBase<LtOp> {
    template<typename T1, typename T2>
    inline auto apply(const T1& a, const T2& b) const {
        return (a < b) ? 1 : 0;
    }
};

/**
 * @brief Less than or equal comparison (a <= b)
 * Non-commutative boolean operation that returns 1 if a <= b, 0 otherwise
 */
struct LeOp : public EltwiseOpBase<LeOp> {
    template<typename T1, typename T2>
    inline auto apply(const T1& a, const T2& b) const {
        return (a <= b) ? 1 : 0;
    }
};

/**
 * @brief Greater than comparison (a > b)
 * Non-commutative boolean operation that returns 1 if a > b, 0 otherwise
 */
struct GtOp : public EltwiseOpBase<GtOp> {
    template<typename T1, typename T2>
    inline auto apply(const T1& a, const T2& b) const {
        return (a > b) ? 1 : 0;
    }
};

/**
 * @brief Greater than or equal comparison (a >= b)
 * Non-commutative boolean operation that returns 1 if a >= b, 0 otherwise
 */
struct GeOp : public EltwiseOpBase<GeOp> {
    template<typename T1, typename T2>
    inline auto apply(const T1& a, const T2& b) const {
        return (a >= b) ? 1 : 0;
    }
};

template <typename InPtrType, typename OutPtrType, typename OpFunc>
inline void eltwise_kernel(::sycl::item<1> index, InPtrType in0, InPtrType in1, OutPtrType out, size_t in1_stride, OpFunc op) {
    auto val0 = in0[index];
    auto val1 = in1[index * in1_stride];
    out[index] = op(val0, val1);
}

// for sycl::buffer
template<typename DType, int Dim, typename Allocator, typename OpFunc>
::sycl::event run_eltwise(::sycl::queue& queue, std::vector<::sycl::event> const& events,
                          ::sycl::buffer<DType, Dim, Allocator>& in0,
                          ::sycl::buffer<DType, Dim, Allocator>& in1,
                          ::sycl::buffer<DType, Dim, Allocator>& out,
                          const ov::Shape& in0_shape, const ov::Shape& in1_shape, const ov::Shape& out_shape,
                          OpFunc op) {
    auto num_elements = ov::shape_size(out_shape);
    size_t in1_stride = ov::shape_size(in1_shape) > 1 ? 1 : 0;

    return queue.submit([&](::sycl::handler& cgh) {
        cgh.depends_on(events);

        auto in0_acc = in0.template get_access<::sycl::access::mode::read>(cgh);
        auto in1_acc = in1.template get_access<::sycl::access::mode::read>(cgh);
        auto out_acc = out.template get_access<::sycl::access::mode::write>(cgh);

        cgh.parallel_for(::sycl::range<1>(num_elements), [=](::sycl::item<1> index) {
            auto in0_ptr = in0_acc.template get_multi_ptr<::sycl::access::decorated::yes>();
            auto in1_ptr = in1_acc.template get_multi_ptr<::sycl::access::decorated::yes>();
            auto out_ptr = out_acc.template get_multi_ptr<::sycl::access::decorated::yes>();
            eltwise_kernel(index, in0_ptr, in1_ptr, out_ptr, in1_stride, op);
        });
    });
}

/**
 * @brief Get the appropriate operation function object based on eltwise_mode
 *
 * This function provides a centralized way to map eltwise_mode enum values
 * to their corresponding operation function objects.
 *
 * @param mode The eltwise operation mode
 * @return The operation function object as a variant
 */
using EltwiseOperator = std::variant<AddOp, SubOp, MulOp, DivOp, MaxOp, MinOp, PowOp, SquaredDiffOp, EqOp, NeOp, LtOp, LeOp, GtOp, GeOp>;

EltwiseOperator get_eltwise_operator(cldnn::eltwise_mode mode) {
    switch (mode) {
        case cldnn::eltwise_mode::sum:
            return AddOp{};
        case cldnn::eltwise_mode::sub:
            return SubOp{};
        case cldnn::eltwise_mode::prod:
            return MulOp{};
        case cldnn::eltwise_mode::div:
            return DivOp{};
        case cldnn::eltwise_mode::max:
            return MaxOp{};
        case cldnn::eltwise_mode::min:
            return MinOp{};
        case cldnn::eltwise_mode::pow:
            return PowOp{};
        case cldnn::eltwise_mode::squared_diff:
            return SquaredDiffOp{};
        case cldnn::eltwise_mode::eq:
            return EqOp{};
        case cldnn::eltwise_mode::ne:
            return NeOp{};
        case cldnn::eltwise_mode::lt:
            return LtOp{};
        case cldnn::eltwise_mode::le:
            return LeOp{};
        case cldnn::eltwise_mode::gt:
            return GtOp{};
        case cldnn::eltwise_mode::ge:
            return GeOp{};
        default:
            return AddOp{}; // fallback to ADD
    }
}

} // namespace

namespace cldnn {
namespace sycl {

struct eltwise_sycl : typed_primitive_sycl_impl<eltwise> {
    using parent = typed_primitive_sycl_impl<eltwise>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::sycl::eltwise_sycl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<eltwise_sycl>(*this);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, typed_primitive_inst<eltwise>& instance) override {
        auto& network = instance.get_network();
        const auto& desc = instance.get_typed_desc<eltwise>();

        auto& stream = downcast<sycl::sycl_stream>(network.get_stream());
        ::sycl::queue& sycl_queue = stream.get_sycl_queue();
        std::vector<::sycl::event> sycl_events = to_sycl_events(events);

        const auto& params = instance.get_impl_params();
        auto in0_shape = params->input_layouts[0].get_shape();
        auto in1_shape = params->input_layouts[1].get_shape();
        auto out_shape = params->output_layouts[0].get_shape();

        auto input0 = instance.input_memory_ptr(0);
        auto input1 = instance.input_memory_ptr(1);
        auto output = instance.output_memory_ptr(0);

        ov::element::Type_t in0_t = params->input_layouts[0].data_type;
        ov::element::Type_t in1_t = params->input_layouts[1].data_type;
        ov::element::Type_t out_t = params->output_layouts[0].data_type;

        OPENVINO_ASSERT(in0_t == in1_t && in0_t == out_t);
        OPENVINO_ASSERT(in0_shape.size() == in1_shape.size() && in0_shape.size() == out_shape.size());

        OPENVINO_ASSERT(ov::shape_size(in0_shape) == ov::shape_size(out_shape));
        if (ov::shape_size(in0_shape) != ov::shape_size(in1_shape)) {
            if (ov::shape_size(in1_shape) == 1) {
                // Handle broadcast case
            } else {
                OPENVINO_THROW("Eltwise does not support broadcasting except for scalar input");
            }
        }


        if (out_t == ov::element::f32) {
            OPENVINO_ASSERT(input0->get_allocation_type() == cldnn::allocation_type::sycl_buffer);
            auto buf_in0 = std::dynamic_pointer_cast<sycl::gpu_buffer>(input0)->get_buffer().reinterpret<float>();
            auto buf_in1 = std::dynamic_pointer_cast<sycl::gpu_buffer>(input1)->get_buffer().reinterpret<float>();
            auto buf_out = std::dynamic_pointer_cast<sycl::gpu_buffer>(output)->get_buffer().reinterpret<float>();

            auto op = get_eltwise_operator(desc->mode);
            auto ev = std::visit([&](auto&& operation) {
                return run_eltwise(sycl_queue, sycl_events, buf_in0, buf_in1, buf_out, in0_shape, in1_shape, out_shape, operation);
            }, op);
            return stream.create_base_event(ev);
        } else if (out_t == ov::element::f16) {
            OPENVINO_ASSERT(input0->get_allocation_type() == cldnn::allocation_type::sycl_buffer);
            auto buf_in0 = std::dynamic_pointer_cast<sycl::gpu_buffer>(input0)->get_buffer().reinterpret<::sycl::half>();
            auto buf_in1 = std::dynamic_pointer_cast<sycl::gpu_buffer>(input1)->get_buffer().reinterpret<::sycl::half>();
            auto buf_out = std::dynamic_pointer_cast<sycl::gpu_buffer>(output)->get_buffer().reinterpret<::sycl::half>();

            auto op = get_eltwise_operator(desc->mode);
            auto ev = std::visit([&](auto&& operation) {
                return run_eltwise(sycl_queue, sycl_events, buf_in0, buf_in1, buf_out, in0_shape, in1_shape, out_shape, operation);
            }, op);
            return stream.create_base_event(ev);
        } else {
            OPENVINO_THROW("No instance for given types found: ", out_t);
        }
    }

    static std::unique_ptr<primitive_impl> create(const eltwise_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        return std::make_unique<eltwise_sycl>(engine, config, nullptr /*weights_reorder*/);
    }
};

std::unique_ptr<primitive_impl> EltwiseImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<eltwise>());
    return sycl::eltwise_sycl::create(static_cast<const eltwise_node&>(node), params);
}

}  // namespace sycl
}  // namespace cldnn
