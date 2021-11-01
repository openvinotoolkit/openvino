#pragma once

#include <openvino/core/node.hpp>

namespace ov {
namespace shape_utils {

#if 0
template <typename To, typename From>
typename std::enable_if<!std::is_convertible<From, To>::value, To>::type cast(From src) {
    OPENVINO_UNREACHABLE("cast failure.");
    return To{};
}

template <typename To, typename From>
typename std::enable_if<std::is_convertible<From, To>::value, To>::type cast(From src) {
    return src;
}
#endif

template <typename T>
inline Interval get_interval(const op::Op* op, T& dim) {
    NODE_VALIDATION_CHECK(op, false, "Cannot get_interval for static shape.");
    return Interval();
}

template <>
inline Interval get_interval<Dimension>(const op::Op* op, Dimension& dim) {
    return dim.get_interval();
}

}  // namespace shape_utils
}  // namespace ov