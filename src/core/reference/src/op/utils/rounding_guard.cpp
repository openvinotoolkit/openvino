#include "openvino/reference/rounding_guard.hpp"

namespace ov {
RoundingGuard::RoundingGuard(int mode) : m_prev_round_mode{std::fegetround()} {
    std::fesetround(mode);
}

RoundingGuard::~RoundingGuard() {
    std::fesetround(m_prev_round_mode);
}
}  // namespace ov
