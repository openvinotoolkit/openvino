#include "openvino/reference/round_guard.hpp"

namespace ov {
RoundGuard::RoundGuard(int mode) : m_prev_round_mode{std::fegetround()} {
    std::fesetround(mode);
}

RoundGuard::~RoundGuard() {
    std::fesetround(m_prev_round_mode);
}
}  // namespace ov
