#include "ngraph/static_shape.hpp"

using namespace ngraph;

Shape StaticShape::to_shape() const {
    std::vector<size_t> shape_dimensions(size());
    std::transform(begin(), end(), shape_dimensions.begin(), [](const StaticDimension& d) {
        return d.get_length();
    });
    return shape_dimensions;
}

bool StaticShape::compatible(const StaticShape& s) const {
    if (size() != s.size())
        return false;
    for (size_t i = 0; i < size(); ++i)
        if (!(*this)[i].compatible(s[i]))
            return false;
    return true;
}

bool StaticShape::merge_into(StaticShape& dst, const StaticShape& src) {
    const auto& dst_rank = dst.size();
    const auto& src_rank = src.size();
    if (dst_rank != src_rank) {
        return false;
    } else {
        bool success = true;
        for (size_t i = 0; i < dst_rank; ++i)
            success &= StaticDimension::merge(dst[i], dst[i], src[i]);
        return success;
    }
}

bool ngraph::StaticShape::broadcast_merge_into(StaticShape& dst,
                                               const StaticShape& src,
                                               const op::AutoBroadcastSpec& autob) {
    const auto& d_rank = dst.rank();
    const auto& s_rank = src.rank();
    switch (autob.m_type) {
    case op::AutoBroadcastType::NONE:
        return true;
    case op::AutoBroadcastType::NUMPY: {
        if (d_rank.is_dynamic() || s_rank.is_dynamic()) {
            dst = PartialShape::dynamic();
            return true;
        } else {
            // Ranks are both static.
            const auto& dst_rank = d_rank.get_length();
            const auto& src_rank = s_rank.get_length();
            const auto& new_rank = std::max(dst_rank, src_rank);
            const auto& ranks_diff = src_rank - dst_rank;
            auto source = src;
            if (ranks_diff > 0)
                dst.insert(dst.begin(), src.begin(), src.begin() + ranks_diff);
            else if (ranks_diff < 0)
                source.insert(source.begin(), -ranks_diff, 1);

            bool success = true;
            for (int64_t i = std::abs(ranks_diff); i < new_rank; ++i) {
                if (source[i] == 1)
                    continue;
                else if (dst[i] == 1)
                    dst[i] = source[i];
                else
                    success &= StaticDimension::merge(dst[i], dst[i], source[i]);
            }
            return success;
        }
    }
    case op::AutoBroadcastType::PDPD: {
        if (d_rank.is_dynamic() || s_rank.is_dynamic()) {
            return true;
        } else {
            // Ranks are both static.
            auto dst_rank = d_rank.get_length();
            auto src_rank = s_rank.get_length();
            if (dst_rank == src_rank && dst.compatible(src))
                return true;

            int64_t axis = autob.m_axis;
            if (axis < -1) {
                return false;
            }
            if (axis == -1) {
                axis = dst_rank - src_rank;
            }

            size_t len = src_rank;
            while (len > 0 && src[len - 1].is_static() && src[len - 1].get_length() == 1) {
                --len;
            }

            for (size_t i = axis; i < axis + len; ++i) {
                if (!(dst[i].compatible(src[i - axis]))) {
                    return false;
                }
            }

            return true;
        }
    }
    default:
        NGRAPH_CHECK(false, "Unsupported auto broadcast type: ", autob.m_type);
    }

    return false;
}

std::ostream& ngraph::operator<<(std::ostream& str, const StaticShape& shape) {
    str << "{";
    bool first = true;
    for (auto& d : shape) {
        if (!first) {
            str << ",";
        }
        str << d.get_length();
        first = false;
    }
    return (str << "}");
}