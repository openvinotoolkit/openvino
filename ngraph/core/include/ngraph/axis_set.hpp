// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ostream>
#include <set>
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    /// \brief A set of axes.
    class AxisSet : public std::set<size_t>
    {
    public:
        NGRAPH_API AxisSet();

        NGRAPH_API AxisSet(const std::initializer_list<size_t>& axes);

        NGRAPH_API AxisSet(const std::set<size_t>& axes);

        NGRAPH_API AxisSet(const std::vector<size_t>& axes);

        NGRAPH_API AxisSet(const AxisSet& axes);

        NGRAPH_API AxisSet& operator=(const AxisSet& v);

        NGRAPH_API AxisSet& operator=(AxisSet&& v) noexcept;

        NGRAPH_API std::vector<int64_t> to_vector() const;
    };

    template <>
    class NGRAPH_API AttributeAdapter<AxisSet> : public ValueAccessor<std::vector<int64_t>>
    {
    public:
        AttributeAdapter(AxisSet& value)
            : m_ref(value)
        {
        }

        const std::vector<int64_t>& get() override;
        void set(const std::vector<int64_t>& value) override;
        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<AxisSet>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        operator AxisSet&() { return m_ref; }

    protected:
        AxisSet& m_ref;
        std::vector<int64_t> m_buffer;
        bool m_buffer_valid{false};
    };

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const AxisSet& axis_set);
} // namespace ngraph
