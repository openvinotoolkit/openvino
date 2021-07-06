// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/partial_shape.hpp"

namespace ngraph
{
    template <typename AXIS_VALUES>
    AXIS_VALUES project(const AXIS_VALUES& axis_values, const AxisSet& axes)
    {
        AXIS_VALUES result;

        for (size_t i = 0; i < axis_values.size(); i++)
        {
            if (axes.find(i) != axes.end())
            {
                result.push_back(axis_values[i]);
            }
        }

        return result;
    }

    template <>
    NGRAPH_API PartialShape project(const PartialShape& shape, const AxisSet& axes);

    // Removes some values from a vector of axis values
    template <typename AXIS_VALUES>
    AXIS_VALUES reduce(const AXIS_VALUES& axis_values, const AxisSet& deleted_axes, bool keep_dims)
    {
        AXIS_VALUES result;

        for (size_t i = 0; i < axis_values.size(); i++)
        {
            if (deleted_axes.find(i) == deleted_axes.end())
            {
                result.push_back(axis_values[i]);
            }
            else
            {
                if (keep_dims)
                    result.push_back(1);
            }
        }

        return result;
    }

    template <>
    NGRAPH_API PartialShape reduce(const PartialShape& shape,
                                   const AxisSet& deleted_axes,
                                   bool keep_dims);

    // TODO: check validity, i.e. that the new axis indices are all less than
    // axis_values.size()+num_new_axes.
    // Add new values at particular axis positions
    template <typename AXIS_VALUES, typename AXIS_VALUE>
    AXIS_VALUES inject_pairs(const AXIS_VALUES& axis_values,
                             std::vector<std::pair<size_t, AXIS_VALUE>> new_axis_pos_value_pairs)
    {
        AXIS_VALUES result;

        size_t original_pos = 0;

        for (size_t result_pos = 0;
             result_pos < axis_values.size() + new_axis_pos_value_pairs.size();
             result_pos++)
        {
            // Would be nice to use std::find_if here but would rather not #include <algorithm> in
            // this header
            auto search_it = new_axis_pos_value_pairs.begin();

            while (search_it != new_axis_pos_value_pairs.end())
            {
                if (search_it->first == result_pos)
                {
                    break;
                }
                ++search_it;
            }

            if (search_it == new_axis_pos_value_pairs.end())
            {
                result.push_back(axis_values[original_pos++]);
            }
            else
            {
                result.push_back(search_it->second);
            }
        }

        return result;
    }

    template <>
    NGRAPH_API PartialShape
        inject_pairs(const PartialShape& shape,
                     std::vector<std::pair<size_t, Dimension>> new_axis_pos_value_pairs);

    // Add a new value at a particular axis position
    template <typename AXIS_VALUES, typename AXIS_VALUE>
    AXIS_VALUES inject(const AXIS_VALUES& axis_values, size_t new_axis_pos, AXIS_VALUE new_axis_val)
    {
        return inject_pairs(axis_values,
                            std::vector<std::pair<size_t, AXIS_VALUE>>{
                                std::pair<size_t, AXIS_VALUE>(new_axis_pos, new_axis_val)});
    }
} // namespace ngraph
