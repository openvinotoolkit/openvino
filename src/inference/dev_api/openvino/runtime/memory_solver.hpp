// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The header provides a declaration of MemorySolver utility class
 * @file openvino/runtime/memory_solver.hpp
 */

#pragma once

#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov {

/**
 * @brief Helps to solve issue of optimal memory allocation only for particular
 *        execution order.
 *
 * It works with abstract data description where
 * - Node is index in execution order
 * - Edge is Box object with size and start-finish indexes (live time)
 *
 * Example:
 *
 * Mem(offset)
 *  |        |____|             Box {4, 5}
 *  |  |_____________|          Box {2, 6}
 *  |     |____|                Box {3, 4}
 *  |  |____|                   Box {2, 3}
 *  |              |____|       Box {6, 7}
 *  |_____________________________________
 *   1  2  3  4  5  6  7  8  9  ExecOrder
 *
 *  Boxes which has an ExecOrder-axis intersection should have no Mem-axis intersections.
 *  The goal is to define a minimal required memory blob to store all boxes with such
 *  constraints and specify all corresponding position on Mem axis(through offset field).
 *
 *  NOTE!
 *  Exec order is predefined.
 */

class MemorySolver {
public:
    /** @brief Representation of edge (size and live time)*/
    struct Box {
        /** Execution order index of first use. The data will be produced here. */
        int start;

        /**
         * The execution order index of last use. After that data will be released.
         * -1 is a reserved value for "till to end". The data will be alive to very
         * end of execution.
         */
        int finish;

        /** Size of data. In abstract unit of measure (byte, simd, cache line, ...) */
        int64_t size;

        /** Box identifier, unique for each box. Will be used to querying calculated offset. */
        int64_t id;
    };

    /** @brief Performes inplace normalization of the input boxes
        @return lifespan of all boxes
    */
    static int normalize_boxes(std::vector<Box>& boxes) {
        int max_ts = 0;
        for (const Box& box : boxes)
            max_ts = std::max(std::max(max_ts, box.start), box.finish);
        for (Box& box : boxes)
            if (box.finish == -1)
                box.finish = max_ts;

        // sort by start and finish ts
        std::sort(boxes.begin(), boxes.end(), [](const Box& l, const Box& r) -> bool {
            return l.start < r.start || (l.start == r.start && l.finish < r.finish);
        });

        // remove unused timestamps (not a begin of some box)
        // each ts should start a box
        std::vector<bool> ts_exist(max_ts + 1);
        for (const Box& b : boxes)
            ts_exist[b.start] = true;

        int rm_ts_s = 0, rm_ts_f = 0;
        int ts_s = 0, ts_f = 0;
        for (Box& b : boxes) {
            while (ts_s < b.start)
                if (!ts_exist[ts_s++])
                    rm_ts_s++;

            if (ts_f > b.finish + 1) {
                ts_f = ts_s;
                rm_ts_f = rm_ts_s;
            }
            while (ts_f <= b.finish)
                if (!ts_exist[ts_f++])
                    rm_ts_f++;

            b.start -= rm_ts_s;
            b.finish -= rm_ts_f;
        }
        return ts_f - rm_ts_f;
    }

    explicit MemorySolver(const std::vector<Box>& boxes) : _boxes(boxes) {
        // TODO: add validation of data correctness:
        // 1. Box.start >= 0 and Box.finish >= -1
        // 2. Box.finish >= Box.start (except Box.finish == -1)
        // 3. Box.size > 0 (or == 0 ?)
        // 4. Box.id == any unique value
        _time_duration = normalize_boxes(_boxes);
    }

    inline bool popup_together_with(MemorySolver::Box& box_new, const MemorySolver::Box& box_old) {
        if (box_new.id + box_new.size > box_old.id && box_old.id + box_old.size > box_new.id) {
            // Move the new one up. There is an intersection
            box_new.id = box_old.id + box_old.size;
            return true;
        } else {
            return false;
        }
    }

    /**
     * @brief Solve memory location with maximal reuse.
     * @return Size of common memory blob required for storing all
     */
    int64_t solve() {
        max_top_depth();  // at first make sure that we no need more for boxes sorted by box.start
        std::vector<std::vector<const Box*>> time_slots(_time_duration);
        for (auto& slot : time_slots)
            slot.reserve(_top_depth);  // 2D array [_time_duration][_top_depth]

        // Sort be box size. First is biggest
        // Comment this line to check other order of box putting
        std::sort(_boxes.begin(), _boxes.end(), [](const Box& l, const Box& r) {
            return l.size > r.size;
        });

        int64_t _min_required = 0;

        for (Box& box : _boxes) {
            // start from bottom and will lift it up if intersect with other present
            int64_t id = box.id;
            box.id = 0;  // id will be used as a temp offset storage
            bool popped_up;
            do {
                popped_up = false;
                for (int i_slot = box.start; i_slot <= box.finish; i_slot++) {
                    for (auto* box_in_slot : time_slots[i_slot]) {
                        // intersect with already stored boxes for all covered time slots
                        // and move up the new one if needed
                        // Execution of 'popup_together_with' is important even if 'popped_up' is already 'true'
                        popped_up = popup_together_with(box, *box_in_slot) || popped_up;
                    }
                }
            } while (popped_up);

            // add current box to covered time slot
            for (int i_slot = box.start; i_slot <= box.finish; i_slot++)
                time_slots[i_slot].push_back(&box);

            // store the max top bound for each box
            _min_required = std::max(_min_required, box.id + box.size);
            _offsets[id] = box.id;  // TODO: move to constructor (use .insert instead of [])
        }

        return _min_required;
    }

    /** Provides calculated offset for specified box id */
    int64_t get_offset(int id) const {
        auto res = _offsets.find(id);
        if (res == _offsets.end())
            OPENVINO_THROW("There are no box for provided ID");
        return res->second;
    }

    /** Additional info. Max sum of box sizes required for any time stamp. */
    int64_t max_depth() {
        if (_depth == -1)
            calc_depth();
        return _depth;
    }
    /** Additional info. Max num of boxes required for any time stamp. */
    int64_t max_top_depth() {
        if (_top_depth == -1)
            calc_depth();
        return _top_depth;
    }

private:
    std::vector<Box> _boxes;
    std::map<int64_t, int64_t> _offsets;
    int64_t _top_depth = -1;
    int64_t _depth = -1;
    int _time_duration = -1;

    void calc_depth() {
        int64_t top_depth = 0;
        int64_t depth = 0;
        std::map<int64_t, std::vector<const Box*>> release_at;

        for (const Box& box : _boxes) {
            int64_t time = box.start;
            depth += box.size;
            top_depth++;

            release_at[box.finish + 1].push_back(&box);

            for (const Box* b : release_at[time]) {
                depth -= b->size;
                top_depth--;
            }
            release_at.erase(time);
            OPENVINO_ASSERT(top_depth > 0);

            _top_depth = std::max(_top_depth, top_depth);
            _depth = std::max(_depth, depth);
        }
    }
};

}  // namespace ov
