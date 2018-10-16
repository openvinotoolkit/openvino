/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef EVENT_HPP
#define EVENT_HPP

#include "nstl.hpp"

namespace mkldnn {
namespace impl {

/** \brief A primitive level synchronization abstraction with error handling */
struct event_t: public c_compatible {
    enum state {
        /** event not happened yet */
        wait,
        /** event happened, no error appeared */
        ready,
        /** event happened, corresponding primitive caused an error */
        error,
        /** event happened, corresponding primitive was not run, because of
         * previous error */
        aborted
    };

    event_t(): _state(event_t::wait) {}

    /** returns current event state */
    inline state get_state() const { return _state; }

    /** sets state to @p astate */
    inline void set_state(state astate) { _state = astate; }

    /** returns true if state() != wait */
    inline bool finished() const { return get_state() != event_t::wait; }

    /** resets state to @c wait */
    inline void reset() { set_state(event_t::wait); }

protected:
    volatile state _state;
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
