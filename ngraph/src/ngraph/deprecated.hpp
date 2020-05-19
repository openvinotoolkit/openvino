//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

//
// The NGRAPH_DEPRECATED macro can be used to deprecate a function declaration. For example:
//
//     void frobnicate() NGRAPH_DEPRECATED("replace with groxify");
//
// If nGraph was built with `-DNGRAPH_DEPRECATED_ENABLE=ON`, the macro will expand to a
// deprecation attribute supported by the compiler, so any use of `frobnicate` will produce a
// compiler warning. Otherwise, `NGRAPH_DEPRECATED` has no effect.
//
// TODO(amprocte): Test to make sure if this works in compilers other than clang. (Should be no
// problem for the moment if it doesn't, since it defaults to `OFF` and we can investigate later
// ways to implement this in other compilers.)
//
#ifdef NGRAPH_DEPRECATED_ENABLE
#define NGRAPH_DEPRECATED(msg) __attribute__((deprecated((msg))))
#define NGRAPH_DEPRECATED_DOC /// \deprecated
#else
#define NGRAPH_DEPRECATED(msg)
#define NGRAPH_DEPRECATED_DOC
#endif
