/*
// Copyright (c) 2016-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "api/cldnn.hpp"
#include <memory>
#include <string>
#include <vector>

#ifndef CLDNN_VERSION_MAJOR
#define CLDNN_VERSION_MAJOR (0)
#endif

#ifndef CLDNN_VERSION_MINOR
#define CLDNN_VERSION_MINOR (0)
#endif

#ifndef CLDNN_VERSION_BUILD
#define CLDNN_VERSION_BUILD (0)
#endif

#ifndef CLDNN_VERSION_REVISION
#define CLDNN_VERSION_REVISION (0)
#endif

namespace cldnn {

version_t get_version() {
    return { CLDNN_VERSION_MAJOR, CLDNN_VERSION_MINOR, CLDNN_VERSION_BUILD, CLDNN_VERSION_REVISION };
}

}
