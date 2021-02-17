//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

/// \file This file is used to generate a series of .cpp files, one for each .hpp
/// file in ngraph, so the line below `#include "${HEADER}"` is expanded out to something
/// like `#include "/home/user/ngraph/src/ngraph/shape.hpp"`. The .cpp files are generated into
/// build/test/include_test/<headers>
/// The resulting .cpp file only includes this single file and then tries to compile it. If this
/// header file (shape.hpp in this example) does not #include everything it needs then the compile
/// will fail. You will need to add any missing #includes to make shape.hpp self-sufficient.
#define NGRAPH_OP(...)
#include "${HEADER}"
#undef NGRAPH_OP
