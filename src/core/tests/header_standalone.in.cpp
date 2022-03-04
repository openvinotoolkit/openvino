// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
