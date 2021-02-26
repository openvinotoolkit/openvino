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

// Contains logic derived from TensorFlow’s bint4 implementation
// https://github.com/tensorflow/tensorflow/blob/d354efc/tensorflow/core/lib/int4/int4.h
// Copyright notice from original source file is as follows.

//*******************************************************************************
//  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//    http://www.apache.org/licenses/LICENSE-2.0
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//==============================================================================

#include <cmath>
#include <iostream>
#include <limits>

#include "ngraph/type/int4.hpp"

using namespace std;
using namespace ngraph;


int4::int4(int8_t value)
{
}

std::string int4::to_string() const
{
    return std::to_string(static_cast<int8_t>(*this));
}

size_t int4::size() const
{
    return sizeof(m_value);
}

int4::operator int8_t() const
{
    return 0;
}

int8_t int4::to_bits() const
{
    return 0;
}
