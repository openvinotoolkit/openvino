/*
// Copyright (c) 2016 Intel Corporation
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

inline uint4 FUNC(reshape_2_to_4)(uint o, uint i, uint y, uint x, uint dst_size_y, uint dst_size_x)
{
    uint _i  = i / (dst_size_y*dst_size_x);
    uint _yx = i % (dst_size_y*dst_size_x);
    uint _y = _yx / dst_size_x;
    uint _x = _yx % dst_size_x;
    return (uint4)(o,_i,_y,_x);
}

inline uint4 FUNC(reshape_4_to_2)(uint o, uint i, uint y, uint x, uint src_size_y, uint src_size_x)
{
    uint _i = i*src_size_y*src_size_x + y*src_size_x + x;
    return (uint4)(o,_i,0,0);
}

inline uint4 FUNC(reshape_dims)(uint o, uint i, uint y, uint x, uint src_size_y, uint src_size_x, uint dst_size_y, uint dst_size_x, uint src_dims, uint dst_dims)
{
    if (src_dims == 4 && dst_dims == 2)
    {
        return FUNC_CALL(reshape_4_to_2)(o,i,y,x,src_size_y,src_size_x);
    }
    else if (src_dims == 2 && dst_dims == 4)
    {
        return FUNC_CALL(reshape_2_to_4)(o,i,y,x,dst_size_y,dst_size_x);
    }
    
    return (uint4)(o,i,y,x);
}