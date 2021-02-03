#!/usr/bin/env python
# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import copy
import numpy as np

def LRN(input, size=3, bias=1.0, alpha=3.0, beta=0.5):
    output = copy.deepcopy(input)
    N = input.shape[0]
    C = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]
    for n in range(N):
        begin_n = max(0, n - (size-1)//2)
        end_n = min(N, n + (size-1)//2 + 1)
        for c in range(C):
            begin_c = max(0, c - (size-1)//2)
            end_c = min(C, c + (size-1)//2 + 1)
            for h in range(H):
                    begin_h = max(0, h - (size-1)//2)
                    end_h = min(H, h + (size-1)//2 + 1)
                    for w in range(W):
                        begin_w = max(0, w - (size-1)//2)
                        end_w = min(W, w + (size-1)//2 + 1)
                        patch = input[n, c, begin_h:end_h, begin_w:end_w]
                        output[n, c, h, w] /= (
                            np.power(bias + (alpha/(size**2)) * np.sum(patch * patch), beta))
    return output

input = np.arange(0, 12, 1).reshape(2, 3, 2, 1).astype(np.float32)
result = LRN(input)
for elem in np.nditer(result):
    print("{:.7f}f,".format(elem))
