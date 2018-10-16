/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

INST_TEST_CASE(AlexNet_NCHW,
    PARAMS(nchw, oihw, FMT_BIAS, nchw,
        2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4),
    PARAMS(nchw, goihw, FMT_BIAS, nchw,
        2, 2, 96, 27, 27, 256, 27, 27, 5, 5, 2, 2, 1, 1),
    PARAMS(nchw, oihw, FMT_BIAS, nchw,
        2, 1, 256, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1),
    PARAMS(nchw, goihw, FMT_BIAS, nchw,
        2, 2, 384, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1),
    PARAMS(nchw, goihw, FMT_BIAS, nchw,
        2, 2, 384, 13, 13, 256, 13, 13, 3, 3, 1, 1, 1, 1)
);

INST_TEST_CASE(AlexNet_Blocked,
    PARAMS(nchw, Ohwi8o, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4),
    PARAMS(nhwc, Ohwi8o, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED_G, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 2, 96, 27, 27, 256, 27, 27, 5, 5, 2, 2, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 256, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED_G, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 2, 384, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED_G, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 2, 384, 13, 13, 256, 13, 13, 3, 3, 1, 1, 1, 1)
);

INST_TEST_CASE(AlexNet_Blocked16,
    PARAMS(nchw, Ohwi16o, FMT_BIAS, FMT_DATA_BLOCKED16,
        2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4),
    PARAMS(nhwc, Ohwi16o, FMT_BIAS, FMT_DATA_BLOCKED16,
        2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4),
    PARAMS(FMT_DATA_BLOCKED16, FMT_WEIGHTS_BLOCKED16_G, FMT_BIAS, FMT_DATA_BLOCKED16,
        2, 2, 96, 27, 27, 256, 27, 27, 5, 5, 2, 2, 1, 1),
    PARAMS(FMT_DATA_BLOCKED16, FMT_WEIGHTS_BLOCKED16, FMT_BIAS, FMT_DATA_BLOCKED16,
        2, 1, 256, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED16, FMT_WEIGHTS_BLOCKED16_G, FMT_BIAS, FMT_DATA_BLOCKED16,
        2, 2, 384, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED16, FMT_WEIGHTS_BLOCKED16_G, FMT_BIAS, FMT_DATA_BLOCKED16,
        2, 2, 384, 13, 13, 256, 13, 13, 3, 3, 1, 1, 1, 1)
);
