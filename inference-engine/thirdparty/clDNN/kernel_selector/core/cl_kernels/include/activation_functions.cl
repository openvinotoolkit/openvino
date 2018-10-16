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
#include "common.cl"

// TODO: use native_exp and use cast for APL
#define ACTIVATION_LOGISTIC(input)                      (UNIT_VAL_ONE/(UNIT_VAL_ONE + exp(-input)))
#define ACTIVATION_HYPERBOLIC_TAN(input)                (tanh(input))
#define ACTIVATION_RELU(input)                          (UNIT_MAX_FUNC(UNIT_VAL_ZERO, input))
#define ACTIVATION_RELU_NEGATIVE_SLOPE(input, slope)    isinf(TO_UNIT_TYPE(slope)) ? ((input >= UNIT_VAL_ZERO) ? \
                                                        input : -TO_UNIT_TYPE(slope)) : \
                                                        (UNIT_MAX_FUNC(input, UNIT_VAL_ZERO) + TO_UNIT_TYPE(slope) * UNIT_MIN_FUNC(input, UNIT_VAL_ZERO))
#define ACTIVATION_ELU(input, alpha)                    (UNIT_MAX_FUNC(input, UNIT_VAL_ZERO) +  \
                                                        TO_UNIT_TYPE(alpha) * (exp(UNIT_MIN_FUNC(input, UNIT_VAL_ZERO)) - UNIT_VAL_ONE));
#define ACTIVATION_CLAMP(input, m, n)                   (UNIT_MAX_FUNC(TO_UNIT_TYPE(m), UNIT_MIN_FUNC(TO_UNIT_TYPE(n), input)))
#define ACTIVATION_SOFTRELU(input)                      (log(UNIT_VAL_ONE + exp(input)))
#define ACTIVATION_ABS(input)                           (fabs(input))
#define ACTIVATION_LINEAR(input, m, n)                  (m*input + n)
#define ACTIVATION_SQUARE(input)                        (input*input)
#define ACTIVATION_SQRT(input)                          (sqrt(input))

#define ACTIVATION_RELU_GRAD(input_grad, input)                          (input_grad * (input > UNIT_VAL_ZERO ? TO_UNIT_TYPE(1) : TO_UNIT_TYPE(0)))
#define ACTIVATION_RELU_NEGATIVE_SLOPE_GRAD(input_grad, input, slope)    (input_grad * ((input > UNIT_VAL_ZERO ? TO_UNIT_TYPE(1) : TO_UNIT_TYPE(0)) + TO_UNIT_TYPE(slope) * (input <= 0 ? TO_UNIT_TYPE(1) : TO_UNIT_TYPE(0))))

#if defined ACTIVATION_FUNCTION_LOGISTIC
    #define ACTIVATION(input, m, n) ACTIVATION_LOGISTIC(input)
#elif defined ACTIVATION_FUNCTION_HYPERBOLIC_TAN
    #define ACTIVATION(input, m, n) ACTIVATION_HYPERBOLIC_TAN(input)
#elif defined ACTIVATION_FUNCTION_RELU
    #define ACTIVATION(input, m, n) ACTIVATION_RELU(input)
#elif defined ACTIVATION_FUNCTION_RELU_NEGATIVE_SLOPE
    #define ACTIVATION(input, m, n) ACTIVATION_RELU_NEGATIVE_SLOPE(input, m)
#elif defined ACTIVATION_FUNCTION_CLAMP
    #define ACTIVATION(input, m, n) ACTIVATION_CLAMP(input, m, n)
#elif defined ACTIVATION_FUNCTION_SOFTRELU
    #define ACTIVATION(input, m, n) ACTIVATION_SOFTRELU(input)    
#elif defined ACTIVATION_FUNCTION_ABS
    #define ACTIVATION(input, m, n) ACTIVATION_ABS(input)
#elif defined ACTIVATION_FUNCTION_LINEAR
    #define ACTIVATION(input, m, n) ACTIVATION_LINEAR(input, m, n)
#elif defined ACTIVATION_FUNCTION_SQUARE
    #define ACTIVATION(input, m, n) ACTIVATION_SQUARE(input)
#elif defined ACTIVATION_FUNCTION_SQRT
    #define ACTIVATION(input, m, n) ACTIVATION_SQRT(input)
#elif defined ACTIVATION_FUNCTION_ELU
    #define ACTIVATION(input, m, n) ACTIVATION_ELU(input, m)
#elif defined ACTIVATION_FUNCTION_RELU_GRAD
    #define ACTIVATION(input_grad, input, m, n) ACTIVATION_RELU_GRAD(input_grad, input)
#elif defined ACTIVATION_FUNCTION_RELU_NEGATIVE_SLOPE_GRAD
    #define ACTIVATION(input_grad, input, m, n) ACTIVATION_RELU_NEGATIVE_SLOPE_GRAD(input_grad, input, m)
#elif defined ACTIVATION_FUNCTION_NONE_GRAD
    #define ACTIVATION(input_grad, input, m, n) input_grad
#else
    #define ACTIVATION(input, m, n) input
#endif
