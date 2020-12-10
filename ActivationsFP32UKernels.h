/*
 * Copyright (c) LAIX, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef LAIX_ACTIVATIONS_UKERNELS
#define LAIX_ACTIVATIONS_UKERNELS

#include <cstdint>
#include <tuple>
#include <vector>
#include "skylark/inference/blas/ActivationsFP32.h"

#include "skylark/inference/blas/Types.h"

namespace fbgemm {

void __attribute__((noinline)) activationkernel_avx256_tanh_1x1(TanhParams* gp);
void __attribute__((noinline)) activationkernel_avx256_tanh_2x1(TanhParams* gp);
void __attribute__((noinline)) activationkernel_avx256_tanh_3x1(TanhParams* gp);
void __attribute__((noinline)) activationkernel_avx256_tanh_4x1(TanhParams* gp);
void __attribute__((noinline)) activationkernel_avx256_tanh_5x1(TanhParams* gp);
void __attribute__((noinline)) activationkernel_avx256_tanh_6x1(TanhParams* gp);
void __attribute__((noinline)) activationkernel_avx256_tanh_7x1(TanhParams* gp);
void __attribute__((noinline)) activationkernel_avx256_tanh_8x1(TanhParams* gp);


void __attribute__((noinline)) activationkernel_avx256_sigmoid_1x1(SigmoidParams* gp);
void __attribute__((noinline)) activationkernel_avx256_sigmoid_2x1(SigmoidParams* gp);
void __attribute__((noinline)) activationkernel_avx256_sigmoid_3x1(SigmoidParams* gp);
void __attribute__((noinline)) activationkernel_avx256_sigmoid_4x1(SigmoidParams* gp);
void __attribute__((noinline)) activationkernel_avx256_sigmoid_5x1(SigmoidParams* gp);
void __attribute__((noinline)) activationkernel_avx256_sigmoid_6x1(SigmoidParams* gp);
void __attribute__((noinline)) activationkernel_avx256_sigmoid_7x1(SigmoidParams* gp);
void __attribute__((noinline)) activationkernel_avx256_sigmoid_8x1(SigmoidParams* gp);


void __attribute__((noinline)) activationkernel_avx256_relu_1x4(ReLUParams* gp);
void __attribute__((noinline)) activationkernel_avx256_relu_2x4(ReLUParams* gp);
void __attribute__((noinline)) activationkernel_avx256_relu_3x4(ReLUParams* gp);
void __attribute__((noinline)) activationkernel_avx256_relu_4x4(ReLUParams* gp);
void __attribute__((noinline)) activationkernel_avx256_relu_5x4(ReLUParams* gp);
void __attribute__((noinline)) activationkernel_avx256_relu_6x4(ReLUParams* gp);
void __attribute__((noinline)) activationkernel_avx256_relu_7x4(ReLUParams* gp);
void __attribute__((noinline)) activationkernel_avx256_relu_8x4(ReLUParams* gp);


namespace wavernn {

void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_1x4(CoarseSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_2x4(CoarseSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_3x4(CoarseSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_4x4(CoarseSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_5x4(CoarseSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_6x4(CoarseSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_7x4(CoarseSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_8x4(CoarseSigmoidParams* gp);


void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_1x4(CoarseTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_2x4(CoarseTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_3x4(CoarseTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_4x4(CoarseTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_5x4(CoarseTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_6x4(CoarseTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_7x4(CoarseTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_8x4(CoarseTanhParams* gp);


void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_1x4(FineSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_2x4(FineSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_3x4(FineSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_4x4(FineSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_5x4(FineSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_6x4(FineSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_7x4(FineSigmoidParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_8x4(FineSigmoidParams* gp);


void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_1x4(FineTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_2x4(FineTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_3x4(FineTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_4x4(FineTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_5x4(FineTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_6x4(FineTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_7x4(FineTanhParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_8x4(FineTanhParams* gp);


void __attribute__((noinline)) wavernnkernel_avx256_hidden_1x4(HiddenParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_hidden_2x4(HiddenParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_hidden_3x4(HiddenParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_hidden_4x4(HiddenParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_hidden_5x4(HiddenParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_hidden_6x4(HiddenParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_hidden_7x4(HiddenParams* gp);
void __attribute__((noinline)) wavernnkernel_avx256_hidden_8x4(HiddenParams* gp);


}  // namespace wavernn

}  // namespace fbgemm

#endif  // LAIX_ACTIVATIONS_UKERNELS
