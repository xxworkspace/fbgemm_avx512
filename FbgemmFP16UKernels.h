/*
 * Copyright (c) LAIX, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef FBGEMM_UKERNELS
#define FBGEMM_UKERNELS
#include <cstdint>
#include <tuple>
#include <vector>
#include "skylark/inference/blas/Types.h"

namespace fbgemm {

void __attribute__((noinline)) gemmkernel_1x2_AVX256_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_2x2_AVX256_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_3x2_AVX256_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_4x2_AVX256_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_5x1_AVX256_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_6x1_AVX256_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_7x1_AVX256_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_8x1_AVX256_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_9x1_AVX256_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_10x1_AVX256_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_11x1_AVX256_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_12x1_AVX256_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_13x1_AVX256_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_14x1_AVX256_fA0fB0fC0(GemmParams* gp);
typedef void (*funcptr_fp16)(GemmParams* gp);
;

}  // namespace fbgemm

#endif  // FBGEMM_UKERNELS
