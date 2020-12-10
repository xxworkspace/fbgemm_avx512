/*
 * Copyright (c) LAIX, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef FBGEMM_UKERNELS_AVX512
#define FBGEMM_UKERNELS_AVX512
#include <cstdint>
#include <tuple>
#include <vector>
#include "skylark/inference/blas/Types.h"

namespace fbgemm {

void __attribute__((noinline)) gemmkernel_1x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_2x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_3x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_4x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_5x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_6x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_7x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_8x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_9x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_10x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_11x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_12x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_13x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_14x2_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_1x4_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_2x4_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_3x4_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_4x4_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_5x4_AVX512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_6x4_AVX512_fA0fB0fC0(GemmParams* gp);
typedef void (*funcptr_fp16)(GemmParams* gp);
;

}  // namespace fbgemm

#endif  // FBGEMM_UKERNELS
