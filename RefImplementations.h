/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <algorithm>
#include <cstdint>

namespace fbgemm {

// reference implementation
/**
 * @brief Reference implementation of matrix multiply with fp32 (single
 * precision) floating point number.
 */
// clang-format off
void matmul_fp_ref(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    const float* Afp32,
    const float* Bfp32,
    float* Cfp32,
    float beta = 0.f);
// clang-format on

// clang-format off
void relu(int m, int n,
          float *h, int ldh,
          const float* a, int lda);
// clang-format on

namespace wavernn {

// clang-format off
// h = u * (h_prev - e) + e;
void hidden(int m, int n,
            float* h, int ldh,
            const float* x, int ldx,
            const float* y, int ldy,
            const float* z, int ldz);

// u = (x + y + z).sigmoid()
void coarse_sigmoid(int m, int n,
                    float* u, int ldu,
                    const float* x, int ldx,
                    const float* y, int ldy,
                    const float* z, int ldz);

// e = (x + y * r + z).tanh()
void coarse_tanh(int m, int n,
                 float* e, int lde,
                 const float* x, int ldx,
                 const float* y, int ldy,
                 const float* r, int ldr,
                 const float* z, int ldz);

// u = (x + y + z + c).sigmoid()
void fine_sigmoid(int m, int n,
                  float* u, int ldu,
                  const float* x, int ldx,
                  const float* y, int ldy,
                  const float* z, int ldz,
                  const float* c, int ldc);

// e = (x + y * r + z + c).tanh()
void fine_tanh(int m, int n,
               float* e, int lde,
               const float* x, int ldx,
               const float* y, int ldy,
               const float* r, int ldr,
               const float* z, int ldz,
               const float* c, int ldc);
// clang-format on

}  // namespace wavernn

}  // namespace fbgemm
