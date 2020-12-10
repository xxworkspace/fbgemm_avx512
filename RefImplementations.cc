/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "skylark/inference/blas/RefImplementations.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

using namespace std;

namespace fbgemm {

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
    float beta) {
  // clang-format on
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += Afp32[i * lda + k] * Bfp32[k * ldb + j];
      }
      Cfp32[i * ldc + j] = beta * Cfp32[i * ldc + j] + sum;
    }
  }
}

// clang-format off
void relu(int m, int n,
                  float *h, int ldh,
                  const float* a, int lda) {
  // clang-format on
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      h[i * ldh + j] = std::max(a[i * lda + j], 0.f);
    }
  }
}

namespace wavernn {

// h = u * (h_prev - e) + e;
// clang-format off
void hidden(int m, int n,
            float* h, int ldh,
            const float* x, int ldx,
            const float* y, int ldy,
            const float* z, int ldz) {
  // clang-format on
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      h[i * ldh + j] = x[i * ldx + j] * y[i * ldy + j] +
                       (1.0 - x[i * ldx + j]) * z[i * ldz + j];
    }
  }
}

// u = (x + y + z).sigmoid()
// clang-format off
void coarse_sigmoid(int m, int n,
                    float* d, int ldd,
                    const float* x, int ldx,
                    const float* y, int ldy,
                    const float* z, int ldz) {
  // clang-format on
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      d[i * ldd + j] =
          1.0 / (1 + exp(-(x[i * ldx + j] + y[i * ldy + j] + z[i * ldz + j])));
    }
  }
}

// e = (x + y * r + z).tanh()
// clang-format off
void coarse_tanh(int m, int n,
                 float* e, int lde,
                 const float* x, int ldx,
                 const float* y, int ldy,
                 const float* r, int ldr,
                 const float* z, int ldz) {
  // clang-format on
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      e[i * lde + j] = tanh(x[i * ldx + j] + y[i * ldy + j] * r[i * ldr + j] +
                            z[i * ldz + j]);
    }
  }
}

// u = (x + y + z + c).sigmoid()
// clang-format off
void fine_sigmoid(int m, int n,
                  float* u, int ldu,
                  const float* x, int ldx,
                  const float* y, int ldy,
                  const float* z, int ldz,
                  const float* c, int ldc) {
  // clang-format on
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      u[i * ldu + j] = 1.0 / (1 + exp(-(x[i * ldx + j] + y[i * ldy + j] +
                                        z[i * ldz + j] + c[i * ldc + j])));
    }
  }
}

// e = (x + y * r + z + c).tanh()
// clang-format off
void fine_tanh(int m, int n,
               float* e, int lde,
               const float* x, int ldx,
               const float* y, int ldy,
               const float* r, int ldr,
               const float* z, int ldz,
               const float* c, int ldc) {
  // clang-format on
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      e[i * lde + j] = tanh(x[i * ldx + j] + y[i * ldy + j] * r[i * ldr + j] +
                            z[i * ldz + j] + c[i * ldc + j]);
    }
  }
}
// clang-format on

}  // namespace wavernn

}  // namespace fbgemm
