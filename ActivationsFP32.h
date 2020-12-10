/*
 * Copyright (c) LAIX, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

// WARNING: this is a legacy fp16 fbgemm implementation and will soon be
// upgraded to match with new fbgemm interface.

#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>

#include "skylark/inference/blas/Types.h"
#include "skylark/inference/blas/Utils.h"

namespace fbgemm {

const struct TanhConstants_t {
  float lower_range;
  float upper_range;
  float alpha_13;
  float alpha_11;
  float alpha_9;
  float alpha_7;
  float alpha_5;
  float alpha_3;
  float alpha_1;
  float beta_6;
  float beta_4;
  float beta_2;
  float beta_0;
} TanhConstants = {
    -9.0f,
    9.0f,
    -2.76076847742355e-16f,
    2.00018790482477e-13f,
    -8.60467152213735e-11f,
    5.12229709037114e-08f,
    1.48572235717979e-05f,
    6.37261928875436e-04f,
    4.89352455891786e-03f,
    1.19825839466702e-06f,
    1.18534705686654e-04f,
    2.26843463243900e-03f,
    4.89352518554385e-03f,
};

struct TanhParams {
  uint64_t m;
  uint64_t b_block_cols;
  float *Z;
  uint64_t ldz;
  const float *A; // condition (mutable)
  uint64_t lda;
  const TanhConstants_t *tc = &TanhConstants;
};

void tanh(const TanhParams &gp);

const struct SigmoidConstants_t {
  float lower_range;
  float upper_range;
  float alpha_9;
  float alpha_7;
  float alpha_5;
  float alpha_3;
  float alpha_1;
  float beta_10;
  float beta_8;
  float beta_6;
  float beta_4;
  float beta_2;
  float beta_0;
  float one_half;
} SigmoidConstants = {
    -18.0f,
    18.0f,
    4.37031012579801e-11f,
    1.15627324459942e-07f,
    6.08574864600143e-05f,
    8.51377133304701e-03f,
    2.48287947061529e-01f,
    6.10247389755681e-13f,
    5.76102136993427e-09f,
    6.29106785017040e-06f,
    1.70198817374094e-03f,
    1.16817656904453e-01f,
    9.93151921023180e-01f,
    0.5f,
};

struct SigmoidParams {
  uint64_t m;
  uint64_t b_block_cols;
  float *Z;
  uint64_t ldz;
  const float *A; // condition (mutable)
  uint64_t lda;
  const struct SigmoidConstants_t *sc = &SigmoidConstants;
};

void sigmoid(const SigmoidParams &gp);

struct ReLUParams {
  uint64_t m;
  uint64_t b_block_cols;
  float *Z;
  uint64_t ldz;
  const float *A; // condition (mutable)
  uint64_t lda;
};

void relu(const ReLUParams &gp);

namespace wavernn {

// z = u * h_prev + (1 - u) * e;
// z = u * (h_prev - e) + e;
struct HiddenParams {
  uint64_t m;
  uint64_t b_block_cols;
  float *Z;
  uint64_t ldz;
  const float *A; // u
  uint64_t lda;
  const float *B; // h_prev
  uint64_t ldb;
  const float *C; // e
  uint64_t ldc;
};

void hidden(const HiddenParams &hp);

// z = (c + a + b).sigmoid()
struct CoarseSigmoidParams {
  uint64_t m;
  uint64_t b_block_cols;
  float *Z;
  uint64_t ldz;
  const float *C; // condition (mutable)
  uint64_t ldc;
  const float *A;
  uint64_t lda;
  const float *B;
  uint64_t ldb;
};

void coarse_sigmoid(const CoarseSigmoidParams &cp);

// z = (c + a * b + d).tanh()
struct CoarseTanhParams {
  uint64_t m;
  uint64_t b_block_cols;
  float *Z;
  uint64_t ldz;
  const float *C; // condition (mutable)
  uint64_t ldc;
  const float *A; // Rh
  uint64_t lda;
  const float *B; // r
  uint64_t ldb;
  const float *D; // input
  uint64_t ldd;
};

void coarse_tanh(const CoarseTanhParams &cp);

// z = (c + a + b + d).sigmoid()
struct FineSigmoidParams {
  uint64_t m;
  uint64_t b_block_cols;
  float *Z;
  uint64_t ldz;
  const float *C; // condition (mutable)
  uint64_t ldc;
  const float *A;
  uint64_t lda;
  const float *B;
  uint64_t ldb;
  const float *D; // current
  uint64_t ldd;
};

void fine_sigmoid(const FineSigmoidParams &cp);

// z = (c + a * b + d + t).tanh()
struct FineTanhParams {
  uint64_t m;
  uint64_t b_block_cols;
  float *Z;
  uint64_t ldz;
  const float *C; // condition (mutable)
  uint64_t ldc;
  const float *A;
  uint64_t lda;
  const float *B; // gate r
  uint64_t ldb;
  const float *D;
  uint64_t ldd;
  const float *T; // current
  uint64_t ldt;
};

void fine_tanh(const FineTanhParams &cp);

}; // namespace wavernn

}; // namespace fbgemm
