/*
 * Copyright (c) LAIX, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "skylark/inference/blas/ActivationsFP32.h"
#include "skylark/inference/blas/cpuinfo.h"
#include "skylark/inference/blas/logging.h"

#include "skylark/inference/blas/ActivationsFP32UKernels.h"
#include "skylark/inference/blas/ActivationsFP32UKernelsAvx512.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <utility>
using namespace std;

#define MAXIMUM_MINI_BATCH_SIZE 8

namespace fbgemm {

// class that performs packing of matrix in
// row-major or col-major format into
// internal packed blocked-row major format
typedef void (*funcptr_relu)(ReLUParams *gp);
typedef void (*funcptr_tanh)(TanhParams *gp);
typedef void (*funcptr_sigmoid)(SigmoidParams *gp);

namespace wavernn {
typedef void (*funcptr_coarse_sigmoid_addition)(CoarseSigmoidParams *gp);
typedef void (*funcptr_coarse_tanh_addition)(CoarseTanhParams *gp);
typedef void (*funcptr_fine_sigmoid_addition)(FineSigmoidParams *gp);
typedef void (*funcptr_fine_tanh_addition)(FineTanhParams *gp);
typedef void (*funcptr_hidden)(HiddenParams *gp);
}

// clang-format off
struct ReLUKernelInfo {
  using knl_ptr = funcptr_relu;
  // optimized kernels to cover all cases
  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx256_kernel = {
      {nullptr,
       activationkernel_avx256_relu_1x4,
       activationkernel_avx256_relu_2x4,
       activationkernel_avx256_relu_3x4,
       activationkernel_avx256_relu_4x4,
       activationkernel_avx256_relu_5x4,
       activationkernel_avx256_relu_6x4,
       activationkernel_avx256_relu_7x4,
       activationkernel_avx256_relu_8x4}};

  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx512_kernel = {
      {nullptr,
       activationkernel_avx512_relu_1x4,
       activationkernel_avx512_relu_2x4,
       activationkernel_avx512_relu_3x4,
       activationkernel_avx512_relu_4x4,
       activationkernel_avx512_relu_5x4,
       activationkernel_avx512_relu_6x4,
       activationkernel_avx512_relu_7x4,
       activationkernel_avx512_relu_8x4}};
  // autotuned kernel splits for various cases m = 1:mb_max
  // may need re-autotuning for new uarch
  static constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> partition = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    {
      {{ { 0, 0 }, { 0, 0 } } },
      {{ { 1, 1 }, { 0, 0 } } },
      {{ { 2, 1 }, { 0, 0 } } },
      {{ { 3, 1 }, { 0, 0 } } },
      {{ { 4, 1 }, { 0, 0 } } },
      {{ { 5, 1 }, { 0, 0 } } },
      {{ { 6, 1 }, { 0, 0 } } },
      {{ { 7, 1 }, { 0, 0 } } },
      {{ { 8, 1 }, { 0, 0 } } },
    }
  };
};
constexpr array<ReLUKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> ReLUKernelInfo::avx256_kernel;
constexpr array<ReLUKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> ReLUKernelInfo::avx512_kernel;
constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> ReLUKernelInfo::partition;
// clang-format on

// autotuned kernel splits for various cases m = 1:mb_max
// clang-format off
void relu(const ReLUParams& rp) {
  // constants
  const int m = rp.m;
  const int mb_max = MAXIMUM_MINI_BATCH_SIZE;

  ReLUParams gp(rp);
  if(cpuinfo_has_x86_avx512f())
    gp.b_block_cols = rp.b_block_cols/16;
  else
    gp.b_block_cols = rp.b_block_cols/8;

  gp.ldz = rp.ldz * sizeof(float);
  gp.lda = rp.lda * sizeof(float);

  
  const array<ReLUKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> *kernel; 
  if(cpuinfo_has_x86_avx512f())
    kernel = &(ReLUKernelInfo::avx512_kernel);
  else
    kernel = &(ReLUKernelInfo::avx256_kernel);
  
  for (auto m0 = 0; m0 < m; m0 += mb_max) {
    int mb = std::min(mb_max, m - m0);
    auto m1 = m0;
    for (auto c = 0; c < 2; c++) {
      auto kernel_nrows = ReLUKernelInfo::partition[mb][c][0];
      auto nkernel_nrows = ReLUKernelInfo::partition[mb][c][1];

      auto m_start = m1, m_end = m1 + kernel_nrows * nkernel_nrows;
      for (auto m2 = m_start; m2 < m_end; m2 += kernel_nrows) {
        gp.m = kernel_nrows;
        gp.Z = rp.Z + m2 * rp.ldz;
        gp.A = rp.A + m2 * rp.lda;
        //ReLUKernelInfo::kernel[kernel_nrows](&gp);
        (*kernel)[kernel_nrows](&gp);
      }
      m1 += kernel_nrows * nkernel_nrows;
    }
  }
}

// clang-format off
struct TanhKernelInfo {
  using knl_ptr = funcptr_tanh;
  // optimized kernels to cover all cases
  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx256_kernel = {
      {nullptr,
       activationkernel_avx256_tanh_1x1,
       activationkernel_avx256_tanh_2x1,
       activationkernel_avx256_tanh_3x1,
       activationkernel_avx256_tanh_4x1,
       activationkernel_avx256_tanh_5x1,
       activationkernel_avx256_tanh_6x1,
       activationkernel_avx256_tanh_7x1,
       activationkernel_avx256_tanh_8x1}};

  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx512_kernel = {
      {nullptr,
       activationkernel_avx512_tanh_1x1,
       activationkernel_avx512_tanh_2x1,
       activationkernel_avx512_tanh_3x1,
       activationkernel_avx512_tanh_4x1,
       activationkernel_avx512_tanh_5x1,
       activationkernel_avx512_tanh_6x1,
       activationkernel_avx512_tanh_7x1,
       activationkernel_avx512_tanh_8x1}};
  // autotuned kernel splits for various cases m = 1:mb_max
  // may need re-autotuning for new uarch
  static constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> partition = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    {
      {{ { 0, 0 }, { 0, 0 } } },
      {{ { 1, 1 }, { 0, 0 } } },
      {{ { 2, 1 }, { 0, 0 } } },
      {{ { 3, 1 }, { 0, 0 } } },
      {{ { 4, 1 }, { 0, 0 } } },
      {{ { 5, 1 }, { 0, 0 } } },
      {{ { 6, 1 }, { 0, 0 } } },
      {{ { 7, 1 }, { 0, 0 } } },
      {{ { 8, 1 }, { 0, 0 } } },
    }
  };
};
constexpr array<TanhKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> TanhKernelInfo::avx256_kernel;
constexpr array<TanhKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> TanhKernelInfo::avx512_kernel;
constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> TanhKernelInfo::partition;
// clang-format on

// autotuned kernel splits for various cases m = 1:mb_max
// clang-format off
void tanh(const TanhParams& rp) {
  // constants
  const int m = rp.m;
  const int mb_max = MAXIMUM_MINI_BATCH_SIZE;

  TanhParams gp(rp);
  if(cpuinfo_has_x86_avx512f())
    gp.b_block_cols = rp.b_block_cols/16;
  else
    gp.b_block_cols = rp.b_block_cols/8;
 
  gp.ldz = rp.ldz * sizeof(float);
  gp.lda = rp.lda * sizeof(float);

  const array<TanhKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> *kernel;
  if(cpuinfo_has_x86_avx512f())
    kernel = &(TanhKernelInfo::avx512_kernel);
  else
    kernel = &(TanhKernelInfo::avx256_kernel);

  for (auto m0 = 0; m0 < m; m0 += mb_max) {
    int mb = std::min(mb_max, m - m0);
    auto m1 = m0;
    for (auto c = 0; c < 2; c++) {
      auto kernel_nrows = TanhKernelInfo::partition[mb][c][0];
      auto nkernel_nrows = TanhKernelInfo::partition[mb][c][1];

      auto m_start = m1, m_end = m1 + kernel_nrows * nkernel_nrows;
      for (auto m2 = m_start; m2 < m_end; m2 += kernel_nrows) {
        gp.m = kernel_nrows;
        gp.Z = rp.Z + m2 * rp.ldz;
        gp.A = rp.A + m2 * rp.lda;
        //TanhKernelInfo::kernel[kernel_nrows](&gp);
        (*kernel)[kernel_nrows](&gp);
      }
      m1 += kernel_nrows * nkernel_nrows;
    }
  }
}

// clang-format off
struct SigmoidKernelInfo {
  using knl_ptr = funcptr_sigmoid;
  // optimized kernels to cover all cases
  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx256_kernel = {
      {nullptr,
       activationkernel_avx256_sigmoid_1x1,
       activationkernel_avx256_sigmoid_2x1,
       activationkernel_avx256_sigmoid_3x1,
       activationkernel_avx256_sigmoid_4x1,
       activationkernel_avx256_sigmoid_5x1,
       activationkernel_avx256_sigmoid_6x1,
       activationkernel_avx256_sigmoid_7x1,
       activationkernel_avx256_sigmoid_8x1}};

  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx512_kernel = {
      {nullptr,
       activationkernel_avx512_sigmoid_1x1,
       activationkernel_avx512_sigmoid_2x1,
       activationkernel_avx512_sigmoid_3x1,
       activationkernel_avx512_sigmoid_4x1,
       activationkernel_avx512_sigmoid_5x1,
       activationkernel_avx512_sigmoid_6x1,
       activationkernel_avx512_sigmoid_7x1,
       activationkernel_avx512_sigmoid_8x1}};
  // autotuned kernel splits for various cases m = 1:mb_max
  // may need re-autotuning for new uarch
  static constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> partition = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    {
      {{ { 0, 0 }, { 0, 0 } } },
      {{ { 1, 1 }, { 0, 0 } } },
      {{ { 2, 1 }, { 0, 0 } } },
      {{ { 3, 1 }, { 0, 0 } } },
      {{ { 4, 1 }, { 0, 0 } } },
      {{ { 5, 1 }, { 0, 0 } } },
      {{ { 6, 1 }, { 0, 0 } } },
      {{ { 7, 1 }, { 0, 0 } } },
      {{ { 8, 1 }, { 0, 0 } } },
    }
  };
};
constexpr array<SigmoidKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> SigmoidKernelInfo::avx256_kernel;
constexpr array<SigmoidKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> SigmoidKernelInfo::avx512_kernel;
constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> SigmoidKernelInfo::partition;
// clang-format on

// autotuned kernel splits for various cases m = 1:mb_max
void sigmoid(const SigmoidParams &rp) {
  // constants
  const int m = rp.m;
  const int mb_max = MAXIMUM_MINI_BATCH_SIZE;

  SigmoidParams gp(rp);
  if (cpuinfo_has_x86_avx512f())
    gp.b_block_cols = rp.b_block_cols / 16;
  else
    gp.b_block_cols = rp.b_block_cols / 8;
  gp.ldz = rp.ldz * sizeof(float);
  gp.lda = rp.lda * sizeof(float);

  const array<SigmoidKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> *kernel;
  if (cpuinfo_has_x86_avx512f())
    kernel = &(SigmoidKernelInfo::avx512_kernel);
  else
    kernel = &(SigmoidKernelInfo::avx256_kernel);

  for (auto m0 = 0; m0 < m; m0 += mb_max) {
    int mb = std::min(mb_max, m - m0);
    auto m1 = m0;
    for (auto c = 0; c < 2; c++) {
      auto kernel_nrows = SigmoidKernelInfo::partition[mb][c][0];
      auto nkernel_nrows = SigmoidKernelInfo::partition[mb][c][1];

      auto m_start = m1, m_end = m1 + kernel_nrows * nkernel_nrows;
      for (auto m2 = m_start; m2 < m_end; m2 += kernel_nrows) {
        gp.m = kernel_nrows;
        gp.Z = rp.Z + m2 * rp.ldz;
        gp.A = rp.A + m2 * rp.lda;
        // SigmoidKernelInfo::kernel[kernel_nrows](&gp);
        (*kernel)[kernel_nrows](&gp);
      }
      m1 += kernel_nrows * nkernel_nrows;
    }
  }
}

namespace wavernn {

// clang-format off
struct HiddenKernelInfo {
  using knl_ptr = funcptr_hidden;
  // optimized kernels to cover all cases
  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx256_kernel = {
      {nullptr,
       wavernnkernel_avx256_hidden_1x4,
       wavernnkernel_avx256_hidden_2x4,
       wavernnkernel_avx256_hidden_3x4,
       wavernnkernel_avx256_hidden_4x4,
       wavernnkernel_avx256_hidden_5x4,
       wavernnkernel_avx256_hidden_6x4,
       wavernnkernel_avx256_hidden_7x4,
       wavernnkernel_avx256_hidden_8x4}};

  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx512_kernel = {
      {nullptr,
       wavernnkernel_avx512_hidden_1x4,
       wavernnkernel_avx512_hidden_2x4,
       wavernnkernel_avx512_hidden_3x4,
       wavernnkernel_avx512_hidden_4x4,
       wavernnkernel_avx512_hidden_5x4,
       wavernnkernel_avx512_hidden_6x4,
       wavernnkernel_avx512_hidden_7x4,
       wavernnkernel_avx512_hidden_8x4}};
  // autotuned kernel splits for various cases m = 1:mb_max
  // may need re-autotuning for new uarch
  static constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> partition = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    {
      {{ { 0, 0 }, { 0, 0 } } },
      {{ { 1, 1 }, { 0, 0 } } },
      {{ { 2, 1 }, { 0, 0 } } },
      {{ { 3, 1 }, { 0, 0 } } },
      {{ { 4, 1 }, { 0, 0 } } },
      {{ { 5, 1 }, { 0, 0 } } },
      {{ { 6, 1 }, { 0, 0 } } },
      {{ { 7, 1 }, { 0, 0 } } },
      {{ { 8, 1 }, { 0, 0 } } },
    }
  };
};
constexpr array<HiddenKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> HiddenKernelInfo::avx256_kernel;
constexpr array<HiddenKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> HiddenKernelInfo::avx512_kernel;
constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> HiddenKernelInfo::partition;
// clang-format on

// autotuned kernel splits for various cases m = 1:mb_max
void hidden(const HiddenParams &rp) {
  // constants
  const int m = rp.m;
  const int mb_max = MAXIMUM_MINI_BATCH_SIZE;

  HiddenParams gp(rp);
  if (cpuinfo_has_x86_avx512f())
    gp.b_block_cols = rp.b_block_cols / 16;
  else
    gp.b_block_cols = rp.b_block_cols / 8;

  gp.ldz = rp.ldz * sizeof(float);
  gp.lda = rp.lda * sizeof(float);
  gp.ldb = rp.ldb * sizeof(float);
  gp.ldc = rp.ldc * sizeof(float);

  const array<HiddenKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> *kernel;
  if (cpuinfo_has_x86_avx512f())
    kernel = &(HiddenKernelInfo::avx512_kernel);
  else
    kernel = &(HiddenKernelInfo::avx256_kernel);

  for (auto m0 = 0; m0 < m; m0 += mb_max) {
    int mb = std::min(mb_max, m - m0);
    auto m1 = m0;
    for (auto c = 0; c < 2; c++) {
      auto kernel_nrows = HiddenKernelInfo::partition[mb][c][0];
      auto nkernel_nrows = HiddenKernelInfo::partition[mb][c][1];

      auto m_start = m1, m_end = m1 + kernel_nrows * nkernel_nrows;
      for (auto m2 = m_start; m2 < m_end; m2 += kernel_nrows) {
        gp.m = kernel_nrows;
        gp.Z = rp.Z + m2 * rp.ldz;
        gp.A = rp.A + m2 * rp.lda;
        gp.B = rp.B + m2 * rp.ldb;
        gp.C = rp.C + m2 * rp.ldc;
        // HiddenKernelInfo::kernel[kernel_nrows](&gp);
        (*kernel)[kernel_nrows](&gp);
      }
      m1 += kernel_nrows * nkernel_nrows;
    }
  }
}

// clang-format off
struct CoarseSigmoidKernelInfo {
  using knl_ptr = funcptr_coarse_sigmoid_addition;
  // optimized kernels to cover all cases
  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx256_kernel = {
      {nullptr,
       wavernnkernel_avx256_coarse_sigmoid_addition_1x4,
       wavernnkernel_avx256_coarse_sigmoid_addition_2x4,
       wavernnkernel_avx256_coarse_sigmoid_addition_3x4,
       wavernnkernel_avx256_coarse_sigmoid_addition_4x4,
       wavernnkernel_avx256_coarse_sigmoid_addition_5x4,
       wavernnkernel_avx256_coarse_sigmoid_addition_6x4,
       wavernnkernel_avx256_coarse_sigmoid_addition_7x4,
       wavernnkernel_avx256_coarse_sigmoid_addition_8x4}};

  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx512_kernel = {
      {nullptr,
       wavernnkernel_avx512_coarse_sigmoid_addition_1x4,
       wavernnkernel_avx512_coarse_sigmoid_addition_2x4,
       wavernnkernel_avx512_coarse_sigmoid_addition_3x4,
       wavernnkernel_avx512_coarse_sigmoid_addition_4x4,
       wavernnkernel_avx512_coarse_sigmoid_addition_5x4,
       wavernnkernel_avx512_coarse_sigmoid_addition_6x4,
       wavernnkernel_avx512_coarse_sigmoid_addition_7x4,
       wavernnkernel_avx512_coarse_sigmoid_addition_8x4}};
  // autotuned kernel splits for various cases m = 1:mb_max
  // may need re-autotuning for new uarch
  static constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> partition = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    {
      {{ { 0, 0 }, { 0, 0 } } },
      {{ { 1, 1 }, { 0, 0 } } },
      {{ { 2, 1 }, { 0, 0 } } },
      {{ { 3, 1 }, { 0, 0 } } },
      {{ { 4, 1 }, { 0, 0 } } },
      {{ { 5, 1 }, { 0, 0 } } },
      {{ { 6, 1 }, { 0, 0 } } },
      {{ { 7, 1 }, { 0, 0 } } },
      {{ { 8, 1 }, { 0, 0 } } },
    }
  };
};
constexpr array<CoarseSigmoidKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> CoarseSigmoidKernelInfo::avx256_kernel;
constexpr array<CoarseSigmoidKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> CoarseSigmoidKernelInfo::avx512_kernel;
constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> CoarseSigmoidKernelInfo::partition;
// clang-format on

// autotuned kernel splits for various cases m = 1:mb_max
void coarse_sigmoid(const CoarseSigmoidParams &rp) {
  // constants
  const int m = rp.m;
  const int mb_max = MAXIMUM_MINI_BATCH_SIZE;

  CoarseSigmoidParams gp(rp);
  if (cpuinfo_has_x86_avx512f())
    gp.b_block_cols = rp.b_block_cols / 16;
  else
    gp.b_block_cols = rp.b_block_cols / 8;

  gp.ldz = rp.ldz * sizeof(float);
  gp.lda = rp.lda * sizeof(float);
  gp.ldb = rp.ldb * sizeof(float);
  gp.ldc = rp.ldc * sizeof(float);

  const array<CoarseSigmoidKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1>
      *kernel;
  if (cpuinfo_has_x86_avx512f())
    kernel = &(CoarseSigmoidKernelInfo::avx512_kernel);
  else
    kernel = &(CoarseSigmoidKernelInfo::avx256_kernel);

  for (auto m0 = 0; m0 < m; m0 += mb_max) {
    int mb = std::min(mb_max, m - m0);
    auto m1 = m0;
    for (auto c = 0; c < 2; c++) {
      auto kernel_nrows = CoarseSigmoidKernelInfo::partition[mb][c][0];
      auto nkernel_nrows = CoarseSigmoidKernelInfo::partition[mb][c][1];

      auto m_start = m1, m_end = m1 + kernel_nrows * nkernel_nrows;
      for (auto m2 = m_start; m2 < m_end; m2 += kernel_nrows) {
        gp.m = kernel_nrows;
        gp.Z = rp.Z + m2 * rp.ldz;
        gp.A = rp.A + m2 * rp.lda;
        gp.B = rp.B + m2 * rp.ldb;
        gp.C = rp.C + m2 * rp.ldc;

        // CoarseSigmoidKernelInfo::kernel[kernel_nrows](&gp);
        (*kernel)[kernel_nrows](&gp);
      }
      m1 += kernel_nrows * nkernel_nrows;
    }
  }

  // Sigmoid
  SigmoidParams sp;
  sp.m = rp.m;
  sp.b_block_cols = rp.b_block_cols;
  sp.Z = rp.Z;
  sp.ldz = rp.ldz;
  sp.A = rp.Z;
  sp.lda = rp.ldz;
  sigmoid(sp);
}

// clang-format off
struct CoarseTanhKernelInfo {
  using knl_ptr = funcptr_coarse_tanh_addition;
  // optimized kernels to cover all cases
  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx256_kernel = {
      {nullptr,
       wavernnkernel_avx256_coarse_tanh_addition_1x4,
       wavernnkernel_avx256_coarse_tanh_addition_2x4,
       wavernnkernel_avx256_coarse_tanh_addition_3x4,
       wavernnkernel_avx256_coarse_tanh_addition_4x4,
       wavernnkernel_avx256_coarse_tanh_addition_5x4,
       wavernnkernel_avx256_coarse_tanh_addition_6x4,
       wavernnkernel_avx256_coarse_tanh_addition_7x4,
       wavernnkernel_avx256_coarse_tanh_addition_8x4}};

  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx512_kernel = {
      {nullptr,
       wavernnkernel_avx512_coarse_tanh_addition_1x4,
       wavernnkernel_avx512_coarse_tanh_addition_2x4,
       wavernnkernel_avx512_coarse_tanh_addition_3x4,
       wavernnkernel_avx512_coarse_tanh_addition_4x4,
       wavernnkernel_avx512_coarse_tanh_addition_5x4,
       wavernnkernel_avx512_coarse_tanh_addition_6x4,
       wavernnkernel_avx512_coarse_tanh_addition_7x4,
       wavernnkernel_avx512_coarse_tanh_addition_8x4}};
  // autotuned kernel splits for various cases m = 1:mb_max
  // may need re-autotuning for new uarch
  static constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> partition = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    {
      {{ { 0, 0 }, { 0, 0 } } },
      {{ { 1, 1 }, { 0, 0 } } },
      {{ { 2, 1 }, { 0, 0 } } },
      {{ { 3, 1 }, { 0, 0 } } },
      {{ { 4, 1 }, { 0, 0 } } },
      {{ { 5, 1 }, { 0, 0 } } },
      {{ { 6, 1 }, { 0, 0 } } },
      {{ { 7, 1 }, { 0, 0 } } },
      {{ { 8, 1 }, { 0, 0 } } },
    }
  };
};
constexpr array<CoarseTanhKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> CoarseTanhKernelInfo::avx256_kernel;
constexpr array<CoarseTanhKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> CoarseTanhKernelInfo::avx512_kernel;
constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> CoarseTanhKernelInfo::partition;
// clang-format on

// autotuned kernel splits for various cases m = 1:mb_max
void coarse_tanh(const CoarseTanhParams &rp) {
  // constants
  const int m = rp.m;
  const int mb_max = MAXIMUM_MINI_BATCH_SIZE;

  CoarseTanhParams gp(rp);
  if (cpuinfo_has_x86_avx512f())
    gp.b_block_cols = rp.b_block_cols / 16;
  else
    gp.b_block_cols = rp.b_block_cols / 8;

  gp.ldz = rp.ldz * sizeof(float);
  gp.lda = rp.lda * sizeof(float);
  gp.ldb = rp.ldb * sizeof(float);
  gp.ldc = rp.ldc * sizeof(float);
  gp.ldd = rp.ldd * sizeof(float);

  const array<CoarseTanhKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1>
      *kernel;
  if (cpuinfo_has_x86_avx512f())
    kernel = &(CoarseTanhKernelInfo::avx512_kernel);
  else
    kernel = &(CoarseTanhKernelInfo::avx256_kernel);

  // addition
  for (auto m0 = 0; m0 < m; m0 += mb_max) {
    int mb = std::min(mb_max, m - m0);
    auto m1 = m0;
    for (auto c = 0; c < 2; c++) {
      auto kernel_nrows = CoarseTanhKernelInfo::partition[mb][c][0];
      auto nkernel_nrows = CoarseTanhKernelInfo::partition[mb][c][1];

      auto m_start = m1, m_end = m1 + kernel_nrows * nkernel_nrows;
      for (auto m2 = m_start; m2 < m_end; m2 += kernel_nrows) {
        gp.m = kernel_nrows;
        gp.Z = rp.Z + m2 * rp.ldz;
        gp.A = rp.A + m2 * rp.lda;
        gp.B = rp.B + m2 * rp.ldb;
        gp.C = rp.C + m2 * rp.ldc;
        gp.D = rp.D + m2 * rp.ldd;
        // CoarseTanhKernelInfo::kernel[kernel_nrows](&gp);
        (*kernel)[kernel_nrows](&gp);
      }
      m1 += kernel_nrows * nkernel_nrows;
    }
  }
  // tanh
  TanhParams th;
  th.m = rp.m;
  th.b_block_cols = rp.b_block_cols;
  th.Z = rp.Z;
  th.ldz = rp.ldz;
  th.A = rp.Z;
  th.lda = rp.ldz;
  tanh(th);
}

// clang-format off
struct FineSigmoidKernelInfo {
  using knl_ptr = funcptr_fine_sigmoid_addition;
  // optimized kernels to cover all cases
  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx256_kernel = {
      {nullptr,
       wavernnkernel_avx256_fine_sigmoid_addition_1x4,
       wavernnkernel_avx256_fine_sigmoid_addition_2x4,
       wavernnkernel_avx256_fine_sigmoid_addition_3x4,
       wavernnkernel_avx256_fine_sigmoid_addition_4x4,
       wavernnkernel_avx256_fine_sigmoid_addition_5x4,
       wavernnkernel_avx256_fine_sigmoid_addition_6x4,
       wavernnkernel_avx256_fine_sigmoid_addition_7x4,
       wavernnkernel_avx256_fine_sigmoid_addition_8x4}};

  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx512_kernel = {
      {nullptr,
       wavernnkernel_avx512_fine_sigmoid_addition_1x4,
       wavernnkernel_avx512_fine_sigmoid_addition_2x4,
       wavernnkernel_avx512_fine_sigmoid_addition_3x4,
       wavernnkernel_avx512_fine_sigmoid_addition_4x4,
       wavernnkernel_avx512_fine_sigmoid_addition_5x4,
       wavernnkernel_avx512_fine_sigmoid_addition_6x4,
       wavernnkernel_avx512_fine_sigmoid_addition_7x4,
       wavernnkernel_avx512_fine_sigmoid_addition_8x4}};
  // autotuned kernel splits for various cases m = 1:mb_max
  // may need re-autotuning for new uarch
  static constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> partition = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    {
      {{ { 0, 0 }, { 0, 0 } } },
      {{ { 1, 1 }, { 0, 0 } } },
      {{ { 2, 1 }, { 0, 0 } } },
      {{ { 3, 1 }, { 0, 0 } } },
      {{ { 4, 1 }, { 0, 0 } } },
      {{ { 5, 1 }, { 0, 0 } } },
      {{ { 6, 1 }, { 0, 0 } } },
      {{ { 7, 1 }, { 0, 0 } } },
      {{ { 8, 1 }, { 0, 0 } } },
    }
  };
};
constexpr array<FineSigmoidKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> FineSigmoidKernelInfo::avx256_kernel;
constexpr array<FineSigmoidKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> FineSigmoidKernelInfo::avx512_kernel;
constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> FineSigmoidKernelInfo::partition;
// clang-format on

// autotuned kernel splits for various cases m = 1:mb_max
void fine_sigmoid(const FineSigmoidParams &rp) {
  // constants
  const int m = rp.m;
  const int mb_max = MAXIMUM_MINI_BATCH_SIZE;

  FineSigmoidParams gp(rp);
  if (cpuinfo_has_x86_avx512f())
    gp.b_block_cols = rp.b_block_cols / 16;
  else
    gp.b_block_cols = rp.b_block_cols / 8;

  gp.ldz = rp.ldz * sizeof(float);
  gp.lda = rp.lda * sizeof(float);
  gp.ldb = rp.ldb * sizeof(float);
  gp.ldc = rp.ldc * sizeof(float);
  gp.ldd = rp.ldd * sizeof(float);

  const array<FineSigmoidKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1>
      *kernel;
  if (cpuinfo_has_x86_avx512f())
    kernel = &(FineSigmoidKernelInfo::avx512_kernel);
  else
    kernel = &(FineSigmoidKernelInfo::avx256_kernel);

  for (auto m0 = 0; m0 < m; m0 += mb_max) {
    int mb = std::min(mb_max, m - m0);
    auto m1 = m0;
    for (auto c = 0; c < 2; c++) {
      auto kernel_nrows = FineSigmoidKernelInfo::partition[mb][c][0];
      auto nkernel_nrows = FineSigmoidKernelInfo::partition[mb][c][1];

      auto m_start = m1, m_end = m1 + kernel_nrows * nkernel_nrows;
      for (auto m2 = m_start; m2 < m_end; m2 += kernel_nrows) {
        gp.m = kernel_nrows;
        gp.Z = rp.Z + m2 * rp.ldz;
        gp.A = rp.A + m2 * rp.lda;
        gp.B = rp.B + m2 * rp.ldb;
        gp.C = rp.C + m2 * rp.ldc;
        gp.D = rp.D + m2 * rp.ldd;
        // FineSigmoidKernelInfo::kernel[kernel_nrows](&gp);
        (*kernel)[kernel_nrows](&gp);
      }
      m1 += kernel_nrows * nkernel_nrows;
    }
  }

  // Sigmoid
  SigmoidParams sp;
  sp.m = rp.m;
  sp.b_block_cols = rp.b_block_cols;
  sp.Z = rp.Z;
  sp.ldz = rp.ldz;
  sp.A = rp.Z;
  sp.lda = rp.ldz;
  sigmoid(sp);
}

// clang-format off
struct FineTanhKernelInfo {
  using knl_ptr = funcptr_fine_tanh_addition;
  // optimized kernels to cover all cases
  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx256_kernel = {
      {nullptr,
       wavernnkernel_avx256_fine_tanh_addition_1x4,
       wavernnkernel_avx256_fine_tanh_addition_2x4,
       wavernnkernel_avx256_fine_tanh_addition_3x4,
       wavernnkernel_avx256_fine_tanh_addition_4x4,
       wavernnkernel_avx256_fine_tanh_addition_5x4,
       wavernnkernel_avx256_fine_tanh_addition_6x4,
       wavernnkernel_avx256_fine_tanh_addition_7x4,
       wavernnkernel_avx256_fine_tanh_addition_8x4}};

  static constexpr array<knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> avx512_kernel = {
      {nullptr,
       wavernnkernel_avx512_fine_tanh_addition_1x4,
       wavernnkernel_avx512_fine_tanh_addition_2x4,
       wavernnkernel_avx512_fine_tanh_addition_3x4,
       wavernnkernel_avx512_fine_tanh_addition_4x4,
       wavernnkernel_avx512_fine_tanh_addition_5x4,
       wavernnkernel_avx512_fine_tanh_addition_6x4,
       wavernnkernel_avx512_fine_tanh_addition_7x4,
       wavernnkernel_avx512_fine_tanh_addition_8x4}};
  // autotuned kernel splits for various cases m = 1:mb_max
  // may need re-autotuning for new uarch
  static constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> partition = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    {
      {{ { 0, 0 }, { 0, 0 } } },
      {{ { 1, 1 }, { 0, 0 } } },
      {{ { 2, 1 }, { 0, 0 } } },
      {{ { 3, 1 }, { 0, 0 } } },
      {{ { 4, 1 }, { 0, 0 } } },
      {{ { 5, 1 }, { 0, 0 } } },
      {{ { 6, 1 }, { 0, 0 } } },
      {{ { 7, 1 }, { 0, 0 } } },
      {{ { 8, 1 }, { 0, 0 } } },
    }
  };
};
constexpr array<FineTanhKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> FineTanhKernelInfo::avx256_kernel;
constexpr array<FineTanhKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> FineTanhKernelInfo::avx512_kernel;
constexpr array<array<array<int, 2>, 2>, MAXIMUM_MINI_BATCH_SIZE + 1> FineTanhKernelInfo::partition;
// clang-format on

// autotuned kernel splits for various cases m = 1:mb_max
void fine_tanh(const FineTanhParams &rp) {
  // constants
  const int m = rp.m;
  const int mb_max = MAXIMUM_MINI_BATCH_SIZE;

  FineTanhParams gp(rp);
  if (cpuinfo_has_x86_avx512f())
    gp.b_block_cols = rp.b_block_cols / 16;
  else
    gp.b_block_cols = rp.b_block_cols / 8;

  gp.ldz = rp.ldz * sizeof(float);
  gp.lda = rp.lda * sizeof(float);
  gp.ldb = rp.ldb * sizeof(float);
  gp.ldc = rp.ldc * sizeof(float);
  gp.ldd = rp.ldd * sizeof(float);
  gp.ldt = rp.ldt * sizeof(float);

  const array<FineTanhKernelInfo::knl_ptr, MAXIMUM_MINI_BATCH_SIZE + 1> *kernel;
  if (cpuinfo_has_x86_avx512f())
    kernel = &(FineTanhKernelInfo::avx512_kernel);
  else
    kernel = &(FineTanhKernelInfo::avx256_kernel);

  for (auto m0 = 0; m0 < m; m0 += mb_max) {
    int mb = std::min(mb_max, m - m0);
    auto m1 = m0;
    for (auto c = 0; c < 2; c++) {
      auto kernel_nrows = FineTanhKernelInfo::partition[mb][c][0];
      auto nkernel_nrows = FineTanhKernelInfo::partition[mb][c][1];

      auto m_start = m1, m_end = m1 + kernel_nrows * nkernel_nrows;
      for (auto m2 = m_start; m2 < m_end; m2 += kernel_nrows) {
        gp.m = kernel_nrows;
        gp.Z = rp.Z + m2 * rp.ldz;
        gp.A = rp.A + m2 * rp.lda;
        gp.B = rp.B + m2 * rp.ldb;
        gp.C = rp.C + m2 * rp.ldc;
        gp.D = rp.D + m2 * rp.ldd;
        gp.T = rp.T + m2 * rp.ldt;
        // FineTanhKernelInfo::kernel[kernel_nrows](&gp);
        (*kernel)[kernel_nrows](&gp);
      }
      m1 += kernel_nrows * nkernel_nrows;
    }
  }
  // tanh
  TanhParams th;
  th.m = rp.m;
  th.b_block_cols = rp.b_block_cols;
  th.Z = rp.Z;
  th.ldz = rp.ldz;
  th.A = rp.Z;
  th.lda = rp.ldz;
  tanh(th);
}

} // namespace wavernn

} // namespace fbgemm
