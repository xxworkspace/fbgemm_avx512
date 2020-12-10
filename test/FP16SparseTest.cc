/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <random>

#include <gtest/gtest.h>

#include "skylark/inference/blas/FbgemmFP16Sparse.h"
#include "skylark/inference/blas/RefImplementations.h"
#include "skylark/inference/blas/bench/BenchUtils.h"
#include "skylark/inference/blas/test/TestUtils.h"
#include "tbb/tbb.h"

using namespace std;
using namespace fbgemm;

namespace {
// The template parameter is transpose of A and B
class FBSparseGemmFP16Test
    : public testing::TestWithParam<pair<matrix_op_t, matrix_op_t>> {};
};  // namespace

INSTANTIATE_TEST_CASE_P(InstantiationName, FBSparseGemmFP16Test,
                        ::testing::Values(pair<matrix_op_t, matrix_op_t>(
                            matrix_op_t::NoTranspose,
                            matrix_op_t::NoTranspose)));

TEST_P(FBSparseGemmFP16Test, Test) {
  vector<vector<int>> shapes;
  random_device r;
  default_random_engine generator(r());
  uniform_int_distribution<int> dm(1, 32);  // 32
  uniform_int_distribution<int> dn(64, 2048);
  uniform_int_distribution<int> dk(16, 128);

  for (int i = 0; i < 1; i++) {
    int m = dm(generator);
    int n = (dn(generator) / 4) * 4;
    int k = (dk(generator) / 16) * 16;
    shapes.push_back({m, n, k});
  }

  float alpha = 1.f, beta = 1.f;
  matrix_op_t atrans, btrans;
  tie(atrans, btrans) = GetParam();

  vector<vector<int>> blocks = {{16, 1}, {8, 1}};
  for (auto s : shapes) {
    for (auto bs : blocks) {
      int block_height = bs[0];
      int block_width = bs[1];

      int m = s[0];
      int n = s[1] * block_width;
      int k = s[2] * block_height;

      cerr << "m = " << m << " n = " << n << " k = " << k;
      if (atrans == matrix_op_t::Transpose) {
        cerr << " A_transposed";
      }
      if (btrans == matrix_op_t::Transpose) {
        cerr << " B_transposed";
      }
      cerr << endl;

      // initialize with small numbers
      aligned_vector<int> Aint(m * k);
      aligned_vector<int> Bint(k * n);
      randFill(Aint, 0, 4);
      randFill(Bint, 0, 4);
      aligned_vector<float> A(Aint.begin(), Aint.end());
      aligned_vector<float> B(Bint.begin(), Bint.end());

      // generate mask and update B
      aligned_vector<int> mask(s[2] * s[1]);
      {
        uniform_int_distribution<int> r(1, 10);
        for (auto i = 0; i < s[2]; i++) {
          for (auto j = 0; j < s[1]; j++) {
            int dense = r(generator) > 8;
            mask[i * s[1] + j] = dense;
            if (dense == 0) {
              int r = i * block_height;
              int c = j * block_width;
              for (auto rx = r; rx < r + block_height; rx++) {
                for (auto cx = c; cx < c + block_width; cx++) {
                  B[rx * n + cx] = 0.f;
                }
              }
            }
          }
        }
      }

      aligned_vector<float> C(m * n, 0.f);
      aligned_vector<float> Cg(m * n, 0.f);

      aligned_vector<float> A_ref(A), B_ref(B), C_ref(C);

      if (atrans == matrix_op_t::Transpose) {
        transpose_matrix(A_ref.data(), k, m);
      }
      if (btrans == matrix_op_t::Transpose) {
        transpose_matrix(B_ref.data(), n, k);
      }

      // Gold via reference sgemm
      matmul_fp_ref(m, n, k, k, n, n, A_ref.data(), B_ref.data(), C_ref.data());
      // fbgemm fp16
      PackedGemmBlockSparseMatrixFP16 Bp(btrans, alpha, B.data(), mask.data(),
                                         s[2], s[1], block_height, block_width);
      cblas_gemm_compute(atrans, m, A.data(), Bp, beta, C.data());

      const int ngroups = 4;
      PackedGemmBlockSparseMatrixFP16 Bg(btrans, alpha, B.data(), mask.data(),
                                         s[2], s[1], block_height, block_width,
                                         ngroups);

      const int shift = (ngroups - 1) * Bg.groupNumCols();
      tbb::parallel_for(size_t(0), size_t(ngroups),
                        [m, &A, &Bg, &beta, &Cg, &shift](size_t t) {
                          auto cdata = Cg.data() + t * Bg.groupNumCols();
                          PackedGemmBlockSparseMatrixFP16 Bgg(Bg, t);
                          cblas_gemm_compute(matrix_op_t::NoTranspose, m,
                                             A.data(), Bgg, beta, cdata, shift);
                          printf("JOB %zu DONE.\n", t);
                        });

      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          float expected = C_ref[i * n + j];
          float actual = C[i * n + j];
          EXPECT_EQ(expected, actual)
              << "GEMM results differ at (" << i << ", " << j << "). ref "
              << expected << " FBGemm " << actual;
          float tfresult = Cg[i * n + j];
          EXPECT_EQ(expected, tfresult)
              << "GEMM results differ at (" << i << ", " << j << "). ref "
              << expected << " ThreadFBGemm " << tfresult;
        }
      }
    }
  }
}
