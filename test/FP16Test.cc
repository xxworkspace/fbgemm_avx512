/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>
#include <array>
#include <numeric>
#include <random>
#include <string>

#include "skylark/inference/blas/FbgemmFP16.h"
#include "skylark/inference/blas/RefImplementations.h"
#include "skylark/inference/blas/bench/BenchUtils.h"
#include "skylark/inference/blas/test/TestUtils.h"
#include "tbb/tbb.h"

#ifdef USE_IACA
#include "iacaMarks.h"
#endif

using namespace std;
using namespace fbgemm;

namespace {
// The template parameter is transpose of A and B
class FBGemmFP16Test
    : public testing::TestWithParam<pair<matrix_op_t, matrix_op_t>> {};
};  // namespace

INSTANTIATE_TEST_CASE_P(InstantiationName, FBGemmFP16Test,
                        ::testing::Values(pair<matrix_op_t, matrix_op_t>(
                            matrix_op_t::NoTranspose,
                            matrix_op_t::NoTranspose) /*,
      pair<matrix_op_t, matrix_op_t>(
          matrix_op_t::NoTranspose, matrix_op_t::Transpose),
      pair<matrix_op_t, matrix_op_t>(
          matrix_op_t::Transpose, matrix_op_t::NoTranspose),
      pair<matrix_op_t, matrix_op_t>(
          matrix_op_t::Transpose, matrix_op_t::Transpose)*/));

TEST_P(FBGemmFP16Test, Test) {
  vector<vector<int>> shapes;
  random_device r;
  const int ngroups = 4;

  default_random_engine generator(r());
  uniform_int_distribution<int> dm(1, 256);
  uniform_int_distribution<int> dnk(32, 256);

  for (int i = 1; i <= 14; i++) {
    shapes.push_back({i, ngroups * 32, 16});
  }
  for (int i = 1; i < 16; i++) {
    int m = dm(generator);
    int n = (dnk(generator) / 32) * 32 * ngroups;
    int k = (dnk(generator) / 16) * 16;
    shapes.push_back({m, n, k});
    if (m > 10) {
      shapes.push_back({(m / 10) * 10, n, k});
    }
  }

  float alpha = 1.f;
  matrix_op_t atrans, btrans;
  tie(atrans, btrans) = GetParam();

  for (auto s : shapes) {
    for (float beta : {0.f, 1.f}) {
      int m = s[0];
      int n = s[1];
      int k = s[2];

      cerr << "m = " << m << " n = " << n << " k = " << k << " beta = " << beta;
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

      aligned_vector<float> C(m * n, 0.f);
      if (beta != 0.0f) {
        randFill(C, 0.f, 4.f);
      }

      aligned_vector<float> Cg(C);
      aligned_vector<float> Cpre(C);

      aligned_vector<float> A_ref(A), B_ref(B), C_ref(C);

      if (atrans == matrix_op_t::Transpose) {
        transpose_matrix(A_ref.data(), k, m);
      }
      if (btrans == matrix_op_t::Transpose) {
        transpose_matrix(B_ref.data(), n, k);
      }

      // Gold via reference sgemm
      matmul_fp_ref(m, n, k, k, n, n, A_ref.data(), B_ref.data(), C_ref.data(),
                    beta);

      // fbgemm fp16
      PackedGemmMatrixFP16 Bp(btrans, k, n, alpha, B.data(), k);
      cblas_gemm_compute(atrans, m, A.data(), Bp, beta, C.data());
      // parallel jobs
      std::vector<int> group_sizes;
      for (int i = 0; i < ngroups; ++i) {
        group_sizes.push_back(n / ngroups);
      }

      PackedGemmMatrixFP16 Bg(btrans, k, n, alpha, B.data(), k, group_sizes);

      static thread_local unique_ptr<std::array<float, 256 * 1024>> scratchpad(
          new std::array<float, 256 * 1024>());
      auto buffer = scratchpad->data();
      PackAMatrix packedA(matrix_op_t::NoTranspose, m, k, A.data(), k, buffer,
                          Bg.blockRowSize());
      tbb::parallel_for(
          size_t(0), size_t(ngroups), [&packedA, &Bg, &beta, &Cg](size_t t) {
            auto cdata = Cg.data() + Bg.groupBeginCol(t);
            PackedGemmMatrixFP16 Bgg(Bg, t);
            cblas_gemm_compute(packedA, Bgg, beta, cdata, Bg.groupShift(t));
            printf("JOB %zu DONE.\n", t);
          });

      PackedGemmMatrixFP16 Bpre(
          btrans, k, n,
          reinterpret_cast<uint16*>(const_cast<float16*>((Bg.data()))), k,
          ngroups);
      tbb::parallel_for(size_t(0), size_t(ngroups),
                        [&packedA, &Bpre, &beta, &Cpre](size_t t) {
                          auto cdata = Cpre.data() + Bpre.groupBeginCol(t);
                          PackedGemmMatrixFP16 Bgg(Bpre, t);
                          cblas_gemm_compute(packedA, Bgg, beta, cdata,
                                             Bpre.groupShift(t));
                          printf("Pre JOB %zu DONE.\n", t);
                        });

      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          float expected = C_ref[i * n + j];
          float actual = C[i * n + j];
          float tfresult = Cg[i * n + j];
          // printf("[%d, %d]: ref %f gemm %f thread %f\n", i, j, expected, actual, tfresult);
          EXPECT_EQ(expected, actual)
              << "GEMM results differ at (" << i << ", " << j << "). ref "
              << expected << " FBGemm " << actual;

          EXPECT_EQ(expected, tfresult)
              << "GEMM results differ at (" << i << ", " << j << "). ref "
              << expected << " ThreadFBGemm " << tfresult;
          float prec = Cpre[i * n + j];
          EXPECT_EQ(expected, prec)
              << "GEMM results differ at (" << i << ", " << j << "). ref "
              << expected << " PreBThreadFBGemm " << prec;
        }
      }
    }
  }
}
