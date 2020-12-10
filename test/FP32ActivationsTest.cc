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

#include "skylark/inference/blas/ActivationsFP32.h"
#include "skylark/inference/blas/RefImplementations.h"
#include "skylark/inference/blas/bench/BenchUtils.h"
#include "skylark/inference/blas/test/TestUtils.h"

using namespace std;
using namespace fbgemm;

namespace {
class FB32ActivationsFPTest
    : public testing::TestWithParam<pair<matrix_op_t, matrix_op_t>> {};
};  // namespace

INSTANTIATE_TEST_CASE_P(InstantiationName, FB32ActivationsFPTest,
                        ::testing::Values(pair<matrix_op_t, matrix_op_t>(
                            matrix_op_t::NoTranspose,
                            matrix_op_t::NoTranspose)));

TEST_P(FB32ActivationsFPTest, Test) {
  vector<vector<int>> shapes;
  random_device r;

  default_random_engine generator(r());
  uniform_int_distribution<int> dm(1, 256);

  const int block_size = 64;
  uniform_int_distribution<int> dn(block_size, 512);

  for (int i = 1; i <= 14; i++) {
    shapes.push_back({i, 4 * block_size});
  }

  const float epsilon = 1e-4;
  for (int i = 1; i < 8; i++) {
    int m = dm(generator);
    int n = (dn(generator) / block_size) * block_size;
    shapes.push_back({m, n});
  }

  for (auto s : shapes) {
    int m = s[0];
    int n = s[1];

    cerr << "m = " << m << " n = " << n << endl;
    aligned_vector<float> A(m * n);
    randFill(A, -8.f, 8.f);
    aligned_vector<float> Ac(A);
    const int ldb = n * 4;
    aligned_vector<float> B(m * ldb);
    randFill(B, -8.f, 8.f);
    const int ldd = n * 2;
    aligned_vector<float> D(m * ldd);
    randFill(D, -8.f, 8.f);
    aligned_vector<float> C(m * n);
    randFill(C, -8.f, 8.f);
    const int ldt = n * 3;
    aligned_vector<float> T(m * ldt);
    randFill(T, -8.f, 8.f);

    const int ldz = n * 6;
    aligned_vector<float> Z_avx(m * ldz);
    randFill(Z_avx, -8.f, 8.f);

    aligned_vector<float> Z_ref(m * ldz);

    // tanh
    {
      TanhParams rp;
      rp.m = m;
      rp.b_block_cols = n;
      rp.Z = Z_avx.data();
      rp.ldz = ldz;
      rp.A = A.data();
      rp.lda = n;
      tanh(rp);

      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          float expected = std::tanh(A[i * n + j]);
          float actual = Z_avx[i * ldz + j];
          EXPECT_NEAR(expected, actual, 1e-4)
              << "Tanh results differ at (" << i << ", " << j << "). ref "
              << expected << " avx " << actual;
        }
      }
    }

    // sigmoid
    {
      SigmoidParams rp;
      rp.m = m;
      rp.b_block_cols = n;
      rp.Z = Z_avx.data();
      rp.ldz = ldz;
      rp.A = A.data();
      rp.lda = n;
      sigmoid(rp);

      rp.Z = Ac.data();
      rp.ldz = n;
      rp.A = Ac.data();
      rp.lda = n;
      sigmoid(rp);
      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          float expected = 1.0 / (1.0 + std::exp(-A[i * n + j]));
          float actual = Z_avx[i * ldz + j];
          float sameza = Ac[i * n + j];
          EXPECT_EQ(actual, sameza)
              << "Sigmoid results differ at (" << i << ", " << j << "). ref "
              << expected << " avx " << actual << " sameZA " << sameza;
          EXPECT_NEAR(expected, actual, 1e-4)
              << "Sigmoid results differ at (" << i << ", " << j << "). ref "
              << expected << " avx " << actual << " sameZA " << sameza;
        }
      }
    }

    // relu
    {
      aligned_vector<float> relu_A(m * n);
      randFill(relu_A, -8.f, 8.f);
      aligned_vector<float> relu_ref(m * n);
      aligned_vector<float> relu_avx(m * n);
      randFill(relu_avx, -1.f, -1.f);
      relu(m, n, relu_ref.data(), n, relu_A.data(), n);

      ReLUParams rp;
      rp.m = m;
      rp.b_block_cols = n;
      rp.Z = relu_avx.data();
      rp.ldz = n;
      rp.A = relu_A.data();
      rp.lda = n;
      relu(rp);

      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          float real = relu_A[i * n + j];
          float expected = relu_ref[i * n + j];
          float actual = relu_avx[i * n + j];
          EXPECT_EQ(expected, actual)
              << "ReLU results differ at (" << i << ", " << j << "). ref "
              << expected << " avx " << actual << " real " << real;
        }
      }
    }

    // coarse sigmoid u = (x + y + z).sigmoid()
    {
      wavernn::coarse_sigmoid(m, n, Z_ref.data(), ldz, C.data(), n, A.data(), n,
                              B.data(), ldb);

      wavernn::CoarseSigmoidParams rp;
      rp.m = m;
      rp.b_block_cols = n;
      rp.Z = Z_avx.data();
      rp.ldz = ldz;
      rp.C = C.data();
      rp.ldc = n;
      rp.A = A.data();
      rp.lda = n;
      rp.B = B.data();
      rp.ldb = ldb;
      wavernn::coarse_sigmoid(rp);

      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          float expected = Z_ref[i * ldz + j];
          float actual = Z_avx[i * ldz + j];
          // float actual = 1.0 / (1.0 + exp(-Z_avx[i * ldz + j]));
          EXPECT_NEAR(expected, actual, epsilon)
              << "CoarseSigmoid results differ at (" << i << ", " << j
              << "). ref " << expected << " avx " << actual;
        }
      }
    }

    // coarse tanh e = (x + y * r + z).tanh()
    {
      wavernn::coarse_tanh(m, n, Z_ref.data(), ldz, C.data(), n, A.data(), n,
                           B.data(), ldb, D.data(), ldd);

      wavernn::CoarseTanhParams rp;
      rp.m = m;
      rp.b_block_cols = n;
      rp.Z = Z_avx.data();
      rp.ldz = ldz;
      rp.C = C.data();
      rp.ldc = n;
      rp.A = A.data();
      rp.lda = n;
      rp.B = B.data();
      rp.ldb = ldb;
      rp.D = D.data();
      rp.ldd = ldd;
      wavernn::coarse_tanh(rp);

      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          float expected = Z_ref[i * ldz + j];
          float actual = Z_avx[i * ldz + j];
          EXPECT_NEAR(expected, actual, epsilon)
              << "CoarseTanh results differ at (" << i << ", " << j << "). ref "
              << expected << " avx " << actual;
        }
      }
    }

    // fine sigmoid
    {
      wavernn::fine_sigmoid(m, n, Z_ref.data(), ldz, C.data(), n, A.data(), n,
                            B.data(), ldb, D.data(), ldd);

      wavernn::FineSigmoidParams rp;
      rp.m = m;
      rp.b_block_cols = n;
      rp.Z = Z_avx.data();
      rp.ldz = ldz;
      rp.C = C.data();
      rp.ldc = n;
      rp.A = A.data();
      rp.lda = n;
      rp.B = B.data();
      rp.ldb = ldb;
      rp.D = D.data();
      rp.ldd = ldd;
      wavernn::fine_sigmoid(rp);

      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          float expected = Z_ref[i * ldz + j];
          float actual = Z_avx[i * ldz + j];
          EXPECT_NEAR(expected, actual, epsilon)
              << "FineSigmoid results differ at (" << i << ", " << j
              << "). ref " << expected << " avx " << actual;
        }
      }
    }

    // fine tanh
    {
      wavernn::fine_tanh(m, n, Z_ref.data(), ldz, C.data(), n, A.data(), n,
                         B.data(), ldb, D.data(), ldd, T.data(), ldt);

      wavernn::FineTanhParams rp;
      rp.m = m;
      rp.b_block_cols = n;
      rp.Z = Z_avx.data();
      rp.ldz = ldz;
      rp.C = C.data();
      rp.ldc = n;
      rp.A = A.data();
      rp.lda = n;
      rp.B = B.data();
      rp.ldb = ldb;
      rp.D = D.data();
      rp.ldd = ldd;
      rp.T = T.data();
      rp.ldt = ldt;
      wavernn::fine_tanh(rp);

      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          float expected = Z_ref[i * ldz + j];
          float actual = Z_avx[i * ldz + j];
          EXPECT_NEAR(expected, actual, 1e-4)
              << "FineTanh results differ at (" << i << ", " << j << "). ref "
              << expected << " avx " << actual;
        }
      }
    }

    // hidden
    {
      wavernn::hidden(m, n, Z_ref.data(), ldz, A.data(), n, B.data(), ldb,
                      C.data(), n);

      wavernn::HiddenParams rp;
      rp.m = m;
      rp.b_block_cols = n;
      rp.Z = Z_avx.data();
      rp.ldz = ldz;
      rp.C = C.data();
      rp.ldc = n;
      rp.A = A.data();
      rp.lda = n;
      rp.B = B.data();
      rp.ldb = ldb;
      wavernn::hidden(rp);

      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          float expected = Z_ref[i * ldz + j];
          float actual = Z_avx[i * ldz + j];
          EXPECT_NEAR(expected, actual, epsilon)
              << "Hidden results differ at (" << i << ", " << j << "). ref "
              << expected << " avx " << actual;
        }
      }
    }
  }
}
