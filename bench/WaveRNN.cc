/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <stdlib.h> /* atoi */
#include <array>
#include <chrono>
#include <cmath>
#include <random>

#include "skylark/inference/blas/ActivationsFP32.h"
#include "skylark/inference/blas/FbgemmFP16.h"
#include "skylark/inference/blas/bench/AlignedVec.h"
#include "skylark/inference/blas/bench/BenchUtils.h"

using namespace std;
using namespace fbgemm;

void performance_test(int batch_size = 4) {
  // cache flush
  bool flush = true;
  std::vector<char> llc;
  if (flush) {
    llc.resize(64L * 1024L * 1024L, 1.0);
  }

  float alpha = 1.f, beta = 1.f;
  matrix_op_t btran = matrix_op_t::NoTranspose;

  std::random_device r;
  std::default_random_engine generator(r());

  const int NITER = (flush) ? 40 : 100;
  std::vector<std::vector<std::vector<int>>> all_shapes;
  std::vector<std::vector<int>> shapes;

  // WaveRNN - 1024
  for (auto m = 1; m <= batch_size; m++) {
    shapes.clear();
    shapes.push_back({m, 1024 / 2, 1024});
    shapes.push_back({m, 512 / 4, 1024 / 2});
    shapes.push_back({m, 256 / 2, 512});
    all_shapes.push_back(shapes);
  }

  // WaveRNN - 896
  for (auto m = 1; m <= batch_size; m++) {
    shapes.clear();
    shapes.push_back({m, 896 / 2, 896});
    shapes.push_back({m, 512 / 4, 896 / 2});
    shapes.push_back({m, 256 / 2, 512});
    all_shapes.push_back(shapes);
  }

  std::string type;

  for (auto s : all_shapes) {
    double ttot = 0, ttot_gemm = 0, ttot_relu = 0, ttot_coarse_sigmoid = 0,
           ttot_coarse_tanh = 0, ttot_fine_sigmoid = 0, ttot_fine_tanh = 0,
           ttot_hidden = 0;

    for (auto i : {0, 1, 2}) {
      int m = s[i][0];
      int n = s[i][1];
      int k = s[i][2];

      aligned_vector<float> A_gemm(m * k, 0.f);
      aligned_vector<float> B_gemm(k * n, 0.f);
      aligned_vector<float> Cp(m * n, NAN);

      // initialize with small numbers
      randFill(A_gemm, 0.f, 4.f);
      randFill(B_gemm, 0.f, 4.f);

      // relu
      aligned_vector<float> relu_A(m * n * 4, NAN);
      aligned_vector<float> relu_D(m * n * 4, NAN);
      randFill(relu_A, -4.f, 4.f);

      aligned_vector<float> A(m * n);
      randFill(A, -8.f, 8.f);
      aligned_vector<float> B(m * n);
      randFill(B, -8.f, 8.f);
      aligned_vector<float> D(m * n);
      randFill(D, -8.f, 8.f);
      aligned_vector<float> C(m * n);
      randFill(C, -8.f, 8.f);
      aligned_vector<float> T(m * n);
      randFill(T, -8.f, 8.f);

      aligned_vector<float> Z_avx(m * n);
      randFill(Z_avx, -8.f, 8.f);

      aligned_vector<float> Z_ref(m * n);

      // fbgemm fp16
      PackedGemmMatrixFP16 Bp(btran, k, n, alpha, B_gemm.data());

      if (beta != 0.0f) {
        randFill(Cp, 0.f, 4.f);
      }

      chrono::time_point<chrono::system_clock> t_begin, t_end;
      static thread_local unique_ptr<std::array<float, 256 * 1024>> scratchpad(
          new std::array<float, 256 * 1024>());

      // -------------------- Dense GEMM --------------------
      type = "UNIVERSAL";

      // benchmark
      for (auto it = -3; it < NITER; it++) {
        t_begin = chrono::system_clock::now();
        // GEMM
        cblas_gemm_compute(matrix_op_t::NoTranspose, m, A_gemm.data(), Bp, beta,
                           Cp.data(), 0, scratchpad->data());
        t_end = chrono::system_clock::now();
        if (it >= 0) {
          double dt = chrono::duration<double>(t_end - t_begin).count();
          ttot_gemm += dt;
          if (i > 0) {
            ttot_gemm += dt;
          }
        }

        if (it >= 0 && i == 1) {
          // ReLU
          ReLUParams rp;
          rp.m = m;
          rp.b_block_cols = n * 4 / 8;
          rp.Z = relu_A.data();
          rp.ldz = n * 4;
          rp.A = relu_A.data();
          rp.lda = n * 4;

          t_begin = chrono::system_clock::now();
          relu(rp);
          t_end = chrono::system_clock::now();

          double dt = chrono::duration<double>(t_end - t_begin).count();
          ttot_relu += dt;
        }

        if (it >= 0 && i == 0) {
          wavernn::CoarseSigmoidParams rp;
          rp.m = m;
          rp.b_block_cols = n / 8;
          rp.Z = Z_avx.data();
          rp.ldz = n;
          rp.C = C.data();
          rp.ldc = n;
          rp.A = A.data();
          rp.lda = n;
          rp.B = B.data();
          rp.ldb = n;

          t_begin = chrono::system_clock::now();
          wavernn::coarse_sigmoid(rp);
          t_end = chrono::system_clock::now();

          double dt = chrono::duration<double>(t_end - t_begin).count();
          ttot_coarse_sigmoid += dt;

          // hidden
          wavernn::HiddenParams hp;
          hp.m = m;
          hp.b_block_cols = n / 8;
          hp.Z = Z_avx.data();
          hp.ldz = n;
          hp.C = C.data();
          hp.ldc = n;
          hp.A = A.data();
          hp.lda = n;
          hp.B = B.data();
          hp.ldb = n;
          t_begin = chrono::system_clock::now();
          wavernn::hidden(hp);
          t_end = chrono::system_clock::now();
          dt = chrono::duration<double>(t_end - t_begin).count();
          ttot_hidden += dt;
        }

        if (it >= 0 && i == 0) {
          wavernn::CoarseTanhParams rp;
          rp.m = m;
          rp.b_block_cols = n / 8;
          rp.Z = Z_avx.data();
          rp.ldz = n;
          rp.C = C.data();
          rp.ldc = n;
          rp.A = A.data();
          rp.lda = n;
          rp.B = B.data();
          rp.ldb = n;
          rp.D = D.data();
          rp.ldd = n;
          t_begin = chrono::system_clock::now();
          wavernn::coarse_tanh(rp);
          t_end = chrono::system_clock::now();

          double dt = chrono::duration<double>(t_end - t_begin).count();
          ttot_coarse_tanh += dt;
        }

        if (it >= 0 && i == 0) {
          wavernn::FineSigmoidParams rp;
          rp.m = m;
          rp.b_block_cols = n / 8;
          rp.Z = Z_avx.data();
          rp.ldz = n;
          rp.C = C.data();
          rp.ldc = n;
          rp.A = A.data();
          rp.lda = n;
          rp.B = B.data();
          rp.ldb = n;
          rp.D = D.data();
          rp.ldd = n;

          t_begin = chrono::system_clock::now();
          wavernn::fine_sigmoid(rp);
          t_end = chrono::system_clock::now();
          double dt = chrono::duration<double>(t_end - t_begin).count();
          ttot_fine_sigmoid += dt;
        }

        if (it >= 0 && i == 0) {
          wavernn::FineTanhParams rp;
          rp.m = m;
          rp.b_block_cols = n / 8;
          rp.Z = Z_avx.data();
          rp.ldz = n;
          rp.C = C.data();
          rp.ldc = n;
          rp.A = A.data();
          rp.lda = n;
          rp.B = B.data();
          rp.ldb = n;
          rp.D = D.data();
          rp.ldd = n;
          rp.T = T.data();
          rp.ldt = n;

          t_begin = chrono::system_clock::now();
          wavernn::fine_tanh(rp);
          t_end = chrono::system_clock::now();
          double dt = chrono::duration<double>(t_end - t_begin).count();
          ttot_fine_tanh += dt;
        }
      }
    }

    double max_coarse_activ = std::max(ttot_coarse_sigmoid, ttot_coarse_tanh);
    double max_fine_activ = std::max(ttot_fine_sigmoid, ttot_fine_tanh);

    ttot = ttot_gemm + 2 * ttot_relu + max_coarse_activ + max_fine_activ +
           2 * ttot_hidden;

    printf(
        "%12s units %4d batch %2d "
        "ParallelRTF = %2.4lf "
        "Gemm(Rh/2+O1+O2+O3+O4) = %2.4lf ReLU(x2) = %2.4lf CoarseActi = %2.4lf "
        "FineActi = %2.4lf Hidden(x2) = %2.4lf\n",
        type.c_str(), s[0][2], s[0][0], ttot * 24000.0 / NITER,
        ttot_gemm * 24000.0 / NITER, 2 * ttot_relu * 24000.0 / NITER,
        max_coarse_activ * 24000.0 / NITER, max_fine_activ * 24000.0 / NITER,
        ttot_hidden);
    ((volatile char*)(llc.data()));
  }
}

int main(int argc, char** argv) {
  int batch_size = 1;
  if (argc == 2) batch_size = std::stoi(argv[1]);
  performance_test(batch_size);
}
