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

#include "skylark/inference/blas/FbgemmFP16.h"
#include "skylark/inference/blas/bench/AlignedVec.h"
#include "skylark/inference/blas/bench/BenchUtils.h"

using namespace std;
using namespace fbgemm;

void performance_test() {
  // cache flush
  bool flush = true;
  std::vector<char> llc;
  if (flush) {
    llc.resize(64L * 1024L * 1024L, 1.0);
  }

  float alpha = 1.f, beta = 0.f;
  matrix_op_t btran = matrix_op_t::NoTranspose;

  std::random_device r;
  std::default_random_engine generator(r());

  const int NITER = (flush) ? 20 : 100;
  std::vector<std::vector<int>> shapes;

  int batch_size = 4;
  // WaveRNN - 1024
  for (auto m = 1; m <= batch_size; m++) {
    shapes.push_back({m, 1024 / 2, 1024});
    shapes.push_back({m, 512, 1024 / 2});
    shapes.push_back({m, 256, 512});
  }

  // WaveRNN - 896
  for (auto m = 1; m <= batch_size; m++) {
    shapes.push_back({m, 896 / 2, 896});
    shapes.push_back({m, 512, 896 / 2});
    shapes.push_back({m, 256, 512});
  }

  std::string type;
  double gflops, gbs, ttot;

  for (auto s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    aligned_vector<float> A(m * k, 0.f);
    aligned_vector<float> B(k * n, 0.f);
    aligned_vector<float> Cs(m * n, 0.f);
    aligned_vector<float> Cp(m * n, NAN);
    aligned_vector<float> Cg(m * n, NAN);

    // initialize with small numbers
    randFill(A, 0.f, 4.f);
    randFill(B, 0.f, 4.f);

    // fbgemm fp16
    PackedGemmMatrixFP16 Bp(btran, k, n, alpha, B.data());

    if (beta != 0.0f) {
      randFill(Cs, 0.f, 4.f);
      Cp = Cs;
    }

    double nflops = 2.0 * (double)m * (double)n * (double)k * (double)NITER;
    double nbytes = (4.0 * (double)m * (double)k + 2.0 * (double)k * (double)n +
                     4.0 * (double)m * (double)n) *
                    NITER;

    chrono::time_point<chrono::system_clock> t_begin, t_end;
    static thread_local unique_ptr<std::array<float, 256 * 1024>> scratchpad(
        new std::array<float, 256 * 1024>());

    static thread_local unique_ptr<std::array<float, 256 * 1024 * 16>>
    multiscratchpad(new std::array<float, 256 * 1024 * 16>());

    // -------------------- Dense GEMM --------------------
    type = "UNIVERSAL";

    ttot = 0;
    for (auto it = -3; it < NITER; it++) {
      // if (flush) {
      //   for (auto i = 0; i < llc.size(); i++) {
      //     llc[i]++;
      //   }
      // }

      t_begin = chrono::system_clock::now();
      cblas_gemm_compute(matrix_op_t::NoTranspose, m, A.data(), Bp, beta,
                         Cp.data(), 0, scratchpad->data());

      t_end = chrono::system_clock::now();

      if (it >= 0) {
        double dt = chrono::duration<double>(t_end - t_begin).count();
        ttot += dt;
      }
    }
    gflops = nflops / ttot / 1e9;
    gbs = nbytes / ttot / 1e9;
    printf(
        "%16s m = %5d n = %5d k = %5d "
        "GFlops = %8.4lf GBytes = %8.4lf "
        "RTF = %8.4lf\n",
        type.c_str(), m, n, k, gflops, gbs, ttot * 24000.0 / NITER);
    ((volatile char*)(llc.data()));
  }
}

int main(int argc, char** argv) {
  performance_test();
}
