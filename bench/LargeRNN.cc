/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <immintrin.h>
#include <stdlib.h> /* atoi */
#include <array>
#include <chrono>
#include <cmath>
#include <random>

#include "skylark/inference/blas/FbgemmFP16.h"
#include "skylark/inference/blas/FbgemmFP16Sparse.h"
#include "skylark/inference/blas/bench/AlignedVec.h"
#include "skylark/inference/blas/bench/BenchUtils.h"
#include "tbb/flow_graph.h"
#include "tbb/tbb.h"

using namespace std;
using namespace fbgemm;

void performance_test(size_t num_threads = 2) {
  num_threads = num_threads > 16 ? 16 : num_threads;

  // cache flush
  bool flush = true;
  std::vector<char> llc;
  if (flush) {
    llc.resize(64L * 1024L * 1024L, 1.0);
  }

  float alpha = 1.f, beta = 0.f;
  matrix_op_t btran = matrix_op_t::NoTranspose;

  using btype = float16;

  std::random_device r;
  std::default_random_engine generator(r());

  const int NITER = (flush) ? 20 : 100;
  std::vector<std::vector<int>> shapes;

  int batch_size = 1;
  // Decoder RNN - 1024
  for (auto m = 1; m <= batch_size; m++) {
    shapes.push_back({m, 1024 * 4, 1024 + 256 + 512});
  }

  // Decoder RNN - 896
  for (auto m = 1; m <= batch_size; m++) {
    shapes.push_back({m, 896 * 4, 896 + 256 + 512});
  }

  // Decoder RNN - 512
  for (auto m = 1; m <= batch_size; m++) {
    shapes.push_back({m, 512 * 4, 512 + 256 + 512});
  }

  std::string type;
  double gflops, gbs, ttot;

  auto m_pInit = std::unique_ptr<tbb::task_scheduler_init>(
      new tbb::task_scheduler_init(num_threads));
  const int num_steps = 80 * 4;

  for (auto s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    aligned_vector<float> A(m * k, 0.f);
    aligned_vector<float> B(k * n, 0.f);
    aligned_vector<float> Cp(m * n, NAN);
    aligned_vector<float> Cg(m * n, NAN);

    // initialize with small numbers
    randFill(A, 0.f, 4.f);
    randFill(B, 0.f, 4.f);

    // fbgemm fp16
    PackedGemmMatrixFP16 Bp(btran, k, n, alpha, B.data());
    // mutlti-thread
    PackedGemmMatrixFP16 Bg(btran, k, n, alpha, B.data(), 512,
                            std::vector<int>(num_threads, n / num_threads));

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
    type = "FBP_" + std::string(typeid(btype).name());

    ttot = 0;
    for (auto it = -3; it < NITER; it++) {
      if (flush) {
        for (auto i = 0; i < llc.size(); i++) {
          llc[i]++;
        }
      }

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
        "\n%36s m = %5d n = %5d k = %5d "
        "Gflops = %8.4lf GBytes = %8.4lf "
        "RNN-Rxh(RTF) = %8.4lf\n",
        type.c_str(), m, n, k, gflops, gbs, ttot * 80.0 / NITER);

    // multi-thread GEMM
    type = "Threads" + std::to_string(num_threads) + "FBP_" +
           std::string(typeid(btype).name());
    {
      ttot = 0;
      t_begin = chrono::system_clock::now();

      auto buffer = multiscratchpad->data();
      PackAMatrix packedA(matrix_op_t::NoTranspose, m, k, A.data(), k, buffer,
                          Bp.blockRowSize());
      for (int i = 0; i < num_steps; ++i) {
        tbb::parallel_for(size_t(0), size_t(num_threads),
                          [&packedA, &Bg, &beta, &Cg, &buffer](size_t t) {
                            auto cdata = Cg.data() + Bg.groupBeginCol(t);
                            PackedGemmMatrixFP16 Bgg(Bg, t);
                            cblas_gemm_compute(packedA, Bgg, beta, cdata,
                                               Bg.groupShift(t));
                          });
      }

      t_end = chrono::system_clock::now();
      double dt = chrono::duration<double>(t_end - t_begin).count();
      ttot += dt;

      gflops = (nflops / NITER) * num_steps / ttot / 1e9;
      gbs = (nbytes / NITER) * num_steps / ttot / 1e9;
      printf(
          "%36s m = %5d n = %5d k = %5d "
          "Gflops = %8.4lf GBytes = %8.4lf "
          "RNN-Rxh(RTF) = %8.4lf\n",
          type.c_str(), m, n, k, gflops, gbs, ttot * 80.0 / num_steps);
    }
    ((volatile char*)(llc.data()));
  }
}

int main(int argc, char** argv) {
  int num_threads = 4;
  if (argc == 2) num_threads = std::stoi(argv[1]);

  performance_test(num_threads);
}
