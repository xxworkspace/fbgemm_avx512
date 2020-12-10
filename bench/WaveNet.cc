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

void performance_test(int batch_size = 4, int num_flows = 4,
                      int flow_layers = 6) {
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

  int channels = 256;
  // WaveNet - 256
  for (auto m = 1; m <= batch_size; m++) {
    shapes.clear();
    shapes.push_back({m, channels, channels});
    shapes.push_back({m, 256, channels});  // skip -> hidden
    shapes.push_back({m, 256, 3 * 10});    // hidden -> softmax
    all_shapes.push_back(shapes);
  }

  // WaveNet - 128
  channels = 128;
  for (auto m = 1; m <= batch_size; m++) {
    shapes.clear();
    shapes.push_back({m, channels, channels});
    shapes.push_back({m, 256, channels});
    shapes.push_back({m, 256, 3 * 10});
    all_shapes.push_back(shapes);
  }

  // WaveNet - 64
  channels = 64;
  for (auto m = 1; m <= batch_size; m++) {
    shapes.clear();
    shapes.push_back({m, channels, channels});
    shapes.push_back({m, 256, channels});
    shapes.push_back({m, 256, 3 * 10});
    all_shapes.push_back(shapes);
  }

  std::string type;

  for (auto s : all_shapes) {
    double ttot = 0, ttot_flow_gemm = 0, ttot_bott_gemm = 0, ttot_relu = 0,
           ttot_sigmoid = 0, ttot_tanh = 0;

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
        if (it >= 0 && i == 0) {
          t_begin = chrono::system_clock::now();
          // GEMM
          cblas_gemm_compute(matrix_op_t::NoTranspose, m, A_gemm.data(), Bp,
                             beta, Cp.data(), 0, scratchpad->data());
          t_end = chrono::system_clock::now();

          double dt = chrono::duration<double>(t_end - t_begin).count();
          ttot_flow_gemm += dt;

          SigmoidParams rp;
          rp.m = m;
          rp.b_block_cols = n / 8;
          rp.Z = Z_avx.data();
          rp.ldz = n;
          rp.A = Z_ref.data();
          rp.lda = n;

          t_begin = chrono::system_clock::now();
          sigmoid(rp);
          t_end = chrono::system_clock::now();
          ttot_sigmoid += chrono::duration<double>(t_end - t_begin).count();

          TanhParams tp;
          tp.m = m;
          tp.b_block_cols = n / 8;
          tp.Z = Z_avx.data();
          tp.ldz = n;
          tp.A = Z_ref.data();
          tp.lda = n;

          t_begin = chrono::system_clock::now();
          tanh(tp);
          t_end = chrono::system_clock::now();
          ttot_tanh += chrono::duration<double>(t_end - t_begin).count();
        }

        if (it >= 0 && i >= 1) {
          t_begin = chrono::system_clock::now();
          // GEMM
          cblas_gemm_compute(matrix_op_t::NoTranspose, m, A_gemm.data(), Bp,
                             beta, Cp.data(), 0, scratchpad->data());
          t_end = chrono::system_clock::now();
          ttot_bott_gemm += chrono::duration<double>(t_end - t_begin).count();

          // ReLU
          ReLUParams rp;
          rp.m = m;
          rp.b_block_cols = n / 8;
          rp.Z = relu_A.data();
          rp.ldz = n;
          rp.A = relu_A.data();
          rp.lda = n;

          t_begin = chrono::system_clock::now();
          relu(rp);
          t_end = chrono::system_clock::now();
          ttot_relu += chrono::duration<double>(t_end - t_begin).count();
        }
      }
    }

    int num_layers = num_flows * flow_layers;
    double max_activ = std::max(ttot_sigmoid, ttot_tanh) * num_layers;
    ttot = ttot_flow_gemm * num_layers + ttot_bott_gemm + ttot_relu + max_activ;

    printf(
        "%12s batch %2d Flows %d Layers %d channels %4d "
        "RTF = %2.4lf FlowGemm(x%d) = %2.4lf SigmodTanh(x%d) = %2.4lf "
        "BottGemm = %2.4lf ReLU = %2.4lf\n",
        type.c_str(), s[0][0], num_flows, flow_layers, s[0][1],
        ttot * 24000.0 / NITER, num_layers,
        ttot_flow_gemm * 24000.0 / NITER * num_layers, num_layers,
        max_activ * 24000.0 / NITER, ttot_bott_gemm * 24000.0 / NITER,
        ttot_relu * 24000.0 / NITER);
    ((volatile char*)(llc.data()));
  }
}

int main(int argc, char** argv) {
  int batch_size = 1;
  int num_flows = 4;
  int flow_layers = 6;
  if (argc == 2) {
    batch_size = std::stoi(argv[1]);
  }
  if (argc == 4) {
    batch_size = std::stoi(argv[1]);
    num_flows = std::stoi(argv[2]);
    flow_layers = std::stoi(argv[3]);
  }
  performance_test(batch_size, num_flows, flow_layers);
}
