/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <array>
#include <chrono>
#include <cmath>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <immintrin.h>

#include "skylark/inference/blas/FbgemmFP16.h"
#include "skylark/inference/blas/FbgemmFP16Sparse.h"
#include "skylark/inference/blas/bench/AlignedVec.h"
#include "skylark/inference/blas/bench/BenchUtils.h"
#include "tbb/flow_graph.h"
#include "tbb/tbb.h"

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

  using btype = float16;

  std::random_device r;
  std::default_random_engine generator(r());

#define dataset 1

#if dataset == 1
  const int NITER = (flush) ? 10 : 100;
  std::vector<std::vector<int>> shapes;

  int batch_size = 4;
  // WaveRNN - 1024
  for (auto m = 1; m <= batch_size; m++) {
    shapes.push_back({m, 1024 / 2, 1024});
  }

  for (auto m = 1; m <= batch_size; m++) {
    shapes.push_back({m, 512, 512});
  }

  for (auto m = 1; m <= batch_size; m++) {
    shapes.push_back({m, 256, 512});
  }

  // WaveRNN - 2048
  for (auto m = 1; m <= batch_size; m++) {
    shapes.push_back({m, 2048 / 2, 2048});
  }

  for (auto m = 1; m <= batch_size; m++) {
    shapes.push_back({m, 512, 1024});
  }

  for (auto m = 1; m <= batch_size; m++) {
    shapes.push_back({m, 256, 512});
  }

#elif dataset == 2
  const int NITER = (flush) ? 10 : 100;
#include "shapes_dataset.h"

#else
  flush = false;
  constexpr int NITER = 1;
  std::vector<std::vector<int>> shapes;
  std::uniform_int_distribution<int> dm(1, 100);
  std::uniform_int_distribution<int> dnk(1, 1024);
  for (int i = 0; i < 1000; i++) {
    int m = dm(generator);
    int n = dnk(generator);
    int k = dnk(generator);
    shapes.push_back({m, n, k});
  }
#endif

  std::string type;
  double gflops, gbs, ttot;
  vector<vector<int>> blocks = {{16, 1}};

  const size_t num_threads = 4;  // num_threads
  auto m_pInit = std::unique_ptr<tbb::task_scheduler_init>(
      new tbb::task_scheduler_init(num_threads));
  const int num_steps = 2400;
  std::vector<int> range;
  for (int i = 0; i < num_steps; ++i) {
    for (int t = 0; t < num_threads; ++t) {
      range.push_back(t);
    }
  }

  for (auto s : shapes) {
    for (auto b : blocks) {
      for (float density : {0.10f}) {
        int block_height = b[0];
        int block_width = b[1];
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
        // generate mask and update B
        aligned_vector<int> mask(s[2] * s[1] / block_height / block_width);
        {
          uniform_int_distribution<int> r(1, 100);
          for (auto i = 0; i < k / block_height; i++) {
            for (auto j = 0; j < n / block_width; j++) {
              int dense = r(generator) / 100.0 < density;
              mask[i * n / block_width + j] = dense;
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

        // fbgemm fp16
        PackedGemmMatrixFP16 Bp(btran, k, n, alpha, B.data());
        // mutlti-thread
        PackedGemmMatrixFP16 Bg(btran, k, n, alpha, B.data(), 512, std::vector<int>(num_threads, n / num_threads));
        // sparse
        PackedGemmBlockSparseMatrixFP16 sparseBp(
            matrix_op_t::NoTranspose, alpha, B.data(), mask.data(),
            k / block_height, n / block_width, block_height, block_width);
        PackedGemmBlockSparseMatrixFP16 sparseBg(
            matrix_op_t::NoTranspose, alpha, B.data(), mask.data(),
            k / block_height, n / block_width, block_height, block_width,
            num_threads);

        if (beta != 0.0f) {
          randFill(Cs, 0.f, 4.f);
          Cp = Cs;
        }

        double nflops = 2.0 * (double)m * (double)n * (double)k * (double)NITER;
        double nbytes =
            (4.0 * (double)m * (double)k + 2.0 * (double)k * (double)n +
             4.0 * (double)m * (double)n) *
            NITER;

        // check correctness at the same time
        for (auto w = 0; w < 3; w++) {
          cblas_gemm_compute(matrix_op_t::NoTranspose, m, A.data(), sparseBp,
                             beta, Cs.data());
          cblas_gemm_compute(matrix_op_t::NoTranspose, m, A.data(), Bp, beta,
                             Cp.data());

          // Compare results
          for (auto i = 0; i < Cs.size(); i++) {
            // printf("%f %f\n", Cs[i], Cp[i]);
            assert(std::abs(Cs[i] - Cp[i]) < 1e-3);
          }
        }

        chrono::time_point<chrono::system_clock> t_begin, t_end;
        static thread_local unique_ptr<std::array<float, 256 * 1024>>
        scratchpad(new std::array<float, 256 * 1024>());

        static thread_local unique_ptr<
            std::array<float, 256 * 1024 * num_threads>>
        multiscratchpad(new std::array<float, 256 * 1024 * num_threads>());

        {  // anther vesion
          vector<vector<int>> offsets;
          offsets.resize(n / block_width);
          aligned_vector<float> sparseBpV2(k * n, NAN);
          aligned_vector<float> Cq(m * n, 0.f);

          int idx = 0;
          for (auto j = 0; j < n / block_width; j++) {
            for (auto i = 0; i < k / block_height; i++) {
              if (mask[i * n / block_width + j] != 0) {
                offsets[j].push_back(i * block_height);
                int r = i * block_height;
                int c = j * block_width;
                for (auto rx = r; rx < r + block_height; rx++) {
                  for (auto cx = c; cx < c + block_width; cx++) {
                    sparseBpV2[idx++] = B[rx * n + cx];
                  }
                }
              }
            }
          }

          type = "V2SparseFBP_" + std::string(typeid(btype).name());
          ttot = 0;
          for (auto it = -3; it < NITER; it++) {
            if (flush) {
              for (auto i = 0; i < llc.size(); i++) {
                llc[i]++;
              }
            }
            assert(block_width == 1);
            t_begin = chrono::system_clock::now();
            {
              int nb = block_height / 8;
              float sum[8] __attribute__((aligned(64)));
              for (auto mi = 0; mi < m; mi++) {
                auto aptr = A.data() + mi * k;
                auto bptr = sparseBpV2.data();
                auto cptr = Cq.data() + mi * n;
                for (auto j = 0; j < n; j++) {
                  __m256 sumV = _mm256_setzero_ps();
                  for (auto f : offsets[j]) {
                    for (auto bi = 0; bi < nb; bi++) {
                      __m256 ra = _mm256_loadu_ps(aptr + f + bi * 8);
                      __m256 rb = _mm256_loadu_ps(bptr);
                      sumV = _mm256_fmadd_ps(ra, rb, sumV);

                      bptr += 8;
                    }
                  }
                  _mm256_store_ps(sum, sumV);
                  cptr[j] = (sum[0] + sum[1]) + (sum[2] + sum[3]) +
                            (sum[4] + sum[5]) + (sum[6] + sum[7]);
                }
              }
            }

            t_end = chrono::system_clock::now();
            if (it >= 0) {
              double dt = chrono::duration<double>(t_end - t_begin).count();
              ttot += dt;
            }
          }
          gflops = nflops / ttot / 1e9;
          gbs = nbytes / ttot / 1e9;
          printf(
              "\n%36s m = %5d n = %5d k = %5d block [%2d, %d] density %1.4f "
              "Gflops = %8.4lf GBytes = %8.4lf "
              "WaveRNN-Rxh(RTF) = %8.4lf\n",
              type.c_str(), m, n, k, block_height, block_width, density, gflops,
              gbs, ttot * 24000.0 / NITER);
        }

        type = "SparseFBP_" + std::string(typeid(btype).name());
        ttot = 0;

        for (auto it = -3; it < NITER; it++) {
          if (flush) {
            for (auto i = 0; i < llc.size(); i++) {
              llc[i]++;
            }
          }
          t_begin = chrono::system_clock::now();
          cblas_gemm_compute(matrix_op_t::NoTranspose, m, A.data(), sparseBp,
                             beta, Cs.data());
          t_end = chrono::system_clock::now();
          if (it >= 0) {
            double dt = chrono::duration<double>(t_end - t_begin).count();
            ttot += dt;
          }
        }
        gflops = nflops / ttot / 1e9;
        gbs = nbytes / ttot / 1e9;
        printf(
            "%36s m = %5d n = %5d k = %5d block [%2d, %d] density %1.4f "
            "Gflops = %8.4lf GBytes = %8.4lf "
            "WaveRNN-Rxh(RTF) = %8.4lf\n",
            type.c_str(), m, n, k, block_height, block_width, density, gflops,
            gbs, ttot * 24000.0 / NITER);

        // -------------------- multi-thread Sparse GEMM --------------------
        type = "Threads" + std::to_string(num_threads) + "SparseFBP_" +
               std::string(typeid(btype).name());
        {
          ttot = 0;
          t_begin = chrono::system_clock::now();
          const int shift = (num_threads - 1) * sparseBg.groupNumCols();
          for (int i = 0; i < num_steps; ++i) {
            tbb::parallel_for(
                size_t(0), size_t(num_threads),
                [m, &A, &sparseBg, &beta, &Cg, &shift](size_t t) {
                  auto cdata = Cg.data() + t * sparseBg.groupNumCols();
                  PackedGemmBlockSparseMatrixFP16 Bgg(sparseBg, t);
                  cblas_gemm_compute(matrix_op_t::NoTranspose, m, A.data(), Bgg,
                                     beta, cdata, shift);
                });
          }

          t_end = chrono::system_clock::now();
          double dt = chrono::duration<double>(t_end - t_begin).count();
          ttot += dt;
        }
        gflops = (nflops / NITER) * num_steps / ttot / 1e9;
        gbs = (nbytes / NITER) * num_steps / ttot / 1e9;
        printf(
            "%36s m = %5d n = %5d k = %5d block [%2d, %d] density %1.4f "
            "Gflops = %8.4lf GBytes = %8.4lf "
            "WaveRNN-Rxh(RTF) = %8.4lf\n",
            type.c_str(), m, n, k, block_height, block_width, density, gflops,
            gbs, ttot * 24000.0 / num_steps);

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
            "%36s m = %5d n = %5d k = %5d block [%2d, %d] density %1.4f "
            "Gflops = %8.4lf GBytes = %8.4lf "
            "WaveRNN-Rxh(RTF) = %8.4lf\n",
            type.c_str(), m, n, k, block_height, block_width, density, gflops,
            gbs, ttot * 24000.0 / NITER);

        // multi-thread GEMM
        type = "Threads" + std::to_string(num_threads) + "FBP_" +
               std::string(typeid(btype).name());
        {
          ttot = 0;
          t_begin = chrono::system_clock::now();

          for (int i = 0; i < num_steps; ++i) {
            auto buffer = multiscratchpad->data();
            tbb::parallel_for(
                size_t(0), size_t(num_threads),
                [m, &A, &Bg, &beta, &Cg, &buffer](size_t t) {
                  auto cdata = Cg.data() + Bg.groupBeginCol(t);
                  PackedGemmMatrixFP16 Bgg(Bg, t);
                  auto spbuffer = buffer + t * 256 * 1024;
                  cblas_gemm_compute(matrix_op_t::NoTranspose, m, A.data(), Bgg,
                                     beta, cdata, Bg.groupShift(t), spbuffer);
                });
          }

          t_end = chrono::system_clock::now();
          double dt = chrono::duration<double>(t_end - t_begin).count();
          ttot += dt;

          gflops = (nflops / NITER) * num_steps / ttot / 1e9;
          gbs = (nbytes / NITER) * num_steps / ttot / 1e9;
          printf(
              "%36s m = %5d n = %5d k = %5d block [%2d, %d] density %1.4f "
              "Gflops = %8.4lf GBytes = %8.4lf "
              "WaveRNN-Rxh(RTF) = %8.4lf\n",
              type.c_str(), m, n, k, block_height, block_width, density, gflops,
              gbs, ttot * 24000.0 / num_steps);
        }

        // multi-thread GEMM
        type = "FlowGraphThreads" + std::to_string(num_threads) + "FBP_" +
               std::string(typeid(btype).name());
        {
          ttot = 0;
          t_begin = chrono::system_clock::now();
          typedef tbb::flow::continue_node<tbb::flow::continue_msg> node_t;
          typedef const tbb::flow::continue_msg& msg_t;
          std::vector<std::unique_ptr<node_t>> task_nodes(num_steps);
          tbb::flow::graph g;
          node_t src(g, [num_steps](msg_t) {});

          auto buffer = multiscratchpad->data();

          for (int i = 0; i < num_steps; ++i) {
            task_nodes[i].reset(new node_t(
                g, [num_threads, m, &A, &Bg, &beta, &Cg, &buffer](msg_t) {
                  cblas_gemm_compute(matrix_op_t::NoTranspose, m, A.data(), Bg,
                                     beta, Cg.data(), 0, buffer);
                }));

            if ((i > 0) && (i % num_threads == 0)) {
              for (int t = 0; t < num_threads; ++t) {
                tbb::flow::make_edge(*task_nodes[i - t - 1].get(),
                                     *task_nodes[i].get());
              }
            } else {
              tbb::flow::make_edge(src, *task_nodes[i].get());
            }
          }
          src.try_put(tbb::flow::continue_msg());
          g.wait_for_all();

          t_end = chrono::system_clock::now();
          double dt = chrono::duration<double>(t_end - t_begin).count();
          ttot += dt;
          task_nodes.clear();

          gflops = (nflops / NITER) * num_steps / ttot / 1e9;
          gbs = (nbytes / NITER) * num_steps / ttot / 1e9;
          printf(
              "%36s m = %5d n = %5d k = %5d block [%2d, %d] density %1.4f "
              "Gflops = %8.4lf GBytes = %8.4lf "
              "WaveRNN-Rxh(RTF) = %8.4lf\n",
              type.c_str(), m, n, k, block_height, block_width, density, gflops,
              gbs, ttot * 24000.0 / num_steps);
        }
        ((volatile char*)(llc.data()));
      }
    }
  }
}

int main(int /*argc*/, char** /*argv*/) {
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif

  performance_test();
}
