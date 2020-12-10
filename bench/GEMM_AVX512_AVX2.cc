/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdlib.h> /* atoi */
#include <thread>
#include <vector>

#include "skylark/inference/blas/FbgemmFP16.h"
#include "skylark/inference/blas/bench/AlignedVec.h"
#include "skylark/inference/blas/bench/BenchUtils.h"
#include "skylark/inference/blas/cpuinfo.h"
#include "tbb/tbb.h"

using namespace std;
using namespace fbgemm;

// grpah wavernn base on blockwavernn
inline void FUNC_GEMM(const fbgemm::PackAMatrix &A,
                      const fbgemm::PackedGemmMatrixFP16 &Bp, float belta,
                      float *C, const int g) {
  auto rdata = C + Bp.groupBeginCol(g);
  fbgemm::PackedGemmMatrixFP16 Bpg(Bp, g);
  fbgemm::cblas_gemm_compute(A, Bpg, belta, rdata, Bp.groupShift(g));
}

#define barrier() __asm__ __volatile__("" ::: "memory")

struct shape_s {
  int m;
  int k;
  int n;
  int g;
  shape_s(int _m, int _k, int _n, int _g) : m(_m), k(_k), n(_n), g(_g) {}
};
void performance_test() {
  vector<shape_s> shapes; // m k n groups
  int batch_size = 4 + 1;
  for (int i = 1; i < batch_size; i++) {
    shapes.emplace_back(i, 896, 448 * 6, 21);
    shapes.emplace_back(i, 896, 448 * 6, 14);
    shapes.emplace_back(i, 896, 448 * 6, 12);
    shapes.emplace_back(i, 896, 448 * 6, 6);
    shapes.emplace_back(i, 896, 448 * 6, 3);
    shapes.emplace_back(i, 896, 448 * 6, 2);
    shapes.emplace_back(i, 896, 448 * 6, 1);
    
    shapes.emplace_back(i, 896, 448 * 3, 7);
    shapes.emplace_back(i, 896, 448 * 3, 6);
    shapes.emplace_back(i, 896, 448 * 3, 3);
    shapes.emplace_back(i, 896, 448 * 3, 2);
    shapes.emplace_back(i, 896, 448 * 3, 1);
    
    shapes.emplace_back(i, 896, 1408, 11);
    shapes.emplace_back(i, 896, 1280, 10);
    shapes.emplace_back(i, 896, 1024, 8);
    shapes.emplace_back(i, 896, 512, 4);
    shapes.emplace_back(i, 896, 256, 2);
    shapes.emplace_back(i, 896, 128, 1);

    shapes.emplace_back(i, 448, 512, 4);
    shapes.emplace_back(i, 448, 512, 2);
    shapes.emplace_back(i, 448, 512, 1);

    shapes.emplace_back(i, 512, 256, 4);
    shapes.emplace_back(i, 512, 256, 2);
    shapes.emplace_back(i, 512, 256, 1);
  }

  float alpha = 1.f, beta = 0.f;
  matrix_op_t btran = matrix_op_t::NoTranspose;
  std::random_device r;
  std::default_random_engine generator(r());
      
  for (auto s : shapes) {
    double t1 = 0, t2 = 0;
    for (auto avx512 : {0,1}) {
      if (avx512)
        fbgemm::CPUIDInfo::GetCPUIDInfo().EnableAVX512f();
      else
        CPUIDInfo::GetCPUIDInfo().DisableAVX512f();
      
      aligned_vector<float> A(s.m * s.k, 0.f);
      aligned_vector<float> B(s.k * s.n, 0.f);
      aligned_vector<float> Cp(s.m * s.n, 0.f);
      
      randFill(A, 0.f, 4.f);
      randFill(B, 0.f, 4.f);

      // fbgemm fp16
      PackedGemmMatrixFP16 Bp(btran, s.k, s.n, alpha, B.data(), 512, s.g);

      static thread_local unique_ptr<std::array<float, 256 * 1024>> scratchpad(
          new std::array<float, 256 * 1024>());

      PackAMatrix packA(btran, s.m, s.k, A.data(), s.k, scratchpad->data(),
                        Bp.blockRowSize());
      
      chrono::time_point<chrono::system_clock> t_begin, t_end;     
      tbb::task_group tg;
#define NITER 10000
      for(int t = 0  ; t < NITER ; t++){
        t_begin = chrono::system_clock::now(); 
        for(int g = 0 ; g < s.g ; g++){
          tg.run([&, g] {
            auto rdata = Cp.data() + Bp.groupBeginCol(g);
            fbgemm::PackedGemmMatrixFP16 Bpg(Bp, g);
            fbgemm::cblas_gemm_compute(packA, Bpg, 0, rdata, Bp.groupShift(g));
          });
        }
        tg.wait();
        t_end = chrono::system_clock::now();
        if (!avx512)
          t1 += chrono::duration<double>(t_end - t_begin).count();
        else
          t2 += chrono::duration<double>(t_end - t_begin).count();
      }
      /*
      tbb::task_group tg;
      for (int g = 0; g < s.g; g++) {
        tg.run([&, g] {
          auto rdata = Cp.data() + Bp.groupBeginCol(g);
          fbgemm::PackedGemmMatrixFP16 Bpg(Bp, g);
          fbgemm::cblas_gemm_compute(packA, Bpg, 0, rdata, Bp.groupShift(g));
          // t[g] = chrono::duration<double>(chrono::system_clock::now() -
          // beg).count();
          // flag[g] = true;
        });
      }
      tg.wait();
      */
    }
    printf("m = %d k = %d n = %d g = %d rtf = %f rtf = %f -> speed up : %f\n", s.m, s.k,
           s.n, s.g, t1/NITER*24000, t2/NITER*24000,
           t1 / t2); // chrono::duration<double>(t_end - t_begin).count());
  }
}

int main(int argc, char **argv) { performance_test(); }
