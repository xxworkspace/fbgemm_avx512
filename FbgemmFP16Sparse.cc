/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "skylark/inference/blas/FbgemmFP16Sparse.h"
#include "skylark/inference/blas/cpuinfo.h"
#include "skylark/inference/blas/logging.h"

#include <immintrin.h>
#include <array>
#include <numeric>
#include <utility>
using namespace std;

namespace fbgemm {

/// class that performs packing of matrix in
/// row-major or col-major format into
/// internal packed blocked-row major format

/// Todo: make it fast with AVX2 transpose
inline void PackA(int nrow, int ncol, const float* from, int ldim, float* to) {
  transpose_simd(nrow, ncol, from, ldim, to, nrow);
}

// clang-format off
PackedGemmBlockSparseMatrixFP16::PackedGemmBlockSparseMatrixFP16(
      const matrix_op_t trans,
      const float alpha,
      const float* smat,
      const int* mask,  // block sparse mask: sparse = 0 dense = 1
      const int nbrow,
      const int nbcol,
      const int block_height,
      const int block_width,
      const int groups)
      : nrow_(nbrow * block_height),
        ncol_(nbcol * block_width),
        brow_(block_height),
        bcol_(block_width),
        nbrow_(nbrow),
        nbcol_(nbcol),
        ngroup_(groups) {
  // clang-format on
  CHECK(block_height % 8 == 0);
  CHECK(block_width == 1);
  CHECK(trans == matrix_op_t::NoTranspose);
  for (int i = 0; i < numGroups(); ++i) {
    group_size_.push_back(numCols() / numGroups());
  }

  size_ = 0;
  for (auto r = 0; r < nbrow_; r++) {
    for (auto c = 0; c < nbcol_; c++) size_ += mask[r * nbcol_ + c];
  }
  // allocate and initialize packed memory
  const int padding = 1024;  // required by sw pipelined kernels
  posix_memalign((void**)&pmat_, 64,
                 size_ * sizeof(FLOAT) * block_height * block_width + padding);
  // copy source matrix into packed matrix
  this->packFromSrc(trans, alpha, smat, mask);
}

PackedGemmBlockSparseMatrixFP16::PackedGemmBlockSparseMatrixFP16(
    const PackedGemmBlockSparseMatrixFP16& other, int gid /*group id*/) {
  copied_ = true;
  bcol_ = other.blockColSize();  // hardwired

  nrow_ = other.numRows();
  ncol_ = other.groupNumCols();
  brow_ = other.blockRowSize();
  ngroup_ = 1;
  pmat_ = other.group(gid);
  nbrow_ = other.numBlockRows();
  nbcol_ = other.groupNumCols() / other.blockColSize();

  block_colptr_.insert(
      block_colptr_.end(),
      other.block_colptr_.begin() + gid * other.groupNumCols(),
      other.block_colptr_.begin() + (gid + 1) * other.groupNumCols());
  offsets_.insert(offsets_.end(),
                  other.offsets_.begin() + gid * other.groupNumCols(),
                  other.offsets_.begin() + (gid + 1) * other.groupNumCols());
  size_ = (blockRowSize() * nbrow_) * (blockColSize() * nbcol_);
  group_size_.push_back(ncol_);
  this->generateGroupBarrierAndShift();
}

void PackedGemmBlockSparseMatrixFP16::packFromSrc(const matrix_op_t trans,
                                                  const float alpha,
                                                  const float* smat,
                                                  const int* mask) {
  this->generateGroupBarrierAndShift();
  // pack
  offsets_.resize(numBlockCols());

  int of = 0;
  for (auto cb = 0; cb < numBlockCols(); cb++) {
    block_colptr_.push_back(pmat_ + of);
    for (auto rb = 0; rb < numBlockRows(); rb++) {
      int dense = mask[rb * numBlockCols() + cb];
      if (dense) {
        offsets_[cb].push_back(rb * blockRowSize());
        int bidx = rb * blockRowSize() * numCols() + cb * blockColSize();
        for (auto r = 0; r < blockRowSize(); r++) {
          pmat_[of++] = alpha * smat[bidx + r * numCols()];
          // cpu_float2half_rn(alpha * smat[bidx + r * ncol_ + r]);
        }
      }
    }
  }

  CHECK(block_colptr_.size() == numBlockCols());
}

FLOAT* PackedGemmBlockSparseMatrixFP16::group(const int g) const {
  CHECK(g < numGroups());
  if (numGroups() > 1) {
    return const_cast<FLOAT*>(block_colptr_.at(g * groupNumCols()));
  }
  return pmat_;
}

void PackedGemmBlockSparseMatrixFP16::generateGroupBarrierAndShift() {
  group_barrier_.resize(numGroups() + 1);
  group_barrier_[0] = 0;
  std::partial_sum(group_size_.begin(), group_size_.end(),
                   group_barrier_.begin() + 1);
  group_shift_.clear();
  for (auto s : group_size_) {
    group_shift_.push_back(numCols() - s);
  }
}

// autotuned kernel splits for various cases m = 1:mb_max
// clang-format off
void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmBlockSparseMatrixFP16& Bp,
    const float beta,
    float* C, const int shift) {
  // clang-format on
  // constants
  const int n = Bp.numCols(), k = Bp.numRows(), ldc = n;
  const int mb_max = 120;
  constexpr int simd_width = 8;
  constexpr int kernel_ncol_blocks = 1;
  constexpr int kernel_ncols = kernel_ncol_blocks * simd_width;

  // // version 1
  // for(auto mi = 0; mi < m; mi++) {
  //   auto aptr = A + mi * k;
  //   auto cptr = C + mi * ldc;
  //   for (auto kb = 0; kb < Bp.numBlockRows(); kb++) {
  //     aptr = A + mi * k + kb * Bp.blockRowSize();
  //     const auto &offsets = Bp.nnzColOffsets(kb);
  //     auto bptr = Bp.blockRowPtr(kb);
  //     for (auto of : offsets) {
  //       aptr = A + mi * k + kb * Bp.blockRowSize();
  //       for (auto bb = 0; bb < Bp.blockRowSize(); bb += 8) {
  //         float sum = ((aptr[0] * bptr[0]) + (aptr[1] * bptr[1])) + ((aptr[2]
  //         * bptr[2]) + (aptr[3] * bptr[3])) +
  //                     ((aptr[4] * bptr[4]) + (aptr[5] * bptr[5])) + ((aptr[6]
  //                     * bptr[6]) + (aptr[7] * bptr[7]));
  //         cptr[of] += sum;
  //         aptr += 8;
  //         bptr += 8;
  //       }
  //     }
  //   }
  // }

  // // version 2
  int nb = Bp.blockRowSize() / 8;
  float sum[8] __attribute__((aligned(64)));
  if (nb == 1) {
    for (auto mi = 0; mi < m; mi++) {
      auto aptr = A + mi * k;
      auto bptr = Bp.data();  // TODO implement batch parallel version
      auto cptr = C + mi * (n + shift);
      for (auto c = 0; c < n; c++) {
        __m256 sumV = _mm256_setzero_ps();
        for (auto of : Bp.Offsets(c)) {
          sumV = _mm256_fmadd_ps(_mm256_loadu_ps(aptr + of),
                                 _mm256_loadu_ps(bptr), sumV);
          bptr += 8;
        }
        _mm256_store_ps(sum, sumV);
        cptr[c] = beta * cptr[c] + (sum[0] + sum[1]) + (sum[2] + sum[3]) +
                  (sum[4] + sum[5]) + (sum[6] + sum[7]);
      }
    }
  } else if (nb == 2) {
    for (auto mi = 0; mi < m; mi++) {
      auto aptr = A + mi * k;
      auto bptr = Bp.data();  // TODO implement batch parallel version
      auto cptr = C + mi * (n + shift);
      for (auto c = 0; c < n; c++) {
        __m256 sumV = _mm256_setzero_ps();
        for (auto of : Bp.Offsets(c)) {
          sumV = _mm256_fmadd_ps(_mm256_loadu_ps(aptr + of),
                                 _mm256_loadu_ps(bptr), sumV);

          sumV = _mm256_fmadd_ps(_mm256_loadu_ps(aptr + of + 8),
                                 _mm256_loadu_ps(bptr + 8), sumV);
          bptr += 16;
        }
        _mm256_store_ps(sum, sumV);
        cptr[c] = beta * cptr[c] + (sum[0] + sum[1]) + (sum[2] + sum[3]) +
                  (sum[4] + sum[5]) + (sum[6] + sum[7]);
      }
    }
  } else {
    for (auto mi = 0; mi < m; mi++) {
      auto aptr = A + mi * k;
      auto bptr = Bp.data();  // TODO implement batch parallel version
      auto cptr = C + mi * (n + shift);
      for (auto c = 0; c < n; c++) {
        __m256 sumV = _mm256_setzero_ps();
        for (auto of : Bp.Offsets(c)) {
          for (auto bi = 0; bi < nb; bi++) {
            __m256 ra = _mm256_loadu_ps(aptr + of + bi * 8);
            __m256 rb = _mm256_loadu_ps(bptr);
            sumV = _mm256_fmadd_ps(ra, rb, sumV);

            bptr += 8;
          }
        }
        _mm256_store_ps(sum, sumV);
        cptr[c] = beta * cptr[c] + (sum[0] + sum[1]) + (sum[2] + sum[3]) +
                  (sum[4] + sum[5]) + (sum[6] + sum[7]);
      }
    }
  }

  // // Final version
  // GemmParams gp;
  // for (auto m0 = 0; m0 < m; m0 += mb_max) {
  //   int mb = std::min(mb_max, m - m0);
  //   assert(mb < KernelInfo::partition.size());
  //   {
  //     uint64_t accum = (beta == 0.0f) ? 0 : 1;
  //     const int kb = Bp.numBlockRows();
  //     auto m1 = 0;
  //     for (auto c = 0; c < 2; c++) {
  //       auto kernel_nrows = KernelInfo::partition[mb][c][0];
  //       auto nkernel_nrows = KernelInfo::partition[mb][c][1];

  //       auto m_start = m1, m_end = m1 + kernel_nrows * nkernel_nrows;
  //       for (auto m2 = m_start; m2 < m_end; m2 += kernel_nrows) {
  //         assert(kernel_nrows * kb < scratchpad->size());
  //         PackA(kernel_nrows, kb, &A[m2 * k], k, scratchpad->data());

  //         int nbcol = Bp.numBlockCols();
  //         gp.k = kb;
  //         gp.A = scratchpad->data();
  //         gp.B = Bp.data();
  //         gp.beta = &beta;
  //         gp.accum = accum;
  //         gp.C = &C[m2 * ldc];
  //         gp.ldc = ldc * sizeof(C[0]);
  //         gp.b_block_cols = nbcol;
  //         gp.b_block_size = gp.k * Bp.blockColSize() * sizeof(gp.B[0]);
  //         // do computation
  //         KernelInfo::kernel[kernel_nrows](&gp);
  //       }
  //       m1 += kernel_nrows * nkernel_nrows;
  //     }
  //   }
  // }
}

}  // namespace fbgemm
