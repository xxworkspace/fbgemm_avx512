/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "skylark/inference/blas/FbgemmFP16.h"
#include "skylark/inference/blas/cpuinfo.h"
#include "skylark/inference/blas/logging.h"

#include "skylark/inference/blas/FbgemmFP16UKernels.h"
#include "skylark/inference/blas/FbgemmFP16UKernelsAvx512.h"

#include <array>
#include <numeric>
#include <utility>
using namespace std;

namespace fbgemm {

// clang-format off
PackedGemmMatrixFP16::PackedGemmMatrixFP16(
      const matrix_op_t trans,
      const int nrow,
      const int ncol,
      const float alpha,
      const float* smat,
      const int brow,
      const std::vector<int>& groups): nrow_(nrow), ncol_(ncol), brow_(brow), group_size_(groups) {
  // clang-format on
  if(cpuinfo_has_x86_avx512f()){
    bcol_ = 32;
    for(auto g : groups){
      CHECK(g%32 == 0);
    }
  }
  else{
    bcol_ = 8;
    for(auto g : groups){
      CHECK(g%16 == 0);
    }
  }

  brow_ = nrow;
  // set up internal packing parameters
  nbrow_ = ((numRows() % blockRowSize()) == 0)
               ? (numRows() / blockRowSize())
               : ((numRows() + blockRowSize()) / blockRowSize());
  last_brow_ =
      ((nrow % blockRowSize()) == 0) ? blockRowSize() : (nrow % blockRowSize());
  nbcol_ = ((numCols() % blockColSize()) == 0)
               ? (numCols() / blockColSize())
               : ((numCols() + blockColSize()) / blockColSize());

  if (numCols() != blockColSize() * nbcol_) {
    LOG(WARNING) << "Packer warning: ncol(" << numCols()
                 << ") is not a multiple of internal block size ("
                 << blockColSize() << ")";
    LOG(WARNING)
        << "lefover is currently done via MKL: hence overhead will inccur";
  }

  // allocate and initialize packed memory
  const int padding = 1024;  // required by sw pipelined kernels
  size_ = (blockRowSize() * nbrow_) * (blockColSize() * nbcol_);
  if (posix_memalign((void**)&pmat_, 64,
                     matSize() * sizeof(float16) + padding)) {
    LOG(ERROR) << "Allocate memory fail.";
    return;
  }
  if (numGroups() > 1) {
    CHECK_EQ(std::accumulate(group_size_.begin(), group_size_.end(), 0),
             numCols());
    group_pmat_.resize(numGroups());
    CHECK_EQ(numCols() % blockColSize(), 0);
    int offset = 0;
    for (auto g = 0; g < numGroups(); g++) {
      group_pmat_[g] = pmat_ + (blockRowSize() * nbrow_) * offset;
      CHECK_EQ(group_size_.at(g) % blockColSize(), 0);
      offset += group_size_.at(g);
    }
  }

  // copy source matrix into packed matrix
  this->packFromSrc(trans, alpha, smat);
}

// clang-format off
PackedGemmMatrixFP16::PackedGemmMatrixFP16(
    const matrix_op_t trans,
    const int nrow,
    const int ncol,
    const uint16* smat,
    const int brow,
    const int num_groups): nrow_(nrow), ncol_(ncol), brow_(brow),
                             group_size_(num_groups, ncol / num_groups) {
  // clang-format on
  CHECK(trans == matrix_op_t::NoTranspose);
  if(cpuinfo_has_x86_avx512f()){
    bcol_ = 32;
    CHECK(ncol%(num_groups*32) == 0);
  }
  else{
    bcol_ = 8;
    CHECK(ncol%(num_groups*16) == 0);
  }

  brow_ = nrow;

  CHECK_EQ(numRows() % blockRowSize(), 0);
  CHECK_EQ(numCols() % blockColSize(), 0);

  // set up internal packing parameters
  nbrow_ = numRows() / blockRowSize();
  last_brow_ = blockRowSize();
  nbcol_ = numCols() / blockColSize();
  size_ = (blockRowSize() * nbrow_) * (blockColSize() * nbcol_);

  copied_ = true;
  pmat_ = reinterpret_cast<float16*>(const_cast<uint16*>(smat));
  if (numGroups() > 1) {
    CHECK_EQ(std::accumulate(group_size_.begin(), group_size_.end(), 0),
             numCols());
    group_pmat_.resize(numGroups());
    CHECK_EQ(numCols() % blockColSize(), 0);
    int offset = 0;
    for (auto g = 0; g < numGroups(); g++) {
      group_pmat_[g] = pmat_ + (blockRowSize() * nbrow_) * offset;
      CHECK_EQ(group_size_.at(g) % blockColSize(), 0);
      offset += group_size_.at(g);
    }
  }
  this->generateGroupBarrierAndShift();
}

int PackedGemmMatrixFP16::addr(const int r_, const int c_) const {
  int r = (int)r_;
  int c = (int)c_;

  int block_row_id = r / blockRowSize(),
      brow_offset = (block_row_id * nbcol_) * (blockRowSize() * blockColSize());
  int block_col_id = c / blockColSize(),
      bcol_offset = block_col_id * ((block_row_id != nbrow_ - 1)
                                        ? (blockRowSize() * blockColSize())
                                        : (last_brow_ * blockColSize()));
  int block_offset = brow_offset + bcol_offset;
  int inblock_offset = r % blockRowSize() * blockColSize() + c % blockColSize();

  int index = block_offset + inblock_offset;
  CHECK(index < matSize());
  return index;
}

int PackedGemmMatrixFP16::groupaddr(const int r, const int c,
                                    const int gnbcol) const {
  int block_row_id = r / blockRowSize(),
      brow_offset = (block_row_id * gnbcol) * (blockRowSize() * blockColSize());
  int block_col_id = c / blockColSize(),
      bcol_offset = block_col_id * ((block_row_id != nbrow_ - 1)
                                        ? (blockRowSize() * blockColSize())
                                        : (last_brow_ * blockColSize()));
  int block_offset = brow_offset + bcol_offset;
  int inblock_offset = r % blockRowSize() * blockColSize() + c % blockColSize();
  int index = block_offset + inblock_offset;
  return index;
}

void PackedGemmMatrixFP16::packFromSrc(const matrix_op_t trans,
                                       const float alpha, const float* smat) {
  this->generateGroupBarrierAndShift();
  bool tr = (trans == matrix_op_t::Transpose);

  if ((numCols() % blockColSize() == 0) && (numGroups() > 1)) {
    // speed up pack
    CHECK_EQ(tr, false);
    CHECK_EQ(numCols() % (blockColSize()*numGroups()), 0);
    for (int i = 0; i < numRows(); i++) {
      int gid = 0;
      int gnbcol = groupNumCols(gid) / blockColSize();
      int of = i * numCols();
      for (int j = 0; j < numCols(); j += blockColSize()) {
        if (j >= group_barrier_.at(gid + 1)) {
          gid += 1;
          CHECK_EQ(groupNumCols(gid) % blockColSize(), 0);
          gnbcol = groupNumCols(gid) / blockColSize();
        }
        auto grpidx = groupaddr(i, j - group_barrier_.at(gid), gnbcol);
        // CHECK_LT(grpidx, nbrow_ * blockRowSize() * groupNumCols(gid));

        auto gptr = group_pmat_[gid] + grpidx;
        auto sptr = smat + (of + j);

        for (int k = 0; k < blockColSize(); ++k) {
          gptr[k] = tconv(alpha * (sptr[k]), pmat_[0]);
        }
      }
    }
    return;
  }

  // pack
  for (int i = 0; i < numRows(); i++) {
    int gid = 0;
    int gnbcol = groupNumCols(gid) / blockColSize();
    for (int j = 0; j < numCols(); j++) {
      float16 v = tconv(alpha * ((tr == false) ? smat[i * numCols() + j]
                                               : smat[i + numRows() * j]),
                        pmat_[0]);
      if (numGroups() > 1) {
        if (j >= group_barrier_.at(gid + 1)) {
          gid += 1;
          gnbcol = groupNumCols(gid) / blockColSize();
        }
        auto gptr = group_pmat_[gid];
        // update group
        gptr[groupaddr(i, j - group_barrier_.at(gid), gnbcol)] = v;
      } else {
        pmat_[addr(i, j)] = v;
      }
    }
  }
}

const float16& PackedGemmMatrixFP16::operator()(const int r,
                                                const int c) const {
  int a = addr(r, c);
  CHECK(r < numRows());
  CHECK(c < numCols());
  CHECK(a < this->matSize());
  return pmat_[a];
}

float16* PackedGemmMatrixFP16::group(const int g) const {
  CHECK(g < numGroups());
  if (numGroups() > 1) return group_pmat_[g];
  return pmat_;
}

void PackedGemmMatrixFP16::generateGroupBarrierAndShift() {
  group_barrier_.resize(numGroups() + 1);
  group_barrier_[0] = 0;
  std::partial_sum(group_size_.begin(), group_size_.end(),
                   group_barrier_.begin() + 1);
  group_shift_.clear();
  for (auto s : group_size_) {
    group_shift_.push_back(numCols() - s);
  }
}

/// class that performs packing of matrix in
/// row-major or col-major format into
/// internal packed blocked-row major format

/// Todo: make it fast with AVX2 transpose
inline void PackA(int nrow, int ncol, const float* from, int ldim, float* to) {
  transpose_simd(nrow, ncol, from, ldim, to, nrow);
}

// clang-format off
struct KernelInfo {
  using knl_ptr = funcptr_fp16;
  // optimized kernels to cover all cases
  static constexpr array<knl_ptr, 15> avx256_kernel = {
      {nullptr,
       gemmkernel_1x2_AVX256_fA0fB0fC0,
       gemmkernel_2x2_AVX256_fA0fB0fC0,
       gemmkernel_3x2_AVX256_fA0fB0fC0,
       gemmkernel_4x2_AVX256_fA0fB0fC0,
       gemmkernel_5x1_AVX256_fA0fB0fC0,
       gemmkernel_6x1_AVX256_fA0fB0fC0,
       gemmkernel_7x1_AVX256_fA0fB0fC0,
       gemmkernel_8x1_AVX256_fA0fB0fC0,
       gemmkernel_9x1_AVX256_fA0fB0fC0,
       gemmkernel_10x1_AVX256_fA0fB0fC0,
       gemmkernel_11x1_AVX256_fA0fB0fC0,
       gemmkernel_12x1_AVX256_fA0fB0fC0,
       gemmkernel_13x1_AVX256_fA0fB0fC0,
       gemmkernel_14x1_AVX256_fA0fB0fC0}};

  static constexpr array<knl_ptr, 15> avx512_kernel_x2 = {
      {nullptr,
       gemmkernel_1x2_AVX512_fA0fB0fC0,
       gemmkernel_2x2_AVX512_fA0fB0fC0,
       gemmkernel_3x2_AVX512_fA0fB0fC0,
       gemmkernel_4x2_AVX512_fA0fB0fC0,
       gemmkernel_5x2_AVX512_fA0fB0fC0,
       gemmkernel_6x2_AVX512_fA0fB0fC0,
       gemmkernel_7x2_AVX512_fA0fB0fC0,
       gemmkernel_8x2_AVX512_fA0fB0fC0,
       gemmkernel_9x2_AVX512_fA0fB0fC0,
       gemmkernel_10x2_AVX512_fA0fB0fC0,
       gemmkernel_11x2_AVX512_fA0fB0fC0,
       gemmkernel_12x2_AVX512_fA0fB0fC0,
       gemmkernel_13x2_AVX512_fA0fB0fC0,
       gemmkernel_14x2_AVX512_fA0fB0fC0}};

  static constexpr array<knl_ptr, 15> avx512_kernel_x4 = {
      {nullptr,
       gemmkernel_1x4_AVX512_fA0fB0fC0,
       gemmkernel_2x4_AVX512_fA0fB0fC0,
       gemmkernel_3x4_AVX512_fA0fB0fC0,
       gemmkernel_4x4_AVX512_fA0fB0fC0,
       gemmkernel_5x4_AVX512_fA0fB0fC0,
       gemmkernel_6x4_AVX512_fA0fB0fC0,
       gemmkernel_7x2_AVX512_fA0fB0fC0,
       gemmkernel_8x2_AVX512_fA0fB0fC0,
       gemmkernel_9x2_AVX512_fA0fB0fC0,
       gemmkernel_10x2_AVX512_fA0fB0fC0,
       gemmkernel_11x2_AVX512_fA0fB0fC0,
       gemmkernel_12x2_AVX512_fA0fB0fC0,
       gemmkernel_13x2_AVX512_fA0fB0fC0,
       gemmkernel_14x2_AVX512_fA0fB0fC0}};
  // autotuned kernel splits for various cases m = 1:mb_max
  // may need re-autotuning for new uarch
  static constexpr array<array<array<int, 2>, 2>, 121> partition = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    {
      {{ { 0, 0 }, { 0, 0 } } },
      {{ { 1, 1 }, { 0, 0 } } },
      {{ { 2, 1 }, { 0, 0 } } },
      {{ { 3, 1 }, { 0, 0 } } },
      {{ { 4, 1 }, { 0, 0 } } },
      {{ { 5, 1 }, { 0, 0 } } },
      {{ { 6, 1 }, { 0, 0 } } },
      {{ { 7, 1 }, { 0, 0 } } },
      {{ { 8, 1 }, { 0, 0 } } },
      {{ { 9, 1 }, { 0, 0 } } },
      {{ { 10, 1 }, { 0, 0 } } },
      {{ { 11, 1 }, { 0, 0 } } },
      {{ { 12, 1 }, { 0, 0 } } },
      {{ { 13, 1 }, { 0, 0 } } },
      {{ { 14, 1 }, { 0, 0 } } },
      {{ { 8, 1 }, { 7, 1 } } },
      {{ { 10, 1 }, { 6, 1 } } },
      {{ { 11, 1 }, { 6, 1 } } },
      {{ { 12, 1 }, { 6, 1 } } },
      {{ { 11, 1 }, { 8, 1 } } },
      {{ { 11, 1 }, { 9, 1 } } },
      {{ { 12, 1 }, { 9, 1 } } },
      {{ { 11, 2 }, { 0, 0 } } },
      {{ { 12, 1 }, { 11, 1 } } },
      {{ { 12, 2 }, { 0, 0 } } },
      {{ { 13, 1 }, { 12, 1 } } },
      {{ { 13, 2 }, { 0, 0 } } },
      {{ { 14, 1 }, { 13, 1 } } },
      {{ { 14, 2 }, { 0, 0 } } },
      {{ { 11, 2 }, { 7, 1 } } },
      {{ { 10, 3 }, { 0, 0 } } },
      {{ { 12, 2 }, { 7, 1 } } },
      {{ { 12, 2 }, { 8, 1 } } },
      {{ { 11, 3 }, { 0, 0 } } },
      {{ { 13, 2 }, { 8, 1 } } },
      {{ { 13, 2 }, { 9, 1 } } },
      {{ { 13, 2 }, { 10, 1 } } },
      {{ { 13, 2 }, { 11, 1 } } },
      {{ { 13, 2 }, { 12, 1 } } },
      {{ { 13, 3 }, { 0, 0 } } },
      {{ { 14, 2 }, { 12, 1 } } },
      {{ { 14, 2 }, { 13, 1 } } },
      {{ { 11, 3 }, { 9, 1 } } },
      {{ { 11, 3 }, { 10, 1 } } },
      {{ { 11, 4 }, { 0, 0 } } },
      {{ { 12, 3 }, { 9, 1 } } },
      {{ { 12, 3 }, { 10, 1 } } },
      {{ { 13, 3 }, { 8, 1 } } },
      {{ { 13, 3 }, { 9, 1 } } },
      {{ { 13, 3 }, { 10, 1 } } },
      {{ { 13, 3 }, { 11, 1 } } },
      {{ { 13, 3 }, { 12, 1 } } },
      {{ { 13, 4 }, { 0, 0 } } },
      {{ { 14, 3 }, { 11, 1 } } },
      {{ { 11, 4 }, { 10, 1 } } },
      {{ { 12, 4 }, { 7, 1 } } },
      {{ { 14, 4 }, { 0, 0 } } },
      {{ { 12, 4 }, { 9, 1 } } },
      {{ { 12, 4 }, { 10, 1 } } },
      {{ { 12, 4 }, { 11, 1 } } },
      {{ { 13, 4 }, { 8, 1 } } },
      {{ { 13, 4 }, { 9, 1 } } },
      {{ { 13, 4 }, { 10, 1 } } },
      {{ { 13, 4 }, { 11, 1 } } },
      {{ { 11, 5 }, { 9, 1 } } },
      {{ { 13, 5 }, { 0, 0 } } },
      {{ { 14, 4 }, { 10, 1 } } },
      {{ { 12, 5 }, { 7, 1 } } },
      {{ { 12, 5 }, { 8, 1 } } },
      {{ { 14, 4 }, { 13, 1 } } },
      {{ { 14, 5 }, { 0, 0 } } },
      {{ { 12, 5 }, { 11, 1 } } },
      {{ { 13, 5 }, { 7, 1 } } },
      {{ { 11, 6 }, { 7, 1 } } },
      {{ { 13, 5 }, { 9, 1 } } },
      {{ { 13, 5 }, { 10, 1 } } },
      {{ { 13, 5 }, { 11, 1 } } },
      {{ { 13, 5 }, { 12, 1 } } },
      {{ { 13, 6 }, { 0, 0 } } },
      {{ { 12, 6 }, { 7, 1 } } },
      {{ { 12, 6 }, { 8, 1 } } },
      {{ { 12, 6 }, { 9, 1 } } },
      {{ { 12, 6 }, { 10, 1 } } },
      {{ { 12, 6 }, { 11, 1 } } },
      {{ { 12, 7 }, { 0, 0 } } },
      {{ { 13, 6 }, { 7, 1 } } },
      {{ { 13, 6 }, { 8, 1 } } },
      {{ { 13, 6 }, { 9, 1 } } },
      {{ { 13, 6 }, { 10, 1 } } },
      {{ { 13, 6 }, { 11, 1 } } },
      {{ { 13, 6 }, { 12, 1 } } },
      {{ { 13, 7 }, { 0, 0 } } },
      {{ { 12, 7 }, { 8, 1 } } },
      {{ { 12, 7 }, { 9, 1 } } },
      {{ { 14, 6 }, { 10, 1 } } },
      {{ { 12, 7 }, { 11, 1 } } },
      {{ { 13, 7 }, { 5, 1 } } },
      {{ { 13, 7 }, { 6, 1 } } },
      {{ { 13, 7 }, { 7, 1 } } },
      {{ { 13, 7 }, { 8, 1 } } },
      {{ { 13, 7 }, { 9, 1 } } },
      {{ { 13, 7 }, { 10, 1 } } },
      {{ { 13, 7 }, { 11, 1 } } },
      {{ { 13, 7 }, { 12, 1 } } },
      {{ { 12, 8 }, { 8, 1 } } },
      {{ { 12, 8 }, { 9, 1 } } },
      {{ { 12, 8 }, { 10, 1 } } },
      {{ { 12, 8 }, { 11, 1 } } },
      {{ { 12, 9 }, { 0, 0 } } },
      {{ { 11, 9 }, { 10, 1 } } },
      {{ { 13, 8 }, { 6, 1 } } },
      {{ { 13, 8 }, { 7, 1 } } },
      {{ { 13, 8 }, { 8, 1 } } },
      {{ { 13, 8 }, { 9, 1 } } },
      {{ { 13, 8 }, { 10, 1 } } },
      {{ { 13, 8 }, { 11, 1 } } },
      {{ { 12, 9 }, { 8, 1 } } },
      {{ { 13, 9 }, { 0, 0 } } },
      {{ { 12, 9 }, { 10, 1 } } },
      {{ { 12, 9 }, { 11, 1 } } },
      {{ { 12, 10 }, { 0, 0 } } }
     }
  };
};
constexpr array<KernelInfo::knl_ptr, 15> KernelInfo::avx256_kernel;
constexpr array<KernelInfo::knl_ptr, 15> KernelInfo::avx512_kernel_x2;
constexpr array<KernelInfo::knl_ptr, 15> KernelInfo::avx512_kernel_x4;

constexpr array<array<array<int, 2>, 2>, 121> KernelInfo::partition;
// clang-format on

// clang-format off
PackAMatrix::PackAMatrix(matrix_op_t trans,
                         int nrow,
                         int ncol,
                         const float* smat,
                         int ld,
                         float* pmat,
                         int bcol) : nrow_(nrow), ncol_(ncol), ld_(ld), bcol_(bcol), pmat_(pmat) {
  // clang-format on
  // ground truth
  CHECK(cpuinfo_has_x86_fma3());
  CHECK(cpuinfo_has_x86_f16c());
  CHECK(trans == matrix_op_t::NoTranspose);

  size_ = bufferSize(numRows(), numCols());
  CHECK_EQ(pack(smat), bufferSize(numRows(), numCols()));
}

int PackAMatrix::pack(const float* smat) {
  // constants
  const int m = numRows(), k = numCols();
  if ((m == 1) && (offset_.size() > 0)) {
    // int offset = bufferSize(numRows(), numCols());
    // memcpy(pmat_, smat, offset * sizeof(float));
    // return offset;
    pmat_ = const_cast<float*>(smat);
    return size();
  }

  const int mb_max = 120;
  int offset = 0;

  for (auto m0 = 0; m0 < m; m0 += mb_max) {
    int mb = std::min(mb_max, m - m0);
    for (auto k_ind = 0; k_ind < k; k_ind += blockColSize()) {
      const int kb = std::min(blockColSize(), numCols() - k_ind);
      auto m1 = m0;
      for (auto c = 0; c < 2; c++) {
        auto kernel_nrows = KernelInfo::partition[mb][c][0];
        auto nkernel_nrows = KernelInfo::partition[mb][c][1];

        auto m_start = m1, m_end = m1 + kernel_nrows * nkernel_nrows;
        for (auto m2 = m_start; m2 < m_end; m2 += kernel_nrows) {
          PackA(kernel_nrows, kb, &smat[m2 * ld() + k_ind], k, pmat_ + offset);

          // (m2, k_ind)
          offset_[m2 * k + k_ind] = offset;
          offset += kernel_nrows * kb;
        }
        m1 += kernel_nrows * nkernel_nrows;
      }
    }
  }
  return offset;
}

// autotuned kernel splits for various cases m = 1:mb_max
// clang-format off
void cblas_gemm_compute(
    const PackAMatrix& A,
    const PackedGemmMatrixFP16& Bp,
    const float beta,
    float* C,
    const int shift) {
  // clang-format on
  // constants
  const int m = A.numRows(), n = Bp.numCols(), k = Bp.numRows(),
            ldc = n + shift;
  const int mb_max = 120;
  const int simd_width = Bp.blockColSize();
  const int kernel_ncol_blocks = 1;
  const int kernel_ncols = kernel_ncol_blocks * simd_width;

  GemmParams gp;
  const array<KernelInfo::knl_ptr, 15>* kernel;
  if(cpuinfo_has_x86_avx512f()){
      if(n%64 == 0)
        kernel = &(KernelInfo::avx512_kernel_x4);
      else
        kernel = &(KernelInfo::avx512_kernel_x2);
  }
  else
    kernel = &(KernelInfo::avx256_kernel);
  
  for (auto m0 = 0; m0 < m; m0 += mb_max) {
    int mb = std::min(mb_max, m - m0);
    // CHECK(mb < KernelInfo::partition.size());
    for (auto k_ind = 0; k_ind < k; k_ind += Bp.blockRowSize()) {
      // set up proper accumulation to avoid "Nan" problem
      float beta_;
      uint64_t accum;
      if (k_ind == 0) {
        // accumulate of beta != 0.0
        // do not!!! accumulate otherwise
        beta_ = beta;
        accum = (beta_ == 0.0f) ? 0 : 1;
      } else {
        // always accumulate with beta_ = 1.0f
        beta_ = 1.0f;
        accum = 1;
      }

      const int kb = std::min(Bp.blockRowSize(), Bp.numRows() - k_ind);
      auto m1 = m0;

      for (auto c = 0; c < 2; c++) {
        auto kernel_nrows = KernelInfo::partition[mb][c][0];
        auto nkernel_nrows = KernelInfo::partition[mb][c][1];

        auto m_start = m1, m_end = m1 + kernel_nrows * nkernel_nrows;
        for (auto m2 = m_start; m2 < m_end; m2 += kernel_nrows) {
          int nbcol = n / Bp.blockColSize();
          gp.k = kb;
          gp.A = A.at(m2, k_ind);
          gp.B = &(Bp(k_ind, 0));
          gp.beta = &beta_;
          gp.accum = accum;
          gp.C = &C[m2 * ldc];
          gp.ldc = ldc * sizeof(C[0]);
          gp.b_block_cols = nbcol;
          gp.b_block_size = gp.k * Bp.blockColSize() * sizeof(gp.B[0]);
          if ((n % Bp.blockColSize()) == 0){
            //KernelInfo::kernel[kernel_nrows](&gp);
            (*kernel)[kernel_nrows](&gp);
          }
          /*else{
            int last_blk_col = nbcol * Bp.blockColSize();
            if (nbcol) {
              KernelInfo::kernel[kernel_nrows](&gp);
            }
            // leftover
            int rem = n - last_blk_col;
            CHECK(rem < kernel_ncols);
            int b = (rem % simd_width) ? ((rem + simd_width) / simd_width)
                                       : (rem / simd_width);
            CHECK(b == 1);
            if ((rem % simd_width) == 0) {
              gp.B = &(Bp(k_ind, last_blk_col));
              gp.C = &C[m2 * ldc + last_blk_col];
              gp.b_block_cols = 1;
              KernelInfo::kernel[kernel_nrows](&gp);
            } else {
              // small temporary buffer
              //Avx512
              float c_tmp[16 * 24] = {0};
              CHECK((16 * 24) > kernel_nrows * kernel_ncols);

              gp.B = &(Bp(k_ind, last_blk_col));
              gp.C = c_tmp;
              gp.ldc = 8 * sizeof(C[0]);
              gp.b_block_cols = 1;
              KernelInfo::kernel[kernel_nrows](&gp);
              for (int i = 0; i < kernel_nrows; i++) {
                // Todo: use assembly
                for (int j = last_blk_col; j < n; j++) {
                  CHECK(i * 8 + (j - last_blk_col) <
                        sizeof(c_tmp) / sizeof(c_tmp[0]));
                  if (accum == 0) {
                    C[(m2 + i) * ldc + j] = c_tmp[i * 8 + (j - last_blk_col)];
                  } else {
                    C[(m2 + i) * ldc + j] = beta_ * C[(m2 + i) * ldc + j] +
                                            c_tmp[i * 8 + (j - last_blk_col)];
                  }
                }
              }
            }
          }*/
        }
        m1 += kernel_nrows * nkernel_nrows;
      }
    }
  }
}

// clang-format off
void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixFP16& Bp,
    const float beta,
    float* C,
    const int shift,
    float* buffer) {
  // clang-format on
  PackAMatrix packedA(transa, m, Bp.numRows(), A, Bp.numRows(), buffer,
                      Bp.blockRowSize());
  cblas_gemm_compute(packedA, Bp, beta, C, shift);
}

// clang-format off
void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixFP16& Bp,
    const float beta,
    float* C,
    const int shift) {
    // private scratchpad storage
    static thread_local unique_ptr<std::array<float, 256 * 1024>> scratchpad(
        new std::array<float, 256 * 1024>());
    CHECK_LE(PackAMatrix::bufferSize(m, Bp.numRows()), 256 * 1024);
    cblas_gemm_compute(transa, m, A, Bp, beta, C, shift, scratchpad->data());
  // clang-format on
}

}  // namespace fbgemm
