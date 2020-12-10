/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

// WARNING: this is a legacy fp16 fbgemm implementation and will soon be
// upgraded to match with new fbgemm interface.

#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>

#include "skylark/inference/blas/Types.h"
#include "skylark/inference/blas/Utils.h"
//#include "skylark/infernece/blas/cpuid_info.h"

namespace fbgemm {

/// class that performs packing of matrix in
/// row-major format into
/// internal packed blocked-row major format
class PackedGemmMatrixFP16 {
 public:
  // takes smat input mamtrix in row-major format;
  // and packs it into gemm-friendly blocked format;
  // allocate space and sets up all the internal variables;
  // also premultiplies by alpha during packing
  // brow_ contains tile size along k dimension
  // and also is # of fmas updates into int16 container
  // before flushing into fp32
  // the smaller the brow_, the higher overhead
  // of flushing is
  /**
   * @param groups when groups > 1, we compute groups number of GEMMs each
   *               multiplies A matrix with B.cols/B.groups.
   */
  // clang-format off
  PackedGemmMatrixFP16(
      const matrix_op_t trans,
      const int nrow,
      const int ncol,
      const float alpha,
      const float* smat,
      const int brow = 512,
      const std::vector<int>& groups={});

  // clang-format off
  PackedGemmMatrixFP16(
      const matrix_op_t trans,
      const int nrow,
      const int ncol,
      const float alpha,
      const float* smat,
      const int brow,
      const int num_groups) : PackedGemmMatrixFP16(trans, nrow, ncol, alpha, smat, brow, std::vector<int>(num_groups, ncol / num_groups))  {}
  // clang-format on

  // clang-format off
  PackedGemmMatrixFP16(
      const matrix_op_t trans,
      const int nrow,
      const int ncol,
      const uint16* smat,
      const int brow = 512,
      const int num_groups=4);
  // clang-format on

  PackedGemmMatrixFP16(const PackedGemmMatrixFP16& other,
                       int gid /*group id*/) {
    copied_ = true;

    nrow_ = other.numRows();
    ncol_ = other.groupNumCols(gid);
    brow_ = other.blockRowSize();
    bcol_ = other.blockColSize();  // hardwired

    // set up internal packing parameters
    nbrow_ = ((numRows() % blockRowSize()) == 0)
                 ? (numRows() / blockRowSize())
                 : ((numRows() + blockRowSize()) / blockRowSize());
    last_brow_ = ((numRows() % blockRowSize()) == 0)
                     ? blockRowSize()
                     : (numRows() % blockRowSize());
    nbcol_ = ((numCols() % blockColSize()) == 0)
                 ? (numCols() / blockColSize())
                 : ((numCols() + blockColSize()) / blockColSize());

    pmat_ = other.group(gid);
    size_ = (blockRowSize() * nbrow_) * (blockColSize() * nbcol_);
    group_size_.push_back(ncol_);
    this->generateGroupBarrierAndShift();
  }

  ~PackedGemmMatrixFP16() {
    if (!copied_) {
      free(pmat_);
    }
  }

  // protected:
  // blocked row-major format address arithmetic
  inline int addr(const int r_, const int c_) const;

  inline int groupaddr(const int r, const int c, const int gnbcol) const;

  void packFromSrc(const matrix_op_t trans, const float alpha,
                   const float* smat);

  const float16& operator()(const int r, const int c) const;

  float16* group(const int g) const;

  const float16* data() const { return pmat_; }

  int matSize() const { return size_; }
  inline int numRows() const { return nrow_; }
  inline int numCols() const { return ncol_; }
  inline int blockRowSize() const { return brow_; }
  inline int blockColSize() const { return bcol_; }

  inline int numGroups() const { return group_size_.size(); }
  inline int groupNumCols(int i) const {
    if (numGroups() > 1) {
      return group_size_.at(i);
    }
    return numCols();
  }
  inline int groupBeginCol(int i) const { return group_barrier_.at(i); }
  inline int groupShift(int i) const { return group_shift_.at(i); }

 private:
  void generateGroupBarrierAndShift();

  int nrow_, ncol_;
  int brow_, last_brow_, bcol_;
  int nbrow_, nbcol_;
  int size_;

  float16* pmat_;
  std::vector<float16*> group_pmat_;
  std::vector<int> group_size_;
  // generated
  std::vector<int> group_barrier_;
  std::vector<int> group_shift_;
  bool copied_ = false;
};

class PackAMatrix {
 public:
  PackAMatrix() = delete;  // no default constructor

  ~PackAMatrix() {}
  // clang-format off
  PackAMatrix(
      matrix_op_t trans,
      int nrow,
      int ncol,
      const float* smat,
      int ld,
      float* pmat,
      int bcol);
  // clang-format on
  /**
   * @return True if this is used as A matrix.
   */
  static constexpr bool isA() { return true; }

  static int bufferSize(int nrow, int ncol) { return nrow * ncol; };
  int size() const { return size_; }

  const float* at(int i, int j) const { return pmat_ + addr(i, j); }

  inline int numRows() const { return nrow_; }
  inline int numCols() const { return ncol_; }
  inline int blockColSize() const { return bcol_; }
  inline int ld() const { return ld_; }

  // pack again
  int pack(const float* smat);
 private:
  /**
   * @return Offset of the element in the packed matrix that was at (i, j) in
   *         the source matrix.
   */
  int addr(int i, int j) const { return offset_.at(i * ncol_ + j); };

  int nrow_;
  int ncol_;
  int ld_;
  int bcol_;

  int size_;
  float* pmat_;
  std::unordered_map<int, int> offset_;
};

/**
 * restrictions: transa == CblasNoTrans
 */
// clang-format off
void cblas_gemm_compute(
    const PackAMatrix& A,
    const PackedGemmMatrixFP16& Bp,
    const float beta,
    float* C,
    const int shift = 0);
void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixFP16& Bp,
    const float beta,
    float* C,
    const int shift = 0);
void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixFP16& Bp,
    const float beta,
    float* C,
    const int shift,
    float* buffer);
// clang-format on

};  // namespace fbgemm
