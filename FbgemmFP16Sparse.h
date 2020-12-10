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
#include <vector>

#include "skylark/inference/blas/Types.h"
#include "skylark/inference/blas/Utils.h"

namespace fbgemm {

using FLOAT = float;

/// class that performs packing of matrix in
/// row-major format into
/// internal packed block-sparse col-major format
class PackedGemmBlockSparseMatrixFP16 {
 public:
  // takes smat input mamtrix in row-major format;
  // and packs it into sparse gemm-friendly blocked format;
  // allocate space and sets up all the internal variables;
  // also premultiplies by alpha during packing
  // brow_ contains tile size along k dimension
  // and also is # of fmas updates into int16 container
  // before flushing into fp32
  // clang-format off
  PackedGemmBlockSparseMatrixFP16(
      const matrix_op_t trans,
      const float alpha,
      const float* smat,
      const int* mask,  // block sparse mask: sparse = 0 dense = 1
      const int nbrow,
      const int nbcol,
      const int block_height = 16,
      const int block_width = 1,
      const int groups = 1);
  // clang-format on

  PackedGemmBlockSparseMatrixFP16(const PackedGemmBlockSparseMatrixFP16& other,
                                  int gid /*group id*/);

  ~PackedGemmBlockSparseMatrixFP16() {
    if (!copied_) {
      free(pmat_);
    }
  }

  void packFromSrc(const matrix_op_t trans, const float alpha,
                   const float* smat, const int* mask);

  const FLOAT* data() const { return pmat_; }

  const FLOAT* blockColPtr(const int c) const { return block_colptr_.at(c); }

  FLOAT* group(const int g) const;

  int matSize() const { return size_; }
  int numRows() const { return nrow_; }
  int numBlockRows() const { return nbrow_; }
  int numCols() const { return ncol_; }
  int numBlockCols() const { return nbcol_; }
  int numGroups() const { return ngroup_; }
  int groupNumCols() const { return numCols() / numGroups(); }

  const std::vector<int>& Offsets(int c) const { return offsets_.at(c); }

  inline int blockRowSize() const { return brow_; }
  inline int blockColSize() const { return bcol_; }
  inline int groupBeginCol(int i) const { return group_barrier_.at(i); }
  inline int groupShift(int i) const { return group_shift_.at(i); }

 private:
  void generateGroupBarrierAndShift();

  int nrow_, ncol_;
  int brow_, bcol_;
  int nbrow_, nbcol_;
  int ngroup_;
  uint64_t size_;
  FLOAT* pmat_;
  std::vector<const FLOAT*> block_colptr_;
  std::vector<std::vector<int> > offsets_;

  // generated
  std::vector<int> group_size_;
  std::vector<int> group_barrier_;
  std::vector<int> group_shift_;
  // sub matrix
  bool copied_ = false;
};

/**
 * restrictions: transa == CblasNoTrans
 */
// clang-format off
void cblas_gemm_compute(const matrix_op_t transa,
                        const int m,
                        const float* A,
                        const PackedGemmBlockSparseMatrixFP16& Bp,
                        const float beta,
                        float* C,
                        const int shift=0);
// clang-format on

};  // namespace fbgemm
