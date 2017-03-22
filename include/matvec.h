#ifndef __MATVEC_H__
#define __MATVEC_H__

#include "types.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#define EPSILON 1.0E-8

static inline bool scalar_equal(double a, double b) {
  return (fabs(a - b) < EPSILON) ? true : false;
}

static inline bool vector_equal(const vector va, const vector vb) {
  return (scalar_equal(va[0], vb[0]) && scalar_equal(va[1], vb[1]) &&
          scalar_equal(va[2], vb[2]))
             ? true
             : false;
}

static inline void vector_zero(vector vec) { vec[0] = vec[1] = vec[2] = 0.0; }

static inline void vector_sum(const vector va, const vector vb, vector result) {
  result[0] = va[0] + vb[0];
  result[1] = va[1] + vb[1];
  result[2] = va[2] + vb[2];
}

static inline void vector_sum_inplace(vector va, const vector vb) {
  va[0] += vb[0];
  va[1] += vb[1];
  va[2] += vb[2];
}

static inline void vector_sub(const vector va, const vector vb, vector result) {
  result[0] = va[0] - vb[0];
  result[1] = va[1] - vb[1];
  result[2] = va[2] - vb[2];
}

static inline void vector_sub_inplace(vector va, const vector vb) {
  va[0] -= vb[0];
  va[1] -= vb[1];
  va[2] -= vb[2];
}

static inline void vector_scalar_mul(const vector va, double s, vector result) {
  result[0] = va[0] * s;
  result[1] = va[1] * s;
  result[2] = va[2] * s;
}

static inline void vector_scalar_mul_inplace(vector va, double s) {
  va[0] = va[0] * s;
  va[1] = va[1] * s;
  va[2] = va[2] * s;
}

static inline void vector_scalar_mul_sum_inplace(const vector va, double s,
                                                 vector result) {
  result[0] += va[0] * s;
  result[1] += va[1] * s;
  result[2] += va[2] * s;
}

static inline double vector_dot(const vector va, const vector vb) {
  return va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2];
}

static inline double vector_len(const vector dist) {
  return sqrt(vector_dot(dist, dist));
}

static inline void vector_dup(const vector vec, vector result) {
  result[0] = vec[0];
  result[1] = vec[1];
  result[2] = vec[2];
}

static inline void vector_normalize(const vector vec, vector result) {
  double norm = vector_len(vec);
  result[0] = vec[0] / norm;
  result[1] = vec[1] / norm;
  result[2] = vec[2] / norm;
}

static inline void vector_normalize_inplace(vector vec) {
  vector tmp;
  vector_normalize(vec, tmp);
  vector_dup(tmp, vec);
}

static inline void vector_cross(const vector va, const vector vb,
                                vector result) {
  result[0] = va[1] * vb[2] - va[2] * vb[1];
  result[1] = va[2] * vb[0] - va[0] * vb[2];
  result[2] = va[0] * vb[1] - va[1] * vb[0];
}

static inline void matrix_dup(const matrix mat, matrix result) {
  result[0] = mat[0];
  result[1] = mat[1];
  result[2] = mat[2];
  result[3] = mat[3];
  result[4] = mat[4];
  result[5] = mat[5];
  result[6] = mat[6];
  result[7] = mat[7];
  result[8] = mat[8];
}

static inline double matrix_det(const matrix mat) {
  double det = 0.0;
  det += mat[0] * (mat[4] * mat[8] - mat[5] * mat[7]);
  det -= mat[1] * (mat[3] * mat[8] - mat[5] * mat[6]);
  det += mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);
  return det;
}

static inline void matrix_inv(const matrix mat, matrix result) {
  double det = matrix_det(mat);
  result[0] = (mat[4] * mat[8] - mat[5] * mat[7]) / det;
  result[1] = (mat[2] * mat[7] - mat[1] * mat[8]) / det;
  result[2] = (mat[1] * mat[5] - mat[2] * mat[4]) / det;
  result[3] = (mat[5] * mat[6] - mat[3] * mat[8]) / det;
  result[4] = (mat[0] * mat[8] - mat[2] * mat[6]) / det;
  result[5] = (mat[2] * mat[3] - mat[0] * mat[5]) / det;
  result[6] = (mat[3] * mat[7] - mat[4] * mat[6]) / det;
  result[7] = (mat[1] * mat[6] - mat[0] * mat[7]) / det;
  result[8] = (mat[0] * mat[4] - mat[1] * mat[3]) / det;
}

static inline void matrix_trans(const matrix mat, matrix result) {
  result[0] = mat[0];
  result[1] = mat[3];
  result[2] = mat[6];
  result[3] = mat[1];
  result[4] = mat[4];
  result[5] = mat[7];
  result[6] = mat[2];
  result[7] = mat[5];
  result[8] = mat[8];
}

static inline void matrix_vector_mul(const matrix mat, const vector vec,
                                     vector result) {
  result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

static inline void matrix_trans_vector_mul(const matrix mat, const vector vec,
                                           vector result) {
  result[0] = mat[0] * vec[0] + mat[3] * vec[1] + mat[6] * vec[2];
  result[1] = mat[1] * vec[0] + mat[4] * vec[1] + mat[7] * vec[2];
  result[2] = mat[2] * vec[0] + mat[5] * vec[1] + mat[8] * vec[2];
}

static inline void matrix_matrix_mul(const matrix matA, const matrix matB,
                                     matrix result) {
  result[0] = matA[0] * matB[0] + matA[1] * matB[3] + matA[2] * matB[6];
  result[1] = matA[0] * matB[1] + matA[1] * matB[4] + matA[2] * matB[7];
  result[2] = matA[0] * matB[2] + matA[1] * matB[5] + matA[2] * matB[8];
  result[3] = matA[3] * matB[0] + matA[4] * matB[3] + matA[5] * matB[6];
  result[4] = matA[3] * matB[1] + matA[4] * matB[4] + matA[5] * matB[7];
  result[5] = matA[3] * matB[2] + matA[4] * matB[5] + matA[5] * matB[8];
  result[6] = matA[6] * matB[0] + matA[7] * matB[3] + matA[8] * matB[6];
  result[7] = matA[6] * matB[1] + matA[7] * matB[4] + matA[8] * matB[7];
  result[8] = matA[6] * matB[2] + matA[7] * matB[5] + matA[8] * matB[8];
}

static inline void matrix_matrix_trans_mul(const matrix matA, const matrix matB,
                                           matrix result) {
  result[0] = matA[0] * matB[0] + matA[1] * matB[1] + matA[2] * matB[2];
  result[1] = matA[0] * matB[3] + matA[1] * matB[4] + matA[2] * matB[5];
  result[2] = matA[0] * matB[6] + matA[1] * matB[7] + matA[2] * matB[8];
  result[3] = matA[3] * matB[0] + matA[4] * matB[1] + matA[5] * matB[2];
  result[4] = matA[3] * matB[3] + matA[4] * matB[4] + matA[5] * matB[5];
  result[5] = matA[3] * matB[6] + matA[4] * matB[7] + matA[5] * matB[8];
  result[6] = matA[6] * matB[0] + matA[7] * matB[1] + matA[8] * matB[2];
  result[7] = matA[6] * matB[3] + matA[7] * matB[4] + matA[8] * matB[5];
  result[8] = matA[6] * matB[6] + matA[7] * matB[7] + matA[8] * matB[8];
}

static inline void rotate_vector_inplace(const matrix rotate_mat, vector vec) {
  vector tmp;
  matrix_vector_mul(rotate_mat, vec, tmp);
  vector_dup(tmp, vec);
}

#endif
