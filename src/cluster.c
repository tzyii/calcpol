#include <stdio.h>

#include "efp.h"
#include "matvec.h"
#include "types.h"
#include "utils.h"

static void skiplines(FILE *fp, size_t nlines) {
  char buf[1024];
  for (size_t n = 0; n < nlines; ++n) {
    fgets(buf, sizeof(buf), fp);
  }
}

static fragment *fragment_alloc(const std_efp_info *efp_ptr) {
  fragment *pfrag = galloc(sizeof(fragment));
  pfrag->std_ptr = efp_ptr;
  pfrag->mult_ptr = galloc(efp_ptr->n_mult_points * sizeof(mult_point));
  pfrag->pol_ptr = galloc(efp_ptr->n_pol_points * sizeof(pol_point));
  return pfrag;
}

static void gen_masscenter(fragment *pfrag) {
  vector tmpvec;
  double w_atom, w_total = 0.0;

  for (size_t k = 0; k < pfrag->std_ptr->n_atoms; ++k) {
    w_atom = GET_ATOM_MASS(pfrag->std_ptr->mult_ptr[k].nuclear);
    w_total += w_atom;
    vector_scalar_mul(pfrag->mult_ptr[k].position, w_atom, tmpvec);
    vector_sum_inplace(pfrag->masscenter, tmpvec);
  }
  vector_scalar_mul_inplace(pfrag->masscenter, 1 / w_total);
}

static inline void quadrupole2tensor(const quadrupole quad, matrix mat) {
  mat[0] = quad[0];
  mat[4] = quad[1];
  mat[8] = quad[2];
  mat[1] = mat[3] = quad[3];
  mat[2] = mat[6] = quad[4];
  mat[5] = mat[7] = quad[5];
}

static inline void tensor2quadrupole_corrected(const matrix mat,
                                               quadrupole quad) {
  double trace = mat[0] + mat[4] + mat[8];

  quad[0] = 1.5 * mat[0] - 0.5 * trace;
  quad[1] = 1.5 * mat[4] - 0.5 * trace;
  quad[2] = 1.5 * mat[8] - 0.5 * trace;
  quad[3] = 1.5 * mat[1];
  quad[4] = 1.5 * mat[2];
  quad[5] = 1.5 * mat[5];
}

static inline void rotate_tensor_rank2(const matrix rotate, const matrix mat,
                                       matrix result) {
  size_t a1, b1, a2, b2;

  for (a2 = 0; a2 < 3; ++a2) {
    for (b2 = 0; b2 < 3; ++b2) {
      result[a2 * 3 + b2] = 0.0;
    }
  }

  for (a1 = 0; a1 < 3; ++a1) {
    for (b1 = 0; b1 < 3; ++b1) {
      for (a2 = 0; a2 < 3; ++a2) {
        for (b2 = 0; b2 < 3; ++b2) {
          result[a2 * 3 + b2] +=
              mat[a1 * 3 + b1] * rotate[a2 * 3 + a1] * rotate[b2 * 3 + b1];
        }
      }
    }
  }
}

static inline void octupole2tensor(const octupole oct, tensor_rank3 thr) {
  thr[0] = oct[0];
  thr[13] = oct[1];
  thr[26] = oct[2];
  thr[1] = thr[3] = thr[9] = oct[3];
  thr[2] = thr[6] = thr[18] = oct[4];
  thr[4] = thr[10] = thr[12] = oct[5];
  thr[14] = thr[16] = thr[22] = oct[6];
  thr[8] = thr[20] = thr[24] = oct[7];
  thr[17] = thr[23] = thr[25] = oct[8];
  thr[5] = thr[7] = thr[11] = thr[15] = thr[19] = thr[21] = oct[9];
}

static inline void tensor2octupole_corrected(const tensor_rank3 thr,
                                             octupole oct) {
  double trace_x = thr[0] + thr[4] + thr[8];
  double trace_y = thr[3] + thr[13] + thr[23];
  double trace_z = thr[2] + thr[14] + thr[26];

  oct[0] = 2.5 * thr[0] - 1.5 * trace_x;
  oct[1] = 2.5 * thr[13] - 1.5 * trace_y;
  oct[2] = 2.5 * thr[26] - 1.5 * trace_z;
  oct[3] = 2.5 * thr[1] - 0.5 * trace_y;
  oct[4] = 2.5 * thr[2] - 0.5 * trace_z;
  oct[5] = 2.5 * thr[4] - 0.5 * trace_x;
  oct[6] = 2.5 * thr[14] - 0.5 * trace_z;
  oct[7] = 2.5 * thr[8] - 0.5 * trace_x;
  oct[8] = 2.5 * thr[17] - 0.5 * trace_y;
  oct[9] = 2.5 * thr[5];
}

static inline void rotate_tensor_rank3(const matrix rotate,
                                       const tensor_rank3 thr,
                                       tensor_rank3 result) {
  size_t a1, b1, c1, a2, b2, c2;

  for (a2 = 0; a2 < 3; ++a2) {
    for (b2 = 0; b2 < 3; ++b2) {
      for (c2 = 0; c2 < 3; ++c2) {
        result[a2 * 9 + b2 * 3 + c2] = 0.0;
      }
    }
  }

  for (a1 = 0; a1 < 3; ++a1) {
    for (b1 = 0; b1 < 3; ++b1) {
      for (c1 = 0; c1 < 3; ++c1) {
        for (a2 = 0; a2 < 3; ++a2) {
          for (b2 = 0; b2 < 3; ++b2) {
            for (c2 = 0; c2 < 3; ++c2) {
              result[a2 * 9 + b2 * 3 + c2] +=
                  thr[a1 * 9 + b1 * 3 + c1] * rotate[a2 * 3 + a1] *
                  rotate[b2 * 3 + b1] * rotate[c2 * 3 + c1];
            }
          }
        }
      }
    }
  }
}

static void fragment_fillinfo(fragment *pfrag) {
  matrix rotate, mat, axis, tmpmat;
  tensor_rank3 thr, tmpthr;
  vector va, vb, vc, vd, translate;
  size_t m, n, idx, a1, b1, c1, a2, b2, c2;
  double trace;
  bond *bond_ptr;

  gen_masscenter(pfrag);

  vector_sub(pfrag->mult_ptr[pfrag->std_ptr->principal[1]].position,
             pfrag->mult_ptr[pfrag->std_ptr->principal[0]].position, va);
  vector_sub(pfrag->mult_ptr[pfrag->std_ptr->principal[3]].position,
             pfrag->mult_ptr[pfrag->std_ptr->principal[2]].position, vb);

  vector_normalize(va, vc);
  vector_scalar_mul(vc, vector_dot(vb, vc), va);
  vector_sub(vb, va, vd);
  vector_normalize(vd, va);
  vector_cross(vc, va, vb);

  axis[0] = vc[0];
  axis[1] = va[0];
  axis[2] = vb[0];
  axis[3] = vc[1];
  axis[4] = va[1];
  axis[5] = vb[1];
  axis[6] = vc[2];
  axis[7] = va[2];
  axis[8] = vb[2];

  matrix_inv(pfrag->std_ptr->axis, mat);
  matrix_matrix_mul(axis, mat, rotate);

  vector_dup(pfrag->std_ptr->masscenter, va);
  rotate_vector_inplace(rotate, va);
  vector_sum(pfrag->masscenter, va, translate);

  for (n = pfrag->std_ptr->n_atoms; n < pfrag->std_ptr->n_mult_points; ++n) {
    bond_ptr = pfrag->std_ptr->bond_ptr + (n - pfrag->std_ptr->n_atoms);
    vector_sum(pfrag->mult_ptr[bond_ptr->index[0]].position,
               pfrag->mult_ptr[bond_ptr->index[1]].position,
               pfrag->mult_ptr[n].position);
    vector_scalar_mul_inplace(pfrag->mult_ptr[n].position, 0.5);
  }

  for (n = 0; n < pfrag->std_ptr->n_pol_points; ++n) {
    matrix_vector_mul(rotate, pfrag->std_ptr->pol_ptr[n].position,
                      pfrag->pol_ptr[n].position);
    vector_sum_inplace(pfrag->pol_ptr[n].position, translate);
  }

  for (n = 0; n < pfrag->std_ptr->n_mult_points; ++n) {
    pfrag->mult_ptr[n].nuclear = pfrag->std_ptr->mult_ptr[n].nuclear;
    pfrag->mult_ptr[n].monopole = pfrag->std_ptr->mult_ptr[n].monopole;
    pfrag->mult_ptr[n].screen = pfrag->std_ptr->mult_ptr[n].screen;
    matrix_vector_mul(rotate, pfrag->std_ptr->mult_ptr[n].dipole,
                      pfrag->mult_ptr[n].dipole);

    quadrupole2tensor(pfrag->std_ptr->mult_ptr[n].quadrupole, tmpmat);
    rotate_tensor_rank2(rotate, tmpmat, mat);
    tensor2quadrupole_corrected(mat, pfrag->mult_ptr[n].quadrupole);

    octupole2tensor(pfrag->std_ptr->mult_ptr[n].octupole, tmpthr);
    rotate_tensor_rank3(rotate, tmpthr, thr);
    tensor2octupole_corrected(thr, pfrag->mult_ptr[n].octupole);
  }

  for (n = 0; n < pfrag->std_ptr->n_pol_points; ++n) {
    rotate_tensor_rank2(rotate, pfrag->std_ptr->pol_ptr[n].polarizability, mat);
    matrix_dup(mat, pfrag->pol_ptr[n].polarizability);
  }
}

void fragment_load(FILE *fp, box *pbox) {
  fragment *ptr;
  list_ptr fragments_list = NULL;
  char buf[1024], moletype[5];
  const char end[] = "end";
  const std_efp_info *efp_ptr;
  vector tmpvec;
  double a, b, c, alpha, beta, gamma;
  size_t n;

  pbox->n_frag = 0;

  skiplines(fp, 4);
  fgets(buf, sizeof(buf), fp);
  sscanf(buf, "%*s %lf %lf %lf %lf %lf %lf %*s", &a, &b, &c, &alpha, &beta,
         &gamma);
  a *= BOHR;
  b *= BOHR;
  c *= BOHR;
  alpha *= DEG2RAD;
  beta *= DEG2RAD;
  gamma *= DEG2RAD;
  pbox->lattice[0] = a;
  pbox->lattice[3] = pbox->lattice[6] = 0.0;
  pbox->lattice[1] = b * cos(gamma);
  pbox->lattice[4] = b * sin(gamma);
  pbox->lattice[7] = 0.0;
  pbox->lattice[2] = cos(beta);
  pbox->lattice[5] = (cos(alpha) - cos(gamma) * cos(beta)) / sin(gamma);
  pbox->lattice[8] = sqrt(1.0 - pbox->lattice[2] * pbox->lattice[2] -
                          pbox->lattice[5] * pbox->lattice[5]);
  pbox->lattice[2] *= c;
  pbox->lattice[5] *= c;
  pbox->lattice[8] *= c;

  while (1) {
    fgets(buf, sizeof(buf), fp);
    if (strncmp(buf, end, sizeof(end) - 1) == 0) {
      break;
    }
    sscanf(buf, "%*s %lf %lf %lf %4s", tmpvec, tmpvec + 1, tmpvec + 2,
           moletype);
    efp_ptr = get_std_efp_info(moletype);
    ptr = fragment_alloc(efp_ptr);
    list_append(&fragments_list, ptr);
    pbox->n_frag += 1;
    for (n = 0; n < ptr->std_ptr->n_atoms; ++n) {
      vector_scalar_mul_inplace(tmpvec, BOHR);
      vector_dup(tmpvec, ptr->mult_ptr[n].position);
      fgets(buf, sizeof(buf), fp);
      if (n + 1 != ptr->std_ptr->n_atoms) {
        sscanf(buf, "%*s %lf %lf %lf %4s", tmpvec, tmpvec + 1, tmpvec + 2,
               moletype);
      }
    }
  }
  list_dump(fragments_list, sizeof(fragment),
            CAST_PTR(&(pbox->frag_ptr), void *));
  list_clear(&fragments_list);

  for (n = 0; n < pbox->n_frag; ++n) {
    fragment_fillinfo(pbox->frag_ptr + n);
  }
}
#if 0
static void fragment_translate(fragment *pfrag, const vector origin,
                               const matrix lattice) {
  size_t n;
  double tmp;
  vector dist, projected, move, tmpvec;
  matrix lattice_inv;

  vector_sub(pfrag->masscenter, origin, dist);
  matrix_inv(lattice, lattice_inv);
  matrix_vector_mul(lattice_inv, dist, projected);

  for (n = 0; n < 3; ++n) {
    tmp = modf(projected[n], &(move[n]));
    if (tmp < 0.0) {
      move[n] -= 1;
    }
  }
  matrix_vector_mul(lattice, move, tmpvec);
  vector_sum(origin, tmpvec, move);

  vector_sub_inplace(pfrag->masscenter, move);

  for (n = 0; n < pfrag->std_ptr->n_mult_points; ++n) {
    vector_sub_inplace(pfrag->mult_ptr[n].position, move);
  }

  for (n = 0; n < pfrag->std_ptr->n_pol_points; ++n) {
    vector_sub_inplace(pfrag->pol_ptr[n].position, move);
  }
}

static void set_box_origin(box *pbox, size_t frag_index) {
  size_t n;
  vector center;

  vector_dup(pbox->frag_ptr[frag_index].masscenter, center);
  for (n = 0; n < pbox->n_frag; ++n) {
    fragment_translate(pbox->frag_ptr + n, center, pbox->lattice);
  }
}
#else
void pseudo_set_box_origin(const box *pbox, size_t origin_index,
                           vector **pmove) {
  size_t i, k;
  const fragment *pfrag;
  double tmp;
  vector origin, dist, projected, move, tmpvec;
  matrix lattice_inv;

  *pmove = galloc(pbox->n_frag * sizeof(vector));
  vector_dup(pbox->frag_ptr[origin_index].masscenter, origin);
  matrix_inv(pbox->lattice, lattice_inv);

  for (i = 0; i < pbox->n_frag; ++i) {
    pfrag = pbox->frag_ptr + i;
    vector_sub(pfrag->masscenter, origin, dist);
    matrix_vector_mul(lattice_inv, dist, projected);

    for (k = 0; k < 3; ++k) {
      tmp = modf(projected[k], &(move[k]));
      if (tmp < 0.0) {
        move[k] -= 1;
      }
    }

    matrix_vector_mul(pbox->lattice, move, tmpvec);
    vector_sum(origin, tmpvec, (*pmove)[i]);
    vector_negative((*pmove)[i]);
  }
}
#endif

typedef struct {
  size_t n_cells;
  vector *cell_ptr;
} supercell;

static list_ptr scellptr = NULL;

static int lookup_cell(void *pelem, const void *key) {
  return vector_equal(*(CAST_PTR(pelem, vector)), *(CAST_PTR(key, vector))) ? 0
                                                                            : 1;
}

static void append_cell(list_ptr *plptr, const vector *pvec) {
  vector *ptr;

  if (list_lookup(*plptr, lookup_cell, pvec) == NULL) {
    ptr = galloc(sizeof(vector));
    (*ptr)[0] = (*pvec)[0];
    (*ptr)[1] = (*pvec)[1];
    (*ptr)[2] = (*pvec)[2];
    list_append(plptr, ptr);
  }
}

static inline void gen_cells(supercell *scptr, size_t cellsum) {
  list_ptr lptr = NULL;
  vector vec;
  size_t na, nb, nc;

  for (na = 0; na <= cellsum; ++na) {
    for (nb = 0; na + nb <= cellsum; ++nb) {
      nc = cellsum - na - nb;
      vec[0] = na;
      vec[1] = nb;
      vec[2] = nc;
      append_cell(&lptr, CAST_PTR(&vec, const vector));
      vec[0] = 0.0 - na;
      vec[1] = 0.0 - nb;
      vec[2] = 0.0 - nc;
      append_cell(&lptr, CAST_PTR(&vec, const vector));
      vec[0] = 0.0 - na;
      vec[1] = nb;
      vec[2] = nc;
      append_cell(&lptr, CAST_PTR(&vec, const vector));
      vec[0] = na;
      vec[1] = 0.0 - nb;
      vec[2] = 0.0 - nc;
      append_cell(&lptr, CAST_PTR(&vec, const vector));
      vec[0] = na;
      vec[1] = 0.0 - nb;
      vec[2] = nc;
      append_cell(&lptr, CAST_PTR(&vec, const vector));
      vec[0] = 0.0 - na;
      vec[1] = nb;
      vec[2] = 0.0 - nc;
      append_cell(&lptr, CAST_PTR(&vec, const vector));
      vec[0] = na;
      vec[1] = nb;
      vec[2] = 0.0 - nc;
      append_cell(&lptr, CAST_PTR(&vec, const vector));
      vec[0] = 0.0 - na;
      vec[1] = 0.0 - nb;
      vec[2] = nc;
      append_cell(&lptr, CAST_PTR(&vec, const vector));
    }
  }
  scptr->n_cells = list_length(lptr);
  list_dump(lptr, sizeof(vector), CAST_PTR(&(scptr->cell_ptr), void *));
  list_clear(&lptr);
}

static void free_foreach(void *pelem) {
  supercell *scptr = CAST_PTR(pelem, supercell);
  free(scptr->cell_ptr);
}

const supercell *get_supercell(size_t cellsum) {
  supercell *scptr;

  scptr = list_index(scellptr, cellsum);
  if (scptr == NULL) {
    for (size_t n = list_length(scellptr); n <= cellsum; ++n) {
      scptr = galloc(sizeof(supercell));
      gen_cells(scptr, n);
      list_append(&scellptr, scptr);
    }
  }
  return scptr;
}

void clear_scellptr(void) {
  list_foreach(scellptr, free_foreach);
  list_clear(&scellptr);
}

void gen_cluster(cluster *cls, const box *pbox, double radius,
                 size_t idx_center) {
  fragment *pfrag;
  vector *pmove_origin, center, pbc, tmpcenter;
  list_ptr lptr = NULL, scellptr;
  const supercell *scptr;
  pol_fragment *ptr;
  size_t n, k, len, cellsum, notfound;
  double r, r_min;

  cls->center = 0;
  cls->n_polfrags = 0;
  cls->n_polpoints = 0;
  cls->n_include = 0;
  cls->include_ptr = NULL;

  pseudo_set_box_origin(pbox, idx_center, &pmove_origin);

  for (n = 0; n < pbox->n_frag; ++n) {
    pfrag = pbox->frag_ptr + n;
    vector_sum(pfrag->masscenter, pmove_origin[n], center);

    cellsum = 0;
    notfound = 0;
    do {
      r_min = radius;
      scptr = get_supercell(cellsum);
      for (k = 0; k < scptr->n_cells; ++k) {
        matrix_vector_mul(pbox->lattice, scptr->cell_ptr[k], pbc);
        vector_sum(center, pbc, tmpcenter);
        r = vector_len(tmpcenter);
        if (r < radius) {
          ptr = galloc(sizeof(pol_fragment));
          ptr->original = pfrag;
          vector_sub(tmpcenter, pmove_origin[idx_center], ptr->masscenter);
          list_append(&lptr, ptr);
          if (scalar_equal(r, 0.0)) {
            cls->center = cls->n_polfrags;
          }
          cls->n_polfrags += 1;
          cls->n_polpoints += ptr->original->std_ptr->n_pol_points;
        }
        if (r_min > r) {
          r_min = r;
        }
      }
      if (r_min >= radius) {
        notfound += 1;
      }
      ++cellsum;
    } while (notfound <= 3);
  }

  cls->n_include = cls->n_polfrags;
  cls->include_ptr = galloc(cls->n_polfrags * sizeof(size_t));
  for (n = 0; n < cls->n_polfrags; ++n) {
    cls->include_ptr[n] = n;
  }

  free(pmove_origin);
  clear_scellptr();
  list_dump(lptr, sizeof(pol_fragment), CAST_PTR(&(cls->polfrag_ptr), void *));
  list_clear(&lptr);
}

void set_polarizable_include(cluster *cls, double radius) {
  cls->n_include = 0;

  for (size_t i = 0; i < cls->n_polfrags; ++i) {
    if (vector_dist(cls->polfrag_ptr[cls->center].masscenter,
                    cls->polfrag_ptr[i].masscenter) < radius) {
      cls->include_ptr[(cls->n_include)++] = i;
    }
  }
}

void modify_center_charge(cluster *cls, int charge) {
  char name[80], tmp[80];
  int old_charge = 0;
  size_t n, nread;
  const std_efp_info *efp_ptr;
  fragment *ptr;

  nread = sscanf(cls->polfrag_ptr[cls->center].original->std_ptr->name,
                 "%[^+-]%d", tmp, &old_charge);
  if (nread == 1) {
    old_charge = 0;
  }

  if (charge == old_charge) {
    return;
  }
  if (charge != 0) {
    snprintf(name, sizeof(name), "%s%+d", tmp, charge);
  } else {
    snprintf(name, sizeof(name), "%s", tmp);
  }
  efp_ptr = get_std_efp_info(name);
  ptr = fragment_alloc(efp_ptr);
  for (n = 0; n < ptr->std_ptr->n_atoms; ++n) {
    vector_dup(cls->polfrag_ptr[cls->center].original->mult_ptr[n].position,
               ptr->mult_ptr[n].position);
  }
  fragment_fillinfo(ptr);
  cls->polfrag_ptr[cls->center].original = ptr;
}
