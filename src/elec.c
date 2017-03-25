#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "cluster.h"
#include "matvec.h"
#include "types.h"
#include "utils.h"

static inline double quadrupole_sum(const quadrupole qua, const vector dist) {
  double sum = 0.0;
  sum += qua[0] * dist[0] * dist[0];
  sum += qua[1] * dist[1] * dist[1];
  sum += qua[2] * dist[2] * dist[2];
  sum += qua[3] * dist[0] * dist[1] * 2.0;
  sum += qua[4] * dist[0] * dist[2] * 2.0;
  sum += qua[5] * dist[1] * dist[2] * 2.0;
  return sum;
}

static inline double octupole_sum(const octupole oct, const vector dist) {
  double sum = 0.0;
  sum += oct[0] * dist[0] * dist[0] * dist[0];
  sum += oct[1] * dist[1] * dist[1] * dist[1];
  sum += oct[2] * dist[2] * dist[2] * dist[2];
  sum += oct[3] * dist[0] * dist[0] * dist[1] * 3.0;
  sum += oct[4] * dist[0] * dist[0] * dist[2] * 3.0;
  sum += oct[5] * dist[1] * dist[1] * dist[0] * 3.0;
  sum += oct[6] * dist[1] * dist[1] * dist[2] * 3.0;
  sum += oct[7] * dist[2] * dist[2] * dist[0] * 3.0;
  sum += oct[8] * dist[2] * dist[2] * dist[1] * 3.0;
  sum += oct[9] * dist[0] * dist[1] * dist[2] * 6.0;
  return sum;
}

static inline double dipole_quadrupole_sum(const dipole di,
                                           const quadrupole qua,
                                           const vector dist) {
  double sum = 0.0;
  sum += qua[0] * di[0] * dist[0];
  sum += qua[1] * di[1] * dist[1];
  sum += qua[2] * di[2] * dist[2];
  sum += qua[3] * (di[0] * dist[1] + di[1] * dist[0]);
  sum += qua[4] * (di[0] * dist[2] + di[2] * dist[0]);
  sum += qua[5] * (di[1] * dist[2] + di[2] * dist[1]);
  return sum;
}

static inline double quadrupole_quadrupole_dot(const quadrupole quaa,
                                               const quadrupole quab) {
  double sum = 0.0;
  sum += quaa[0] * quab[0];
  sum += quaa[1] * quab[1];
  sum += quaa[2] * quab[2];
  sum += quaa[3] * quab[3] * 2.0;
  sum += quaa[4] * quab[4] * 2.0;
  sum += quaa[5] * quab[5] * 2.0;
  return sum;
}

static inline double quadrupole_quadrupole_sum(const quadrupole quaa,
                                               const quadrupole quab,
                                               const vector dist) {
  vector quaad = {quaa[0] * dist[0] + quaa[3] * dist[1] + quaa[4] * dist[2],
                  quaa[3] * dist[0] + quaa[1] * dist[1] + quaa[5] * dist[2],
                  quaa[4] * dist[0] + quaa[5] * dist[1] + quaa[2] * dist[2]};
  vector quabd = {quab[0] * dist[0] + quab[3] * dist[1] + quab[4] * dist[2],
                  quab[3] * dist[0] + quab[1] * dist[1] + quab[5] * dist[2],
                  quab[4] * dist[0] + quab[5] * dist[1] + quab[2] * dist[2]};
  return vector_dot(quaad, quabd);
}

static inline double calc_nuclear_screen_damping(double s_other, double r) {
  return 1.0 - exp(-s_other * r);
}

static inline double calc_screen_damping(double s_other, double s_me,
                                         double r) {
  if (fabs(s_other - s_me) < 1.0E-5) {
    return 1.0 - (1.0 + 0.5 * s_other * r) * exp(-s_other * r);
  } else {
    return 1.0 -
           exp(-s_other * r) * s_me * s_me / (s_me * s_me - s_other * s_other) -
           exp(-s_me * r) * s_other * s_other /
               (s_other * s_other - s_me * s_me);
  }
}

static double calc_nuclear_nuclear_energy(const pol_fragment *pfragA,
                                          const pol_fragment *pfragB) {
  double energy = 0.0, r;
  const mult_point *pmi, *pmj;
  vector dist, moveA, moveB, moveAB;
  size_t i, j;

  vector_sub(pfragA->masscenter, pfragA->original->masscenter, moveA);
  vector_sub(pfragB->masscenter, pfragB->original->masscenter, moveB);
  vector_sub(moveB, moveA, moveAB);

  for (i = 0; i < pfragA->original->std_ptr->n_atoms; ++i) {
    pmi = pfragA->original->mult_ptr + i;
    for (j = 0; j < pfragB->original->std_ptr->n_atoms; ++j) {
      pmj = pfragB->original->mult_ptr + j;
      vector_sub(pmj->position, pmi->position, dist);
      vector_sum_inplace(dist, moveAB);
      r = vector_len(dist);
      energy += pmi->nuclear * pmj->nuclear / r;
    }
  }
  return energy;
}

static double calc_nuclear_multipole_energy(const pol_fragment *pfragA,
                                            const pol_fragment *pfragB) {
  double energy = 0.0, damping, r, r2, r3, r5, r7;
  const mult_point *pmi, *pmj;
  vector dist, moveA, moveB, moveAB;
  size_t i, j;

  vector_sub(pfragA->masscenter, pfragA->original->masscenter, moveA);
  vector_sub(pfragB->masscenter, pfragB->original->masscenter, moveB);
  vector_sub(moveB, moveA, moveAB);

  for (i = 0; i < pfragA->original->std_ptr->n_atoms; ++i) {
    pmi = pfragA->original->mult_ptr + i;
    for (j = 0; j < pfragB->original->std_ptr->n_mult_points; ++j) {
      pmj = pfragB->original->mult_ptr + j;
      vector_sub(pmj->position, pmi->position, dist);
      vector_sum_inplace(dist, moveAB);
      r = vector_len(dist);
      r2 = r * r;
      r3 = r2 * r;
      r5 = r2 * r3;
      r7 = r2 * r5;
      damping = calc_nuclear_screen_damping(pmj->screen, r);
      energy += damping * pmi->nuclear * pmj->monopole / r;
      energy -= pmi->nuclear / r3 * vector_dot(pmj->dipole, dist);
      energy += pmi->nuclear / r5 * quadrupole_sum(pmj->quadrupole, dist);
      energy -= pmi->nuclear / r7 * octupole_sum(pmj->octupole, dist);
    }
  }
  return energy;
}

static double calc_multipole_multipole_energy(const pol_fragment *pfragA,
                                              const pol_fragment *pfragB) {
  double energy = 0.0, damping, si, sj, r, r2, r3, r5, r7, r9;
  double dsumi, dsumj, qsumi, qsumj;
  const mult_point *pmi, *pmj;
  vector dist, dist_opposite, moveA, moveB, moveAB;
  size_t i, j;

  vector_sub(pfragA->masscenter, pfragA->original->masscenter, moveA);
  vector_sub(pfragB->masscenter, pfragB->original->masscenter, moveB);
  vector_sub(moveB, moveA, moveAB);

  for (i = 0; i < pfragA->original->std_ptr->n_mult_points; ++i) {
    pmi = pfragA->original->mult_ptr + i;
    for (j = 0; j < pfragB->original->std_ptr->n_mult_points; ++j) {
      pmj = pfragB->original->mult_ptr + j;
      vector_sub(pmj->position, pmi->position, dist);
      vector_sum_inplace(dist, moveAB);
      vector_scalar_mul(dist, -1.0, dist_opposite);
      r = vector_len(dist);
      r2 = r * r;
      r3 = r2 * r;
      r5 = r2 * r3;
      r7 = r2 * r5;
      r9 = r2 * r7;
      damping = calc_screen_damping(pmj->screen, pmi->screen, r);
      dsumi = vector_dot(pmi->dipole, dist_opposite);
      dsumj = vector_dot(pmj->dipole, dist);
      qsumi = quadrupole_sum(pmi->quadrupole, dist_opposite);
      qsumj = quadrupole_sum(pmj->quadrupole, dist);

      energy += damping * (pmi->monopole) * (pmj->monopole) / r;

      energy -= (pmi->monopole) / r3 * dsumj;
      energy += (pmi->monopole) / r5 * qsumj;
      energy -= (pmi->monopole) / r7 * octupole_sum(pmj->octupole, dist);
      energy -= (pmj->monopole) / r3 * dsumi;
      energy += (pmj->monopole) / r5 * qsumi;
      energy -=
          (pmj->monopole) / r7 * octupole_sum(pmi->octupole, dist_opposite);

      energy +=
          vector_dot(pmi->dipole, pmj->dipole) / r3 + 3.0 * dsumi * dsumj / r5;

      energy +=
          -5.0 / r7 * dsumi * qsumj -
          2.0 / r5 * dipole_quadrupole_sum(pmi->dipole, pmj->quadrupole, dist);
      energy += -5.0 / r7 * dsumj * qsumi -
                2.0 / r5 * dipole_quadrupole_sum(pmj->dipole, pmi->quadrupole,
                                                 dist_opposite);

      energy += (2.0 / r5 * quadrupole_quadrupole_dot(pmi->quadrupole,
                                                      pmj->quadrupole) -
                 20.0 / r7 * quadrupole_quadrupole_sum(pmi->quadrupole,
                                                       pmj->quadrupole, dist) +
                 35.0 / r9 * qsumi * qsumj) /
                3.0;
    }
  }
  return energy;
}

void alloc_pol_mem(cluster *pcls) {
  for (size_t n = 0; n < pcls->n_polfrags; ++n) {
    if (pcls->polfrag_ptr[n].pol_status == NULL) {
      pcls->polfrag_ptr[n].pol_status =
          galloc(sizeof(pol_point_status) *
                 pcls->polfrag_ptr[n].original->std_ptr->n_pol_points);
    }
  }
}

void free_pol_mem(cluster *pcls) {
  for (size_t n = 0; n < pcls->n_polfrags; ++n) {
    free(pcls->polfrag_ptr[n].pol_status);
  }
}

void init_pol_mem(cluster *pcls) {
  for (size_t n = 0; n < pcls->n_polfrags; ++n) {
    memset(pcls->polfrag_ptr[n].pol_status, 0,
           sizeof(pol_point_status) *
               pcls->polfrag_ptr[n].original->std_ptr->n_pol_points);
  }
}

void init_dipole_induced(cluster *pcls) {
  for (size_t i = 0; i < pcls->n_polfrags; ++i) {
    for (size_t j = 0; j < pcls->polfrag_ptr[i].original->std_ptr->n_pol_points;
         ++j) {
      vector_zero(pcls->polfrag_ptr[i].pol_status[j].dipole);
    }
  }
}

void init_field_induced(cluster *pcls) {
  for (size_t i = 0; i < pcls->n_polfrags; ++i) {
    for (size_t j = 0; j < pcls->polfrag_ptr[i].original->std_ptr->n_pol_points;
         ++j) {
      vector_zero(pcls->polfrag_ptr[i].pol_status[j].field_induced);
    }
  }
}

void init_field_immut(cluster *pcls) {
  for (size_t i = 0; i < pcls->n_polfrags; ++i) {
    for (size_t j = 0; j < pcls->polfrag_ptr[i].original->std_ptr->n_pol_points;
         ++j) {
      vector_zero(pcls->polfrag_ptr[i].pol_status[j].field_immut);
    }
  }
}

void calc_mult_field(pol_fragment *pfragA, const pol_fragment *pfragB) {
  double r, r2, r3, r5, r7, damping;
  double dsum, qsum;
  const pol_point *ppi;
  const mult_point *pmj;
  vector dist, field, moveA, moveB, moveAB;
  vector field_undamped, tmpvec;
  size_t n, i, j;

  vector_sub(pfragA->masscenter, pfragA->original->masscenter, moveA);
  vector_sub(pfragB->masscenter, pfragB->original->masscenter, moveB);
  vector_sub(moveA, moveB, moveAB);
  for (i = 0; i < pfragA->original->std_ptr->n_pol_points; ++i) {
    vector_zero(field);
    ppi = pfragA->original->pol_ptr + i;
    for (j = 0; j < pfragB->original->std_ptr->n_atoms; ++j) {
      pmj = pfragB->original->mult_ptr + j;
      vector_sub(ppi->position, pmj->position, dist);
      vector_sum_inplace(dist, moveAB);
      r = vector_len(dist);
      r2 = r * r;
      r3 = r2 * r;

      damping = 1.0 - exp(-0.6 * r2) * (1.0 + 0.6 * r2);

      vector_scalar_mul_sum_inplace(dist, damping * pmj->nuclear / r3, field);
    }
    for (j = 0; j < pfragB->original->std_ptr->n_mult_points; ++j) {
      pmj = pfragB->original->mult_ptr + j;
      vector_sub(ppi->position, pmj->position, dist);
      vector_sum_inplace(dist, moveAB);
      r = vector_len(dist);
      r2 = r * r;
      r3 = r2 * r;
      r5 = r2 * r3;
      r7 = r2 * r5;

      damping = 1.0 - exp(-0.6 * r2) * (1.0 + 0.6 * r2);

      vector_zero(field_undamped);

      dsum = vector_dot(pmj->dipole, dist);
      qsum = quadrupole_sum(pmj->quadrupole, dist);

      vector_scalar_mul_sum_inplace(dist, pmj->monopole / r3, field_undamped);

      vector_scalar_mul_sum_inplace(dist, 3.0 / r5 * dsum, field_undamped);
      vector_scalar_mul_sum_inplace(pmj->dipole, -1 / r3, field_undamped);

      tmpvec[0] = pmj->quadrupole[0] * dist[0] + pmj->quadrupole[3] * dist[1] +
                  pmj->quadrupole[4] * dist[2];
      tmpvec[1] = pmj->quadrupole[3] * dist[0] + pmj->quadrupole[1] * dist[1] +
                  pmj->quadrupole[5] * dist[2];
      tmpvec[2] = pmj->quadrupole[4] * dist[0] + pmj->quadrupole[5] * dist[1] +
                  pmj->quadrupole[2] * dist[2];
      vector_scalar_mul_sum_inplace(dist, 5.0 / r7 * qsum, field_undamped);
      vector_scalar_mul_sum_inplace(tmpvec, -2.0 / r5, field_undamped);

      vector_scalar_mul_sum_inplace(field_undamped, damping, field);
    }
    vector_sum_inplace(pfragA->pol_status[i].field_immut, field);
  }
}

void calc_induced_dipole_field(pol_fragment *pfragA,
                               const pol_fragment *pfragB) {
  double r, r2, r3, r5, psum, damping;
  const pol_point *ppi, *ppj;
  pol_point_status *ppstat;
  vector dist, field, moveA, moveB, moveAB;
  vector field_undamped;
  size_t n, i, j;

  vector_sub(pfragA->masscenter, pfragA->original->masscenter, moveA);
  vector_sub(pfragB->masscenter, pfragB->original->masscenter, moveB);
  vector_sub(moveA, moveB, moveAB);
  for (i = 0; i < pfragA->original->std_ptr->n_pol_points; ++i) {
    vector_zero(field);
    ppi = pfragA->original->pol_ptr + i;
    for (j = 0; j < pfragB->original->std_ptr->n_pol_points; ++j) {
      ppj = pfragB->original->pol_ptr + j;
      ppstat = pfragB->pol_status + j;
      vector_sub(ppi->position, ppj->position, dist);
      vector_sum_inplace(dist, moveAB);
      r = vector_len(dist);
      r2 = r * r;
      r3 = r2 * r;
      r5 = r2 * r3;

      damping = 1.0 - exp(-0.6 * r2) * (1.0 + 0.6 * r2);

      psum = vector_dot(ppstat->dipole, dist);
      vector_zero(field_undamped);
      vector_scalar_mul_sum_inplace(dist, 3.0 / r5 * psum, field_undamped);
      vector_scalar_mul_sum_inplace(ppstat->dipole, -1 / r3, field_undamped);

      vector_scalar_mul_sum_inplace(field_undamped, damping, field);
    }
    vector_sum_inplace(pfragA->pol_status[i].field_induced, field);
  }
}

double calc_induced_dipole(pol_fragment *pfrag, double mix) {
  size_t i;
  vector field, dipole, old_dipole;
  pol_point *ppi;
  pol_point_status *ppstat;
  double convergence = 0.0;

  for (i = 0; i < pfrag->original->std_ptr->n_pol_points; ++i) {
    ppi = pfrag->original->pol_ptr + i;
    ppstat = pfrag->pol_status + i;
    vector_dup(ppstat->dipole, old_dipole);

    vector_sum(ppstat->field_induced, ppstat->field_immut, field);
    matrix_vector_mul(ppi->polarizability, field, dipole);
    vector_scalar_mul_inplace(dipole, 1 - mix);
    vector_scalar_mul_inplace(ppstat->dipole, mix);
    vector_sum_inplace(ppstat->dipole, dipole);

    vector_sub_inplace(old_dipole, ppstat->dipole);
    convergence += vector_dot(old_dipole, old_dipole);
  }
  return convergence;
}

double calc_fragment_polarization_energy(const pol_fragment *pfrag) {
  double energy = 0.0;
  size_t i;
  pol_point_status *ppstat;

  for (i = 0; i < pfrag->original->std_ptr->n_pol_points; ++i) {
    ppstat = pfrag->pol_status + i;
    energy -= 0.5 * vector_dot(ppstat->dipole, ppstat->field_immut);
  }
  return energy;
}

double calc_2body_electrostatic_energy(const pol_fragment *pfragA,
                                       const pol_fragment *pfragB) {
  double energy = 0.0;

  energy += calc_nuclear_nuclear_energy(pfragA, pfragB);
  energy += calc_nuclear_multipole_energy(pfragA, pfragB);
  energy += calc_nuclear_multipole_energy(pfragB, pfragA);
  energy += calc_multipole_multipole_energy(pfragA, pfragB);

  return energy;
}
