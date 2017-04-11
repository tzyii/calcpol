#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "cluster.h"
#include "elec.h"
#include "matvec.h"
#include "types.h"
#include "utils.h"

static size_t get_exclude_idx(const cluster *pcls, size_t **ptr) {
  bool found = false;
  size_t n = 0, nexclude = pcls->n_polfrags - pcls->n_include;

  if (nexclude > 0) {
    *ptr = galloc(sizeof(size_t) * nexclude);

    for (size_t i = 0; i < pcls->n_polfrags; ++i) {
      found = false;
      for (size_t j = 0; j < pcls->n_include; ++j) {
        if (pcls->include_ptr[j] == i) {
          found = true;
          break;
        }
      }
      if (!found) {
        (*ptr)[n++] = i;
      }
    }
  }
  return nexclude;
}

double calc_electrostatic_energy(const cluster *pcls) {
  double energy = 0.0;
  size_t nexclude, *exptr = NULL;

  if (pcls->n_include != 0) {
    nexclude = get_exclude_idx(pcls, &exptr);

#pragma omp parallel for schedule(dynamic) reduction(+ : energy)
    for (size_t i = 0; i < pcls->n_include - 1; ++i) {
      for (size_t j = i + 1; j < pcls->n_include; ++j) {
        energy += calc_2body_electrostatic_energy(
            pcls->polfrag_ptr + pcls->include_ptr[i],
            pcls->polfrag_ptr + pcls->include_ptr[j]);
      }
      for (size_t j = 0; j < nexclude; ++j) {
        energy += calc_2body_electrostatic_energy(pcls->polfrag_ptr +
                                                      pcls->include_ptr[i],
                                                  pcls->polfrag_ptr + exptr[j]);
      }
    }
    free(exptr);
  }
  return energy;
}

static void polarize_scf(cluster *pcls) {
  double convergence, mix;
  size_t npoint = 0;

  for (size_t n = 0; n < pcls->n_include; ++n) {
    npoint +=
        pcls->polfrag_ptr[pcls->include_ptr[n]].original->std_ptr->n_pol_points;
  }

  fprintf(stdout, "Iterative Polarize:\n");

  for (size_t niter = 1; niter <= 200; ++niter) {
    convergence = 0.0;
    mix = (niter > 3) ? 0.50 : 0.90 - niter * 0.1;
    init_field_induced(pcls);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < pcls->n_include; ++i) {
      for (size_t j = 0; j < pcls->n_include; ++j) {
        if (i == j) {
          continue;
        }
        calc_induced_dipole_field(pcls->polfrag_ptr + pcls->include_ptr[i],
                                  pcls->polfrag_ptr + pcls->include_ptr[j]);
      }
    }
#pragma omp parallel for schedule(dynamic) reduction(+ : convergence)
    for (size_t i = 0; i < pcls->n_include; ++i) {
      convergence +=
          calc_induced_dipole(pcls->polfrag_ptr + pcls->include_ptr[i], mix);
    }
    convergence = sqrt(convergence / npoint);
    fprintf(stdout, "Iteration %3zu: RMS Dipole = %12.9f\n", niter,
            convergence);
    if (convergence < 1.0e-07) {
      fprintf(stdout, "Converged!\n");
      break;
    }
  }
}

double calc_polarization_energy(cluster *pcls) {
  double energy = 0.0, convergence, mix;

  init_pol_mem(pcls);

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < pcls->n_include; ++i) {
    for (size_t j = 0; j < pcls->n_polfrags; ++j) {
      if (pcls->include_ptr[i] == j) {
        continue;
      }
      calc_mult_field(pcls->polfrag_ptr + pcls->include_ptr[i],
                      pcls->polfrag_ptr + j);
    }
  }

  polarize_scf(pcls);

#pragma omp parallel for schedule(dynamic) reduction(+ : energy)
  for (size_t i = 0; i < pcls->n_include; ++i) {
    energy += calc_fragment_polarization_energy(pcls->polfrag_ptr +
                                                pcls->include_ptr[i]);
  }

  return energy;
}

void calc_ionization_energy(cluster *pcls, double *cation, double *anion) {
  double elec_n = 0.0, elec_c = 0.0, elec_a = 0.0;
  double pol_n = 0.0, pol_c = 0.0, pol_a = 0.0;
  size_t npoint = 0;
  vector **pfield;

  pfield = galloc(pcls->n_include * sizeof(vector *));
  for (size_t n = 0; n < pcls->n_include; ++n) {
    pfield[n] = galloc(pcls->polfrag_ptr[pcls->include_ptr[n]]
                           .original->std_ptr->n_pol_points *
                       sizeof(vector));
  }

  fprintf(stdout, "Set the central molecule to be neutral and initailize all "
                  "polarization status.\n");
  modify_center_charge(pcls, 0);
  init_pol_mem(pcls);

  fprintf(stdout, "Calculate permanent multipole interaction energy with these "
                  "of central molecule...\n");
#pragma omp parallel for schedule(dynamic) reduction(+ : elec_n)
  for (size_t i = 0; i < pcls->n_polfrags; ++i) {
    if (i == pcls->center) {
      continue;
    }
    elec_n += calc_2body_electrostatic_energy(pcls->polfrag_ptr + pcls->center,
                                              pcls->polfrag_ptr + i);
  }

  fprintf(stdout,
          "Calculate fields upon polarizable molecules exerted by "
          "permanent multipoles of molecules other than themselves...\n");
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < pcls->n_include; ++i) {
    for (size_t j = 0; j < pcls->n_polfrags; ++j) {
      if (pcls->include_ptr[i] == j || j == pcls->center) {
        continue;
      }
      calc_mult_field(pcls->polfrag_ptr + pcls->include_ptr[i],
                      pcls->polfrag_ptr + j);
    }
    for (size_t k = 0; k < pcls->polfrag_ptr[pcls->include_ptr[i]]
                               .original->std_ptr->n_pol_points;
         ++k) {
      vector_dup(
          pcls->polfrag_ptr[pcls->include_ptr[i]].pol_status[k].field_immut,
          pfield[i][k]);
    }
    if (pcls->include_ptr[i] == pcls->center) {
      continue;
    }
    calc_mult_field(pcls->polfrag_ptr + pcls->include_ptr[i],
                    pcls->polfrag_ptr + pcls->center);
  }

  fprintf(stdout, "Calculate induced dipoles...\n");
  polarize_scf(pcls);

  fprintf(stdout, "Calculate polarization energy <pol_n>...\n");
#pragma omp parallel for schedule(dynamic) reduction(+ : pol_n)
  for (size_t i = 0; i < pcls->n_include; ++i) {
    pol_n += calc_fragment_polarization_energy(pcls->polfrag_ptr +
                                               pcls->include_ptr[i]);
  }

  if (cation != NULL) {
    fprintf(stdout,
            "Set the central molecule to be cationic and initailize all "
            "polarization status.\n");
    modify_center_charge(pcls, 1);
    init_pol_mem(pcls);

    fprintf(stdout, "Calculate permanent multipole interaction energy with "
                    "these of central molecule...\n");
#pragma omp parallel for schedule(dynamic) reduction(+ : elec_c)
    for (size_t i = 0; i < pcls->n_polfrags; ++i) {
      if (i == pcls->center) {
        continue;
      }
      elec_c += calc_2body_electrostatic_energy(
          pcls->polfrag_ptr + pcls->center, pcls->polfrag_ptr + i);
    }

    fprintf(stdout, "Renew fields exerted by permanent multipoles of central "
                    "molecule...\n");
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < pcls->n_include; ++i) {
      for (size_t k = 0; k < pcls->polfrag_ptr[pcls->include_ptr[i]]
                                 .original->std_ptr->n_pol_points;
           ++k) {
        vector_dup(
            pfield[i][k],
            pcls->polfrag_ptr[pcls->include_ptr[i]].pol_status[k].field_immut);
      }
      if (pcls->include_ptr[i] == pcls->center) {
        continue;
      }
      calc_mult_field(pcls->polfrag_ptr + pcls->include_ptr[i],
                      pcls->polfrag_ptr + pcls->center);
    }

    fprintf(stdout, "Calculate induced dipoles...\n");
    polarize_scf(pcls);

    fprintf(stdout, "Calculate polarization energy <pol_c>...\n");
#pragma omp parallel for schedule(dynamic) reduction(+ : pol_c)
    for (size_t i = 0; i < pcls->n_include; ++i) {
      pol_c += calc_fragment_polarization_energy(pcls->polfrag_ptr +
                                                 pcls->include_ptr[i]);
    }
  }

  if (anion != NULL) {
    fprintf(stdout, "Set the central molecule to be anionic and initailize all "
                    "polarization status.\n");
    modify_center_charge(pcls, -1);
    init_pol_mem(pcls);

    fprintf(stdout, "Calculate permanent multipole interaction energy with "
                    "these of central molecule...\n");
#pragma omp parallel for schedule(dynamic) reduction(+ : elec_a)
    for (size_t i = 0; i < pcls->n_polfrags; ++i) {
      if (i == pcls->center) {
        continue;
      }
      elec_a += calc_2body_electrostatic_energy(
          pcls->polfrag_ptr + pcls->center, pcls->polfrag_ptr + i);
    }

    fprintf(stdout, "Renew fields exerted by permanent multipoles of central "
                    "molecule...\n");
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < pcls->n_include; ++i) {
      for (size_t k = 0; k < pcls->polfrag_ptr[pcls->include_ptr[i]]
                                 .original->std_ptr->n_pol_points;
           ++k) {
        vector_dup(
            pfield[i][k],
            pcls->polfrag_ptr[pcls->include_ptr[i]].pol_status[k].field_immut);
      }
      if (pcls->include_ptr[i] == pcls->center) {
        continue;
      }
      calc_mult_field(pcls->polfrag_ptr + pcls->include_ptr[i],
                      pcls->polfrag_ptr + pcls->center);
    }

    fprintf(stdout, "Calculate induced dipoles...\n");
    polarize_scf(pcls);

    fprintf(stdout, "Calculate polarization energy <pol_a>...\n");
#pragma omp parallel for schedule(dynamic) reduction(+ : pol_a)
    for (size_t i = 0; i < pcls->n_include; ++i) {
      pol_a += calc_fragment_polarization_energy(pcls->polfrag_ptr +
                                                 pcls->include_ptr[i]);
    }
  }

  for (size_t n = 0; n < pcls->n_include; ++n) {
    free(pfield[n]);
    pfield[n] = NULL;
  }
  free(pfield);
  pfield = NULL;

  if (cation != NULL) {
    *cation = elec_n + pol_n - elec_c - pol_c;
  }

  if (anion != NULL) {
    *anion = elec_n + pol_n - elec_a - pol_a;
  }
}

static void list_polarize_scf(pol_fragment **inc_ptr, size_t ninc) {
  double convergence, mix;
  size_t npoint = 0;

  for (size_t n = 0; n < ninc; ++n) {
    npoint += inc_ptr[n]->original->std_ptr->n_pol_points;
  }

  fprintf(stdout, "Iterative Polarize:\n");

  for (size_t niter = 1; niter <= 200; ++niter) {
    convergence = 0.0;
    mix = (niter > 3) ? 0.50 : 0.90 - niter * 0.1;
    for (size_t m = 0; m < ninc; ++m) {
      for (size_t n = 0; n < inc_ptr[m]->original->std_ptr->n_pol_points; ++n) {
        vector_zero(inc_ptr[m]->pol_status[n].field_induced);
      }
    }
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < ninc; ++i) {
      for (size_t j = 0; j < ninc; ++j) {
        if (i == j) {
          continue;
        }
        calc_induced_dipole_field(inc_ptr[i], inc_ptr[j]);
      }
    }
#pragma omp parallel for schedule(dynamic) reduction(+ : convergence)
    for (size_t i = 0; i < ninc; ++i) {
      convergence += calc_induced_dipole(inc_ptr[i], mix);
    }
    convergence = sqrt(convergence / npoint);
    fprintf(stdout, "Iteration %3zu: RMS Dipole = %12.9f\n", niter,
            convergence);
    if (convergence < 1.0e-07) {
      fprintf(stdout, "Converged!\n");
      break;
    }
  }
    /*for (size_t m = 0; m < ninc; ++m) {
      for (size_t n = 0; n < inc_ptr[m]->original->std_ptr->n_pol_points; ++n) {
        fprintf(stdout, "%9.5f %9.5f %9.5f\n", inc_ptr[m]->pol_status[n].field_induced[0], inc_ptr[m]->pol_status[n].field_induced[1], inc_ptr[m]->pol_status[n].field_induced[2]);
      }
    }*/
}

double calc_reorganization_energy(cluster *pclsA, cluster *pclsB, int charge) {
  double pol_a = 0.0, pol_b = 0.0;
  size_t m, n, npol = 0, ninc = 0, centers[2];
  size_t *inc_idx_ptr;
  bool found = false;
  pol_fragment **pol_ptr, **inc_ptr, *cent_ptr[2], *pfragA, *pfragB;
  vector **pfield;

  pol_ptr =
      galloc(sizeof(pol_fragment *) * (pclsA->n_polfrags + pclsB->n_polfrags));
  inc_ptr =
      galloc(sizeof(pol_fragment *) * (pclsA->n_polfrags + pclsB->n_polfrags));
  inc_idx_ptr =
      galloc(sizeof(size_t) * (pclsA->n_polfrags + pclsB->n_polfrags));

  fprintf(stdout, "Merge the two clusters...\n");
  centers[0] = pclsA->center;
  for (size_t A_pol_idx = 0; A_pol_idx < pclsA->n_polfrags; ++A_pol_idx) {
    pol_ptr[npol++] = pclsA->polfrag_ptr + A_pol_idx;
  }
  for (size_t A_inc_idx = 0; A_inc_idx < pclsA->n_include; ++A_inc_idx) {
    inc_idx_ptr[ninc++] = pclsA->include_ptr[A_inc_idx];
  }
  for (size_t B_pol_idx = 0; B_pol_idx < pclsB->n_polfrags; ++B_pol_idx) {
    pfragB = pclsB->polfrag_ptr + B_pol_idx;
    found = false;
    for (size_t A_pol_idx = 0; A_pol_idx < pclsA->n_polfrags; ++A_pol_idx) {
      pfragA = pclsA->polfrag_ptr + A_pol_idx;
      if (vector_equal(pfragA->masscenter, pfragB->masscenter)) {
        found = true;
        if (B_pol_idx == pclsB->center) {
          centers[1] = A_pol_idx;
        }
        for (size_t B_inc_idx = 0; B_inc_idx < pclsB->n_include; ++B_inc_idx) {
          if (B_pol_idx == pclsB->include_ptr[B_inc_idx]) {
            bool ininc = false;
            for (size_t A_inc_idx = 0; A_inc_idx < pclsA->n_include;
                 ++A_inc_idx) {
              if (A_pol_idx == pclsA->include_ptr[A_inc_idx]) {
                ininc = true;
                break;
              }
            }
            if (!ininc) {
              inc_idx_ptr[ninc++] = A_pol_idx;
            }
          }
        }
        break;
      }
    }
    if (!found) {
      if (B_pol_idx == pclsB->center) {
        centers[1] = npol;
      }
      for (size_t B_inc_idx = 0; B_inc_idx < pclsB->n_include; ++B_inc_idx) {
        if (B_pol_idx == pclsB->include_ptr[B_inc_idx]) {
          inc_idx_ptr[ninc++] = npol;
          break;
        }
      }
      pol_ptr[npol++] = pfragB;
    }
  }
  pol_ptr[centers[1]] = pclsB->polfrag_ptr + pclsB->center;
  cent_ptr[0] = pol_ptr[centers[0]];
  cent_ptr[1] = pol_ptr[centers[1]];
  fprintf(stdout,
          "There are %lu(%lu polarizable) molecules in merged cluster.\n", npol,
          ninc);

  for (size_t inc_idx = 0; inc_idx < ninc; ++inc_idx) {
    inc_ptr[inc_idx] = pol_ptr[inc_idx_ptr[inc_idx]];
  }

  pfield = galloc(ninc * sizeof(vector *));
  for (size_t inc_idx = 0; inc_idx < ninc; ++inc_idx) {
    pfield[inc_idx] = galloc(inc_ptr[inc_idx]->original->std_ptr->n_pol_points *
                             sizeof(vector));
  }

  fprintf(stdout, "Set the two centers to be neutral and initailize all "
                  "polarization status.\n");
  modify_center_charge(pclsA, 0);
  modify_center_charge(pclsB, 0);
  init_pol_mem(pclsA);
  init_pol_mem(pclsB);

  fprintf(stdout, "Calculate fields upon polarizable molecules exerted by "
                  "permanent multipoles of molecules other than themselves and "
                  "centers...\n");
#pragma omp parallel for schedule(dynamic)
  for (size_t inc_idx = 0; inc_idx < ninc; ++inc_idx) {
    for (size_t pol_idx = 0; pol_idx < npol; ++pol_idx) {
      if (inc_idx_ptr[inc_idx] == pol_idx || pol_idx == centers[0] ||
          pol_idx == centers[1]) {
        continue;
      }
      calc_mult_field(inc_ptr[inc_idx], pol_ptr[pol_idx]);
    }
    for (size_t idx = 0;
         idx < inc_ptr[inc_idx]->original->std_ptr->n_pol_points; ++idx) {
      vector_dup(inc_ptr[inc_idx]->pol_status[idx].field_immut,
                 pfield[inc_idx][idx]);
    }
  }

  fprintf(stdout,
          "Set the central molecule of cluster A to be charged(%+d) and "
          "initailize all polarization status.\n",
          charge);
  modify_center_charge(pclsA, charge);
  init_pol_mem(pclsA);
  init_pol_mem(pclsB);

  fprintf(
      stdout,
      "Add fields exerted by permannent multipoles of central molecules...\n");
#pragma omp parallel for schedule(dynamic)
  for (size_t inc_idx = 0; inc_idx < ninc; ++inc_idx) {
    for (size_t idx = 0;
         idx < inc_ptr[inc_idx]->original->std_ptr->n_pol_points; ++idx) {
      vector_dup(pfield[inc_idx][idx],
                 inc_ptr[inc_idx]->pol_status[idx].field_immut);
    }
    if (inc_idx_ptr[inc_idx] == centers[0]) {
      calc_mult_field(inc_ptr[inc_idx], pol_ptr[centers[1]]);
    } else if (inc_idx_ptr[inc_idx] == centers[1]) {
      calc_mult_field(inc_ptr[inc_idx], pol_ptr[centers[0]]);
    } else {
      calc_mult_field(inc_ptr[inc_idx], pol_ptr[centers[0]]);
      calc_mult_field(inc_ptr[inc_idx], pol_ptr[centers[1]]);
    }
  }

  fprintf(stdout, "Calculate induced dipoles...\n");
  list_polarize_scf(inc_ptr, ninc);

  fprintf(stdout, "Set the central molecules of cluster A and B to be neutral "
                  "and charged(%+d) respectively.\n",
          charge);
  modify_center_charge(pclsA, 0);
  modify_center_charge(pclsB, charge);
  init_field_induced(pclsA);
  init_field_induced(pclsB);

  fprintf(
      stdout,
      "Renew field exerted by permannent multipoles of central molecules...\n");
  fprintf(stdout, "(Treat induced dipole of surrounding molecules as "
                  "permannent for central molecules)\n");
#pragma omp parallel for schedule(dynamic) reduction(+ : pol_a)
  for (size_t inc_idx = 0; inc_idx < ninc; ++inc_idx) {
    for (size_t idx = 0;
         idx < inc_ptr[inc_idx]->original->std_ptr->n_pol_points; ++idx) {
      vector_dup(pfield[inc_idx][idx],
                 inc_ptr[inc_idx]->pol_status[idx].field_immut);
    }
    if (inc_idx_ptr[inc_idx] != centers[0] &&
        inc_idx_ptr[inc_idx] != centers[1]) {
      calc_mult_field(inc_ptr[inc_idx], pol_ptr[centers[0]]);
      calc_mult_field(inc_ptr[inc_idx], pol_ptr[centers[1]]);
    } else {
      if (inc_idx_ptr[inc_idx] == centers[0]) {
        calc_mult_field(inc_ptr[inc_idx], pol_ptr[centers[1]]);
      } else {
        calc_mult_field(inc_ptr[inc_idx], pol_ptr[centers[0]]);
      }
      for (size_t idx = 0; idx < ninc; ++idx) {
        if (inc_idx_ptr[idx] == centers[0] || inc_idx_ptr[idx] == centers[1]) {
          continue;
        } else {
          calc_induced_dipole_field(inc_ptr[inc_idx], inc_ptr[idx]);
        }
      }
      add_induced_field(inc_ptr[inc_idx]);
    }
  }

  fprintf(stdout, "Initialize induced dipoles of central molecules.\n");
  //zero_induced_dipole(pol_ptr[centers[0]]);
  //zero_induced_dipole(pol_ptr[centers[1]]);

  fprintf(stdout, "Calculate induced dipoles of central molecules...\n");
  //list_polarize_scf(cent_ptr, 2);

#pragma omp parallel for schedule(dynamic)
  for (size_t inc_idx = 0; inc_idx < ninc; ++inc_idx) {
    zero_induced_field(inc_ptr[inc_idx]);
    if (inc_idx_ptr[inc_idx] == centers[0] ||
        inc_idx_ptr[inc_idx] == centers[1]) {
      for (size_t idx = 0;
           idx < inc_ptr[inc_idx]->original->std_ptr->n_pol_points; ++idx) {
        vector_dup(pfield[inc_idx][idx],
                   inc_ptr[inc_idx]->pol_status[idx].field_immut);
      }
      if (inc_idx_ptr[inc_idx] == centers[0]) {
        calc_mult_field(inc_ptr[inc_idx], pol_ptr[centers[1]]);
      } else {
        calc_mult_field(inc_ptr[inc_idx], pol_ptr[centers[0]]);
      }
    }
    for (size_t idx = 0; idx < ninc; ++idx) {
      if (inc_idx == idx) {
        continue;
      }
      calc_induced_dipole_field(inc_ptr[inc_idx], inc_ptr[idx]);
    }
    pol_a += calc_fragment_polarization_energy(inc_ptr[inc_idx]);
  }

  fprintf(stdout, "Calculate polarization energy <pol_s2_s1>...\n");

  pol_a += calc_fragment_polarization_energy(pol_ptr[centers[0]]) +
           calc_fragment_polarization_energy(pol_ptr[centers[1]]);

  init_dipole_induced(pclsA);
  init_dipole_induced(pclsB);

  fprintf(stdout, "Calculate induced dipoles...\n");
  list_polarize_scf(inc_ptr, ninc);

  fprintf(stdout, "Calculate polarization energy <pol_s2_s2>...\n");
#pragma omp parallel for schedule(dynamic) reduction(+ : pol_b)
  for (size_t inc_idx = 0; inc_idx < ninc; ++inc_idx) {
    pol_b += calc_fragment_polarization_energy(inc_ptr[inc_idx]);
  }

fprintf(stdout, "pol: %12.9f %12.9f\n", pol_a, pol_b);
  modify_center_charge(pclsA, 0);
  modify_center_charge(pclsB, 0);
  init_pol_mem(pclsA);
  init_pol_mem(pclsB);

  free(pol_ptr);
  free(inc_ptr);
  free(inc_idx_ptr);
  pol_ptr = NULL;
  inc_ptr = NULL;
  inc_idx_ptr = NULL;

  return pol_a - pol_b;
}
