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
  if (n != nexclude) {
    fprintf(stderr, "%s[%d]\n", __FILE__, __LINE__);
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

  fprintf(stdout, "Calculate Neutral\n");
  init_pol_mem(pcls);

#pragma omp parallel for schedule(dynamic) reduction(+ : elec_n)
  for (size_t i = 0; i < pcls->n_polfrags; ++i) {
    if (i == pcls->center) {
      continue;
    }
    elec_n += calc_2body_electrostatic_energy(pcls->polfrag_ptr + pcls->center,
                                              pcls->polfrag_ptr + i);
  }

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

  init_field_induced(pcls);
  polarize_scf(pcls);

#pragma omp parallel for schedule(dynamic) reduction(+ : pol_n)
  for (size_t i = 0; i < pcls->n_include; ++i) {
    pol_n += calc_fragment_polarization_energy(pcls->polfrag_ptr +
                                               pcls->include_ptr[i]);
  }

  if (cation != NULL) {
    fprintf(stdout, "Calculate Cation\n");
    modify_center_charge(pcls, 1);
    init_pol_mem(pcls);

#pragma omp parallel for schedule(dynamic) reduction(+ : elec_c)
    for (size_t i = 0; i < pcls->n_polfrags; ++i) {
      if (i == pcls->center) {
        continue;
      }
      elec_c += calc_2body_electrostatic_energy(
          pcls->polfrag_ptr + pcls->center, pcls->polfrag_ptr + i);
    }

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

    polarize_scf(pcls);

#pragma omp parallel for schedule(dynamic) reduction(+ : pol_c)
    for (size_t i = 0; i < pcls->n_include; ++i) {
      pol_c += calc_fragment_polarization_energy(pcls->polfrag_ptr +
                                                 pcls->include_ptr[i]);
    }
  }

  if (anion != NULL) {
    fprintf(stdout, "Calculate Anion\n");
    modify_center_charge(pcls, -1);
    init_pol_mem(pcls);

#pragma omp parallel for schedule(dynamic) reduction(+ : elec_a)
    for (size_t i = 0; i < pcls->n_polfrags; ++i) {
      if (i == pcls->center) {
        continue;
      }
      elec_a += calc_2body_electrostatic_energy(
          pcls->polfrag_ptr + pcls->center, pcls->polfrag_ptr + i);
    }

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

    polarize_scf(pcls);

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
