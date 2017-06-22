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

double calc_induction_energy(cluster *pcls) {
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
    energy += calc_fragment_induction_energy(pcls->polfrag_ptr +
                                                pcls->include_ptr[i]);
  }

  return energy;
}

void calc_polarization_energy(cluster *pcls, double *cation, double *anion) {
  double nn_n = 0.0, nm_n = 0.0, nd_n = 0.0, nq_n = 0.0, no_n = 0.0;
  double mm_n = 0.0, md_n = 0.0, mq_n = 0.0, mo_n = 0.0, dd_n = 0.0;
  double dq_n = 0.0, qq_n = 0.0;
  double nn_c = 0.0, nm_c = 0.0, nd_c = 0.0, nq_c = 0.0, no_c = 0.0;
  double mm_c = 0.0, md_c = 0.0, mq_c = 0.0, mo_c = 0.0, dd_c = 0.0;
  double dq_c = 0.0, qq_c = 0.0;
  double nn_a = 0.0, nm_a = 0.0, nd_a = 0.0, nq_a = 0.0, no_a = 0.0;
  double mm_a = 0.0, md_a = 0.0, mq_a = 0.0, mo_a = 0.0, dd_a = 0.0;
  double dq_a = 0.0, qq_a = 0.0;
  detailed_energy denergy;
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

  fprintf(stdout, "Decomposition permanent multipole interaction energy"
                  "of the whole cluster...\n");
#pragma omp parallel for schedule(dynamic) private(denergy)                    \
    reduction(+ : elec_n, nn_n, nm_n, nd_n, nq_n, no_n, mm_n, md_n, mq_n,      \
              mo_n, dd_n, dq_n, qq_n)
  for (size_t i = 1; i < pcls->n_polfrags; ++i) {
    for (size_t j = 0; j < i; ++j) {
      elec_n += calc_2body_electrostatic_energy_detail(
          pcls->polfrag_ptr + i, pcls->polfrag_ptr + j, &denergy);
      nn_n += denergy.nn;
      nm_n += denergy.nm;
      nd_n += denergy.nd;
      nq_n += denergy.nq;
      no_n += denergy.no;
      mm_n += denergy.mm;
      md_n += denergy.md;
      mq_n += denergy.mq;
      mo_n += denergy.mo;
      dd_n += denergy.dd;
      dq_n += denergy.dq;
      qq_n += denergy.qq;
    }
  }
  fprintf(stdout, "Total electrostatic energy(whole cluster): % 25.12f Ha\n"
                  "  nuclear    -    nuclear: % 25.12f Ha\n"
                  "  nuclear    -   momopole: % 25.12f Ha\n"
                  "  nuclear    -     dipole: % 25.12f Ha\n"
                  "  nuclear    - quadrupole: % 25.12f Ha\n"
                  "  nuclear    -   octopole: % 25.12f Ha\n"
                  "  monopole   -   monopole: % 25.12f Ha\n"
                  "  monopole   -     dipole: % 25.12f Ha\n"
                  "  monopole   - quadrupole: % 25.12f Ha\n"
                  "  monopole   -   octopole: % 25.12f Ha\n"
                  "  dipole     -     dipole: % 25.12f Ha\n"
                  "  dipole     - quadrupole: % 25.12f Ha\n"
                  "  quadrupole - quadrupole: % 25.12f Ha\n",
          elec_n, nn_n, nm_n, nd_n, nq_n, no_n, mm_n, md_n, mq_n, mo_n, dd_n,
          dq_n, qq_n);

  elec_n = nn_n = nm_n = nd_n = nq_n = no_n = mm_n = md_n = mq_n = mo_n = dd_n =
      dq_n = qq_n = 0.0;

  fprintf(stdout, "Calculate permanent multipole interaction energy with these "
                  "of central molecule...\n");
#pragma omp parallel for schedule(dynamic) private(denergy)                    \
    reduction(+ : elec_n, nn_n, nm_n, nd_n, nq_n, no_n, mm_n, md_n, mq_n,      \
              mo_n, dd_n, dq_n, qq_n)
  for (size_t i = 0; i < pcls->n_polfrags; ++i) {
    if (i == pcls->center) {
      continue;
    }
    elec_n += calc_2body_electrostatic_energy_detail(
        pcls->polfrag_ptr + pcls->center, pcls->polfrag_ptr + i, &denergy);
    nn_n += denergy.nn;
    nm_n += denergy.nm;
    nd_n += denergy.nd;
    nq_n += denergy.nq;
    no_n += denergy.no;
    mm_n += denergy.mm;
    md_n += denergy.md;
    mq_n += denergy.mq;
    mo_n += denergy.mo;
    dd_n += denergy.dd;
    dq_n += denergy.dq;
    qq_n += denergy.qq;
  }
  fprintf(stdout,
          "Total electrostatic energy(central molecule with others): % 25.12f Ha\n"
          "  nuclear    -    nuclear: % 25.12f Ha\n"
          "  nuclear    -   momopole: % 25.12f Ha\n"
          "  nuclear    -     dipole: % 25.12f Ha\n"
          "  nuclear    - quadrupole: % 25.12f Ha\n"
          "  nuclear    -   octopole: % 25.12f Ha\n"
          "  monopole   -   monopole: % 25.12f Ha\n"
          "  monopole   -     dipole: % 25.12f Ha\n"
          "  monopole   - quadrupole: % 25.12f Ha\n"
          "  monopole   -   octopole: % 25.12f Ha\n"
          "  dipole     -     dipole: % 25.12f Ha\n"
          "  dipole     - quadrupole: % 25.12f Ha\n"
          "  quadrupole - quadrupole: % 25.12f Ha\n",
          elec_n, nn_n, nm_n, nd_n, nq_n, no_n, mm_n, md_n, mq_n, mo_n, dd_n,
          dq_n, qq_n);

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

  fprintf(stdout, "Calculate induction energy <pol_n>...\n");
#pragma omp parallel for schedule(dynamic) reduction(+ : pol_n)
  for (size_t i = 0; i < pcls->n_include; ++i) {
    pol_n += calc_fragment_induction_energy(pcls->polfrag_ptr +
                                               pcls->include_ptr[i]);
  }
  fprintf(stdout, "Total induction energy(cluster): % 25.12f Ha\n", pol_n);

  if (cation != NULL) {
    fprintf(stdout,
            "Set the central molecule to be cationic and initailize all "
            "polarization status.\n");
    modify_center_charge(pcls, 1);
    init_pol_mem(pcls);

    fprintf(stdout, "Calculate permanent multipole interaction energy with "
                    "these of central molecule...\n");
#pragma omp parallel for schedule(dynamic) private(denergy)                    \
    reduction(+ : elec_c, nn_c, nm_c, nd_c, nq_c, no_c, mm_c, md_c, mq_c,      \
              mo_c, dd_c, dq_c, qq_c)
    for (size_t i = 0; i < pcls->n_polfrags; ++i) {
      if (i == pcls->center) {
        continue;
      }
      elec_c += calc_2body_electrostatic_energy_detail(
          pcls->polfrag_ptr + pcls->center, pcls->polfrag_ptr + i, &denergy);
      nn_c += denergy.nn;
      nm_c += denergy.nm;
      nd_c += denergy.nd;
      nq_c += denergy.nq;
      no_c += denergy.no;
      mm_c += denergy.mm;
      md_c += denergy.md;
      mq_c += denergy.mq;
      mo_c += denergy.mo;
      dd_c += denergy.dd;
      dq_c += denergy.dq;
      qq_c += denergy.qq;
    }
    fprintf(stdout,
            "Total electrostatic energy(central molecule with others): % 25.12f Ha\n"
            "  nuclear    -    nuclear: % 25.12f Ha\n"
            "  nuclear    -   momopole: % 25.12f Ha\n"
            "  nuclear    -     dipole: % 25.12f Ha\n"
            "  nuclear    - quadrupole: % 25.12f Ha\n"
            "  nuclear    -   octopole: % 25.12f Ha\n"
            "  monopole   -   monopole: % 25.12f Ha\n"
            "  monopole   -     dipole: % 25.12f Ha\n"
            "  monopole   - quadrupole: % 25.12f Ha\n"
            "  monopole   -   octopole: % 25.12f Ha\n"
            "  dipole     -     dipole: % 25.12f Ha\n"
            "  dipole     - quadrupole: % 25.12f Ha\n"
            "  quadrupole - quadrupole: % 25.12f Ha\n",
            elec_c, nn_c, nm_c, nd_c, nq_c, no_c, mm_c, md_c, mq_c, mo_c, dd_c,
            dq_c, qq_c);

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

    fprintf(stdout, "Calculate induction energy <pol_c>...\n");
#pragma omp parallel for schedule(dynamic) reduction(+ : pol_c)
    for (size_t i = 0; i < pcls->n_include; ++i) {
      pol_c += calc_fragment_induction_energy(pcls->polfrag_ptr +
                                                 pcls->include_ptr[i]);
    }
    fprintf(stdout, "Total induction energy(cluster): % 25.12f Ha\n", pol_c);
    fprintf(stdout, "Polarization energy of cation: % 25.12f Ha\n"
                    "  Electrostatic part change: % 25.12f Ha\n"
                    "  induction part change: % 25.12f Ha\n",
            elec_n + pol_n - elec_c - pol_c, elec_n - elec_c, pol_n - pol_c);
  }

  if (anion != NULL) {
    fprintf(stdout, "Set the central molecule to be anionic and initailize all "
                    "polarization status.\n");
    modify_center_charge(pcls, -1);
    init_pol_mem(pcls);

    fprintf(stdout, "Calculate permanent multipole interaction energy with "
                    "these of central molecule...\n");
#pragma omp parallel for schedule(dynamic) private(denergy)                    \
    reduction(+ : elec_a, nn_a, nm_a, nd_a, nq_a, no_a, mm_a, md_a, mq_a,      \
              mo_a, dd_a, dq_a, qq_a)
    for (size_t i = 0; i < pcls->n_polfrags; ++i) {
      if (i == pcls->center) {
        continue;
      }
      elec_a += calc_2body_electrostatic_energy_detail(
          pcls->polfrag_ptr + pcls->center, pcls->polfrag_ptr + i, &denergy);
      nn_a += denergy.nn;
      nm_a += denergy.nm;
      nd_a += denergy.nd;
      nq_a += denergy.nq;
      no_a += denergy.no;
      mm_a += denergy.mm;
      md_a += denergy.md;
      mq_a += denergy.mq;
      mo_a += denergy.mo;
      dd_a += denergy.dd;
      dq_a += denergy.dq;
      qq_a += denergy.qq;
    }
    fprintf(stdout,
            "Total electrostatic energy(central molecule with others): % 25.12f Ha\n"
            "  nuclear    -    nuclear: % 25.12f Ha\n"
            "  nuclear    -   momopole: % 25.12f Ha\n"
            "  nuclear    -     dipole: % 25.12f Ha\n"
            "  nuclear    - quadrupole: % 25.12f Ha\n"
            "  nuclear    -   octopole: % 25.12f Ha\n"
            "  monopole   -   monopole: % 25.12f Ha\n"
            "  monopole   -     dipole: % 25.12f Ha\n"
            "  monopole   - quadrupole: % 25.12f Ha\n"
            "  monopole   -   octopole: % 25.12f Ha\n"
            "  dipole     -     dipole: % 25.12f Ha\n"
            "  dipole     - quadrupole: % 25.12f Ha\n"
            "  quadrupole - quadrupole: % 25.12f Ha\n",
            elec_a, nn_a, nm_a, nd_a, nq_a, no_a, mm_a, md_a, mq_a, mo_a, dd_a,
            dq_a, qq_a);

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

    fprintf(stdout, "Calculate induction energy <pol_a>...\n");
#pragma omp parallel for schedule(dynamic) reduction(+ : pol_a)
    for (size_t i = 0; i < pcls->n_include; ++i) {
      pol_a += calc_fragment_induction_energy(pcls->polfrag_ptr +
                                                 pcls->include_ptr[i]);
    }
    fprintf(stdout, "Total induction energy(cluster): % 25.12f Ha\n", pol_a);
    fprintf(stdout, "Polarization energy of anion: % 25.12f Ha\n"
                    "  Electrostatic part change: % 25.12f Ha\n"
                    "  induction part change: % 25.12f Ha\n",
            elec_n + pol_n - elec_a - pol_a, elec_n - elec_a, pol_n - pol_a);
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
