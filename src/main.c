#include <omp.h>

#include "core.h"
#include "fragments.h"
#include "utils.h"

static void useage(void) {
  fprintf(stderr, "clacpol [input] [center] [full_radius] [pol_radius][charge] "
                  "[nthread]\n");
}

int main(int argc, char *argv[]) {
  box sys;
  cluster cls;
  size_t i, j, center_index;
  double r_full, r_pol, e_elec, e_pol, total = 0.0;
  int charge, nthreads;
  char filename[1024];

  if (argc == 7 && sscanf(argv[1], "%s", filename) &&
      sscanf(argv[2], "%zd", &center_index) &&
      sscanf(argv[3], "%lf", &r_full) && sscanf(argv[4], "%lf", &r_pol) &&
      sscanf(argv[5], "%d", &charge) && sscanf(argv[6], "%d", &nthreads)) {
    fprintf(stdout, "Input File: %s\n", filename);
    fprintf(stdout, "Central Molecular Index: %lu\n", center_index);
    fprintf(stdout, "Cluster Radius: %.5lf Angstrom\n", r_full);
    fprintf(stdout, "Polarizable Radius: %.5lf Angstrom\n", r_pol);
    fprintf(stdout, "Central Molecular Charge: %d\n", charge);
    fprintf(stdout, "Number of Threads: %d\n", nthreads);
  } else {
    useage();
    exit(-1);
  }

  setvbuf(stdout, NULL, _IONBF, 0);

  omp_set_num_threads(nthreads);

  fprintf(stdout, "\n******************************\n\n");
  fragment_load(fopen(filename, "r"), &sys);
  fprintf(stdout, "There are %lu molecules in this box.\n", sys.n_frag);
  set_box_origin(&sys, center_index);
  fprintf(
      stdout,
      "Move the origin of coordinate to the centroid of the %ldth molecule.\n",
      center_index);
  gen_cluster(&cls, &sys, r_full * BOHR);
  fprintf(stdout, "Generate a cluster with center at origin and radius equal "
                  "to %.5lf angstrom.\n",
          r_full);
  fprintf(stdout, "There are %lu molecules in this cluster.\n", cls.n_polfrags);
  modify_center_charge(&cls, charge);
  fprintf(stdout, "Set the charge of the central molecule to %+d.\n", charge);

  fprintf(stdout, "\n******************************\n\n");
  e_elec = calc_electrostatic_energy(&cls, r_pol * BOHR);
  alloc_pol_mem(&cls);
  fprintf(stdout, "Iterative Polarizaion:\n");
  e_pol = calc_polarization_energy(&cls, r_pol * BOHR);
  total = e_elec + e_pol;
  fprintf(stdout, "Electrostatic Energy: %15.9f Ha\n", e_elec);
  fprintf(stdout, "Polarization  Energy: %15.9f Ha\n", e_pol);
  fprintf(stdout, "Total         Energy: %15.9f Ha\n", total);
  return 0;
}
