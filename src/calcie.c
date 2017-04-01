#include <omp.h>

#include "calc.h"
#include "cluster.h"
#include "elec.h"
#include "utils.h"

static void useage(void) {
  fprintf(stderr, "clacie [input] [charge] [center] [full_radius] [pol_radius] "
                  "[nthread]\n");
}

int main(int argc, char *argv[]) {
  box sys;
  cluster cls;
  size_t i, j, center_index;
  double r_full, r_pol, cation = 0.0, anion = 0.0;
  double *pcation = NULL, *panion = NULL;
  int nthreads, charge;
  char filename[1024];

  if (argc == 7 && sscanf(argv[1], "%s", filename) &&
      sscanf(argv[2], "%d", &charge) && sscanf(argv[3], "%zd", &center_index) &&
      sscanf(argv[4], "%lf", &r_full) && sscanf(argv[5], "%lf", &r_pol) &&
      sscanf(argv[6], "%d", &nthreads)) {
    fprintf(stdout, "Input File: %s\n", filename);
    fprintf(stdout, "Charge: %d\n", charge);
    fprintf(stdout, "Central Molecular Index: %lu\n", center_index);
    fprintf(stdout, "Cluster Radius: %.5lf Angstrom\n", r_full);
    fprintf(stdout, "Polarizable Radius: %.5lf Angstrom\n", r_pol);
    fprintf(stdout, "Number of Threads: %d\n", nthreads);
  } else {
    useage();
    exit(-1);
  }

  if (charge == 0) {
    pcation = &cation;
    panion = &anion;
  } else if (charge == -1) {
    panion = &anion;
  } else if (charge == 1) {
    pcation = &cation;
  } else {
    fprintf(stderr, "Charge should be 0, -1 or 1.\n");
  }

  setvbuf(stdout, NULL, _IONBF, 0);

  omp_set_num_threads(nthreads);

  fprintf(stdout, "\n******************************\n\n");
  fragment_load(fopen(filename, "r"), &sys);
  fprintf(stdout, "There are %lu molecules in this box.\n", sys.n_frag);
  gen_cluster(&cls, &sys, r_full * BOHR, center_index);
  fprintf(stdout, "Generate a cluster with radius equal "
                  "to %.5lf angstrom.\n",
          r_full);
  set_polarizable_include(&cls, r_pol * BOHR);
  fprintf(stdout, "There are %lu(%lu polarizable) molecules in this cluster.\n",
          cls.n_polfrags, cls.n_include);
  fprintf(stdout, "\n******************************\n\n");
  alloc_pol_mem(&cls);
  calc_ionization_energy(&cls, &cation, &anion);
  fprintf(stdout, "\n******************************\n\n");
  fprintf(stdout, "Cation Ionization Energy: %15.9f Ha\n", cation);
  fprintf(stdout, "Anion  Ionization Energy: %15.9f Ha\n", anion);
  return 0;
}
