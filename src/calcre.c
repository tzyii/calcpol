#include <omp.h>

#include "calc.h"
#include "cluster.h"
#include "elec.h"
#include "utils.h"

static void useage(void) {
  fprintf(stderr, "clacre [input] [charge] [centerA] [centerB] [full_radius] "
                  "[pol_radius] [nthread]\n");
}

int main(int argc, char *argv[]) {
  box sys;
  cluster clsA, clsB;
  size_t i, j, center_index_A, center_index_B;
  double r_full, r_pol, e_re;
  int nthreads, charge;
  char filename[1024];

  if (argc == 8 && sscanf(argv[1], "%s", filename) &&
      sscanf(argv[2], "%d", &charge) &&
      sscanf(argv[3], "%zd", &center_index_A) &&
      sscanf(argv[4], "%zd", &center_index_B) &&
      sscanf(argv[5], "%lf", &r_full) && sscanf(argv[6], "%lf", &r_pol) &&
      sscanf(argv[7], "%d", &nthreads)) {
    fprintf(stdout, "Input File: %s\n", filename);
    fprintf(stdout, "Charge: %d\n", charge);
    fprintf(stdout, "Central Molecular Index: %lu %lu\n", center_index_A,
            center_index_B);
    fprintf(stdout, "Cluster Radius: %.5lf Angstrom\n", r_full);
    fprintf(stdout, "Polarizable Radius: %.5lf Angstrom\n", r_pol);
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
  gen_cluster(&clsA, &sys, r_full * BOHR, center_index_A);
  fprintf(stdout, "Generate cluster A with radius equal "
                  "to %.5lf angstrom.\n",
          r_full);
  set_polarizable_include(&clsA, r_pol * BOHR);
  fprintf(stdout, "There are %lu(%lu polarizable) molecules in cluster A.\n",
          clsA.n_polfrags, clsA.n_include);
  gen_cluster(&clsB, &sys, r_full * BOHR, center_index_B);
  fprintf(stdout, "Generate cluster B with radius equal "
                  "to %.5lf angstrom.\n",
          r_full);
  set_polarizable_include(&clsB, r_pol * BOHR);
  fprintf(stdout, "There are %lu(%lu polarizable) molecules in cluster B.\n",
          clsA.n_polfrags, clsA.n_include);
  fprintf(stdout, "\n******************************\n\n");
  alloc_pol_mem(&clsA);
  alloc_pol_mem(&clsB);
  e_re = calc_reorganization_energy(&clsA, &clsB, charge);
  fprintf(stdout, "\n******************************\n\n");
  fprintf(stdout, "Reorganization Energy: %15.9f Ha\n", e_re);
  return 0;
}
