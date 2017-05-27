#ifndef __TYPES_H__
#define __TYPES_H__

#include <stddef.h>

typedef double vector[3];
typedef double matrix[9];
typedef matrix tensor_rank2;
typedef double tensor_rank3[27];

typedef double monopole;
typedef double dipole[3];
typedef double quadrupole[6];
typedef double octupole[10];

typedef struct {
  vector position;
  monopole nuclear;
  monopole monopole;
  dipole dipole;
  quadrupole quadrupole;
  octupole octupole;
  double screen;
} mult_point;

typedef struct {
  vector position;
  matrix polarizability;
} pol_point;

typedef struct { size_t index[2]; } bond;

typedef struct {
  char name[80];
  size_t n_atoms;
  size_t n_bonds;
  size_t n_mult_points;
  size_t n_pol_points;
  size_t principal[4];
  matrix axis;
  vector masscenter;
  mult_point *mult_ptr;
  pol_point *pol_ptr;
  bond *bond_ptr;
} std_efp_info;

typedef struct {
  const std_efp_info *std_ptr;
  matrix rotate;
  vector translate;
  vector masscenter;
  mult_point *mult_ptr;
  pol_point *pol_ptr;
} fragment;

typedef struct {
  dipole dipole;
  vector field_induced;
  vector field_immut;
} pol_point_status;

typedef struct {
  fragment *original;
  vector masscenter;
  pol_point_status *pol_status;
} pol_fragment;

typedef struct {
  matrix lattice;
  size_t n_frag;
  fragment *frag_ptr;
} box;

typedef struct {
  size_t center;
  size_t n_polfrags;
  size_t n_include;
  size_t *include_ptr;
  pol_fragment *polfrag_ptr;
} cluster;

typedef struct {
  double nn;
  double nm;
  double nd;
  double nq;
  double no;
  double mm;
  double md;
  double mq;
  double mo;
  double dd;
  double dq;
  double qq;
} detailed_energy;

#endif
