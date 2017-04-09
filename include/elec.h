#ifndef __CORE_H__
#define __CORE_H__

#include "types.h"

extern void alloc_pol_mem(cluster *pcls);

extern void free_pol_mem(cluster *pcls);

extern void init_pol_mem(cluster *pcls);

extern void init_dipole_induced(cluster *pcls);

extern void init_field_induced(cluster *pcls);

extern void init_field_immut(cluster *pcls);

extern void calc_mult_field(pol_fragment *pfragA, const pol_fragment *pfragB);

extern void calc_induced_dipole_field(pol_fragment *pfragA, const pol_fragment *pfragB);

extern double calc_induced_dipole(pol_fragment *pfrag, double mix);

extern void add_induced_field(pol_fragment *pfrag);

extern void zero_induced_dipole(pol_fragment *pfrag);

extern double calc_fragment_polarization_energy(const pol_fragment *pfrag);

extern double calc_2body_electrostatic_energy(const pol_fragment *pfragA, const pol_fragment *pfragB);

#endif
