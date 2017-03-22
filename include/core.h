#ifndef __CORE_H__
#define __CORE_H__

#include "types.h"

extern void alloc_pol_mem(cluster *pcls);

extern void init_pol_mem(cluster *pcls);

extern double calc_electrostatic_energy(const cluster *pcls, double radius);

extern double calc_polarization_energy(const cluster *pcls, double radius);

#endif
