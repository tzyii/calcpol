#ifndef __CALC_H__
#define __CALC_H__

#include "types.h"

extern double calc_electrostatic_energy(const cluster *pcls);

extern double calc_induction_energy(const cluster *pcls);

extern void calc_polarization_energy(cluster *pcls, double *cation,
                                   double *anion);

#endif
