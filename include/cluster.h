#ifndef __FRAGMENTS_H__
#define __FRAGMENTS_H__

#include "stdio.h"
#include "types.h"

extern void fragment_load(FILE *fp, box *pbox);

extern void gen_cluster(cluster *cls, const box *pbox, double radius,
                        size_t idx_center);

extern void set_polarizable_include(cluster *cls, double radius);

extern void modify_center_charge(cluster *cls, int charge);

#endif
