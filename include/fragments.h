#ifndef __FRAGMENTS_H__
#define __FRAGMENTS_H__

#include "stdio.h"
#include "types.h"

extern void fragment_load(FILE *fp, box *pbox);

extern void set_box_origin(box *pbox, size_t frag_index);

extern void gen_cluster(cluster *cls, const box *pbox, double radius);

extern void modify_center_charge(cluster *cls, int charge);

#endif
