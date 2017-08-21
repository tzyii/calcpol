#ifndef __UTILS_H__
#define __UTILS_H__

#include "types.h"
#include <stdlib.h>
#include <string.h>

static inline void *galloc(size_t size) {
  void *ptr = malloc(size);
  memset(ptr, 0, size);
  return ptr;
}

#define CAST_PTR(ptr, type) ((type *)(ptr))

typedef struct _list_elem {
  void *pelem;
  struct _list_elem *next;
} * list_ptr;

typedef int(list_lookup_function)(void *, const void *);
typedef void(list_foreach_function)(void *);

void list_append(list_ptr *plptr, void *pelem);

void *list_index(list_ptr lptr, size_t idx);

size_t list_length(list_ptr lptr);

void list_dump(list_ptr lptr, size_t size, void **dest_ptr, size_t *order);

void *list_lookup(list_ptr lptr, list_lookup_function lookup,
                  const void *target);

void list_foreach(list_ptr lptr, list_foreach_function foreach);

void list_clear(list_ptr *pptr);

#define LIST_ELEMENT_SIZE sizeof(struct _list_elem)

#define BOHR 1.889726124565

#define DEG2RAD (3.141592653590 / 180.0)

static const double atom_mass[] = {
    0.00000000,  1.00794000,  4.00260200,  6.94100000,  9.01218200,
    10.81100000, 12.01070000, 14.00670000, 15.99940000, 18.99840320,
    20.17970000, 22.98976928, 24.30500000, 26.98153860, 28.08550000,
    30.97376200, 32.06500000, 35.45300000, 39.94800000, 39.09830000,
    40.07800000, 44.95591200, 47.86700000, 50.94150000, 51.99610000,
    54.93804500, 55.84500000, 58.93319500, 58.69340000, 63.54600000,
    65.38000000, 69.72300000, 72.64000000, 74.92160000, 78.96000000,
    79.90400000, 83.79800000};

#define GET_ATOM_MASS(nuclear) (atom_mass[(int)(nuclear)])

#endif
