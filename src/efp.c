#include "efp.h"
#include "matvec.h"
#include "types.h"
#include "utils.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static list_ptr pefp = NULL;
static list_ptr pmult = NULL;
static list_ptr ppol = NULL;
static list_ptr pbond = NULL;

static int lookup_efp(void *pelem, const void *key);
static void parse_mult_basic(FILE *fp, std_efp_info *info);
static void parse_mult_monopoles(FILE *fp, std_efp_info *info);
static void parse_mult_dipoles(FILE *fp, std_efp_info *info);
static void parse_mult_quadrupoles(FILE *fp, std_efp_info *info);
static void parse_mult_octupoles(FILE *fp, std_efp_info *info);
static void parse_mult_screens(FILE *fp, std_efp_info *info);
static void parse_pols(FILE *fp, std_efp_info *info);
static void parse_principal(FILE *fp, std_efp_info *info);
static void gen_axis(std_efp_info *ptr);
static void gen_masscenter(std_efp_info *ptr);

std_efp_info *parse_efp(char[]);
const std_efp_info *get_std_efp_info(const char *key) {
  std_efp_info *result = NULL;

  result = list_lookup(pefp, lookup_efp, key);
  if (result == NULL) {
    char filename[84];
    const char suffix[] = ".efp";
    snprintf(filename, sizeof(filename) - sizeof(suffix) + 1, "%s", key);
    strcat(filename, suffix);
    result = parse_efp(filename);
    list_append(&pefp, result);
    strncpy(result->name, key, sizeof(result->name) - 1);
  }
  return result;
}

std_efp_info *parse_efp(char filename[]) {
  char buf[1024];
  char *key[] = {" COORDINATES", " MONOPOLES",    " DIPOLES",
                 " QUADRUPOLES", " OCTUPOLES",    " POLARIZABLE POINTS",
                 "SCREEN2",      "PRINCIPAL AXIS"};
  int n = 0;

  FILE *fp = fopen(filename, "r");
  std_efp_info *eptr = galloc(sizeof(std_efp_info));

  while (!feof(fp)) {
    fgets(buf, sizeof(buf), fp);
    if (n < sizeof(key) / sizeof(void *) &&
        strncmp(buf, key[n], sizeof(key[n]) - 1) == 0) {
      switch (n) {
      case 0:
        parse_mult_basic(fp, eptr);
        break;
      case 1:
        parse_mult_monopoles(fp, eptr);
        break;
      case 2:
        parse_mult_dipoles(fp, eptr);
        break;
      case 3:
        parse_mult_quadrupoles(fp, eptr);
        break;
      case 4:
        parse_mult_octupoles(fp, eptr);
        break;
      case 5:
        parse_pols(fp, eptr);
        break;
      case 6:
        parse_mult_screens(fp, eptr);
        break;
      case 7:
        parse_principal(fp, eptr);
        break;
      }
      ++n;
    } else {
      continue;
    }
  }
  list_dump(pmult, sizeof(mult_point), CAST_PTR(&(eptr->mult_ptr), void *));
  list_dump(ppol, sizeof(pol_point), CAST_PTR(&(eptr->pol_ptr), void *));
  list_dump(pbond, sizeof(bond), CAST_PTR(&(eptr->bond_ptr), void *));
  gen_axis(eptr);
  gen_masscenter(eptr);
  list_clear(&pmult);
  list_clear(&ppol);
  list_clear(&pbond);
  fclose(fp);
  return eptr;
}

static int lookup_efp(void *pelem, const void *key) {
  std_efp_info *ptr = CAST_PTR(pelem, std_efp_info);
  return strncmp(ptr->name, key, sizeof(ptr->name)) == 0 ? 0 : 1;
}

static void parse_mult_basic(FILE *fp, std_efp_info *info) {
  char buf[1024], str[80];
  char end[] = " STOP";
  size_t index = 1, len, k;
  double tmppos[3], tmpnuc;
  void *ptr;
  mult_point *mptr;
  bond *bptr;

  while (!feof(fp)) {
    fgets(buf, sizeof(buf), fp);
    if (strncmp(buf, end, sizeof(end) - 1) == 0) {
      break;
    } else {
      sscanf(buf, "%s %lf %lf %lf %*s %lf", str, tmppos, tmppos + 1, tmppos + 2,
             &tmpnuc);
      info->n_mult_points += 1;
      if (str[0] == 'A') {
        info->n_atoms += 1;
      } else {
        while (index < info->n_atoms) {
          len = sprintf(buf, "%zu", index + 1);
          if (strncmp(buf, str + 2, len) == 0) {
            ptr = galloc(sizeof(bond));
            list_append(&pbond, ptr);
            bptr = CAST_PTR(ptr, bond);
            sscanf(str + 2 + len, "%lu", &(bptr->index[1]));
            bptr->index[0] = index;
            bptr->index[1] -= 1;
            break;
          } else {
            index += 1;
          }
        }
      }
      ptr = galloc(sizeof(mult_point));
      list_append(&pmult, ptr);
      mptr = CAST_PTR(ptr, mult_point);
      for (k = 0; k < 3; ++k) {
        mptr->position[k] = tmppos[k];
      }
      mptr->nuclear = tmpnuc;
    }
  }
  info->n_bonds = info->n_mult_points - info->n_atoms;
}

static void parse_mult_monopoles(FILE *fp, std_efp_info *info) {
  char buf[1024];
  double tmpmono;

  for (size_t i = 0; i < info->n_mult_points; ++i) {
    fgets(buf, sizeof(buf), fp);
    sscanf(buf, "%*s %lf %*s", &tmpmono);
    CAST_PTR(list_index(pmult, i), mult_point)->monopole = tmpmono;
  }
}

static void parse_mult_dipoles(FILE *fp, std_efp_info *info) {
  char buf[1024];
  double tmpdip[3];
  size_t k;
  mult_point *ptr;

  for (size_t i = 0; i < info->n_mult_points; ++i) {
    fgets(buf, sizeof(buf), fp);
    sscanf(buf, "%*s %lf %lf %lf", tmpdip, tmpdip + 1, tmpdip + 2);
    ptr = CAST_PTR(list_index(pmult, i), mult_point);
    for (k = 0; k < 3; ++k) {
      ptr->dipole[k] = tmpdip[k];
    }
  }
}

static void parse_mult_quadrupoles(FILE *fp, std_efp_info *frag) {
  char buf[2048];
  double tmpqua[6];
  size_t n = 0, k;
  mult_point *ptr;

  for (size_t i = 0; i < frag->n_mult_points; ++i) {
    fgets(buf, sizeof(buf), fp);
    n = strlen(buf);
    fgets(buf + n, sizeof(buf) - n, fp);
    sscanf(buf, "%*s %lf %lf %lf %lf %*s %lf %lf", tmpqua, tmpqua + 1,
           tmpqua + 2, tmpqua + 3, tmpqua + 4, tmpqua + 5);
    ptr = CAST_PTR(list_index(pmult, i), mult_point);
    for (k = 0; k < 6; ++k) {
      ptr->quadrupole[k] = tmpqua[k];
    }
  }
}

static void parse_mult_octupoles(FILE *fp, std_efp_info *frag) {
  char buf[3072];
  double tmpoct[10];
  size_t n = 0, k;
  mult_point *ptr;

  for (size_t i = 0; i < frag->n_mult_points; ++i) {
    fgets(buf, sizeof(buf), fp);
    n = strlen(buf);
    fgets(buf + n, sizeof(buf) - n, fp);
    n = strlen(buf);
    fgets(buf + n, sizeof(buf) - n, fp);
    sscanf(buf, "%*s %lf %lf %lf %lf %*s %lf %lf %lf %lf %*s %lf %lf", tmpoct,
           tmpoct + 1, tmpoct + 2, tmpoct + 3, tmpoct + 4, tmpoct + 5,
           tmpoct + 6, tmpoct + 7, tmpoct + 8, tmpoct + 9);
    ptr = CAST_PTR(list_index(pmult, i), mult_point);
    for (k = 0; k < 10; ++k) {
      ptr->octupole[k] = tmpoct[k];
    }
  }
}

static void parse_mult_screens(FILE *fp, std_efp_info *frag) {
  char buf[1024];
  double tmpscr;

  for (size_t i = 0; i < frag->n_mult_points; ++i) {
    fgets(buf, sizeof(buf), fp);
    sscanf(buf, "%*s %*s %lf", &tmpscr);
    CAST_PTR(list_index(pmult, i), mult_point)->screen = tmpscr;
  }
}

static void parse_pols(FILE *fp, std_efp_info *info) {
  char buf[3072];
  char end[] = " STOP";
  size_t n = 0, k;
  double tmppos[3], tmppol[9];
  void *ptr;
  pol_point *pptr;

  while (!feof(fp)) {
    fgets(buf, sizeof(buf), fp);
    if (strncmp(buf, end, sizeof(end) - 1) == 0) {
      break;
    } else {
      sscanf(buf, "%*s %lf %lf %lf", tmppos, tmppos + 1, tmppos + 2);
      fgets(buf, sizeof(buf), fp);
      n = strlen(buf);
      fgets(buf + n, sizeof(buf) - n, fp);
      n = strlen(buf);
      fgets(buf + n, sizeof(buf) - n, fp);
      sscanf(buf, "%lf %lf %lf %lf %*s %lf %lf %lf %lf %*s %lf", tmppol,
             tmppol + 1, tmppol + 2, tmppol + 3, tmppol + 4, tmppol + 5,
             tmppol + 6, tmppol + 7, tmppol + 8);
      info->n_pol_points += 1;
      ptr = galloc(sizeof(pol_point));
      list_append(&ppol, ptr);
      pptr = CAST_PTR(ptr, pol_point);
      for (k = 0; k < 3; ++k) {
        pptr->position[k] = tmppos[k];
      }
      pptr->polarizability[0] = tmppol[0];
      pptr->polarizability[1] = tmppol[3];
      pptr->polarizability[2] = tmppol[4];
      pptr->polarizability[3] = tmppol[6];
      pptr->polarizability[4] = tmppol[1];
      pptr->polarizability[5] = tmppol[5];
      pptr->polarizability[6] = tmppol[7];
      pptr->polarizability[7] = tmppol[8];
      pptr->polarizability[8] = tmppol[2];
    }
  }
}

static void parse_principal(FILE *fp, std_efp_info *info) {
  char buf[1024];
  size_t tmpint[4], k;

  fgets(buf, sizeof(buf), fp);
  sscanf(buf, "%lu %lu %lu %lu", tmpint, tmpint + 1, tmpint + 2, tmpint + 3);
  for (k = 0; k < 4; ++k) {
    info->principal[k] = tmpint[k] - 1;
  }
}

static void gen_axis(std_efp_info *ptr) {
  const mult_point *mptr = ptr->mult_ptr;

  vector va, vb, vc, vd;
  vector_sub(mptr[ptr->principal[1]].position, mptr[ptr->principal[0]].position,
             va);
  vector_sub(mptr[ptr->principal[3]].position, mptr[ptr->principal[2]].position,
             vb);

  vector_normalize(va, vc);
  vector_scalar_mul(vc, vector_dot(vb, vc), va);
  vector_sub(vb, va, vd);
  vector_normalize(vd, va);
  vector_cross(vc, va, vb);

  ptr->axis[0] = vc[0];
  ptr->axis[1] = va[0];
  ptr->axis[2] = vb[0];
  ptr->axis[3] = vc[1];
  ptr->axis[4] = va[1];
  ptr->axis[5] = vb[1];
  ptr->axis[6] = vc[2];
  ptr->axis[7] = va[2];
  ptr->axis[8] = vb[2];
}

static void gen_masscenter(std_efp_info *ptr) {
  vector tmpvec;
  const mult_point *mptr = ptr->mult_ptr;
  double w_atom, w_total = 0.0;

  for (size_t k = 0; k < ptr->n_atoms; ++k) {
    w_atom = GET_ATOM_MASS(mptr[k].nuclear);
    w_total += w_atom;
    vector_scalar_mul(mptr[k].position, w_atom, tmpvec);
    vector_sum_inplace(ptr->masscenter, tmpvec);
  }
  vector_scalar_mul(ptr->masscenter, 1 / w_total, tmpvec);
  vector_dup(tmpvec, ptr->masscenter);
}
