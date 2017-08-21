#include "utils.h"
#include <stdlib.h>
#include <string.h>

void list_append(list_ptr *plptr, void *pelem) {
  list_ptr tail = *plptr;
  list_ptr ptr = galloc(LIST_ELEMENT_SIZE);
  memset(ptr, 0, LIST_ELEMENT_SIZE);
  if (tail == NULL) {
    tail = *plptr = ptr;
  } else {
    while (tail->next != NULL) {
      tail = tail->next;
    }
    tail = tail->next = ptr;
  }
  tail->pelem = pelem;
}

void *list_index(list_ptr lptr, size_t idx) {
  size_t i = 0;
  while (lptr != NULL) {
    if (i == idx) {
      break;
    }
    lptr = lptr->next;
    ++i;
  }
  return (lptr == NULL) ? NULL : lptr->pelem;
}

size_t list_length(list_ptr lptr) {
  size_t len = 0;
  while (lptr != NULL) {
    lptr = lptr->next;
    ++len;
  }
  return len;
}

void list_dump(list_ptr lptr, size_t size, void **dest_ptr, size_t *order) {
  void *ptr;
  size_t nmemb = list_length(lptr);
  *dest_ptr = ptr = galloc(nmemb * size);

  if (order == NULL) {
    while (lptr != NULL) {
      memcpy(ptr, lptr->pelem, size);
      ptr += size;
      lptr = lptr->next;
    }
  } else {
    for (size_t i = 0; i < nmemb; ++i) {
      memcpy(ptr, list_index(lptr, order[i]), size);
      ptr += size;
    }
  }
}

void *list_lookup(list_ptr lptr, list_lookup_function lookup,
                  const void *target) {
  void *ptr = NULL;
  while (lptr != NULL) {
    if (lookup(lptr->pelem, target) == 0) {
      ptr = lptr->pelem;
      break;
    }
    lptr = lptr->next;
  }
  return ptr;
}

void list_foreach(list_ptr lptr, list_foreach_function foreach) {
  while (lptr != NULL) {
    foreach (lptr->pelem)
      ;
    lptr = lptr->next;
  }
}

void list_clear(list_ptr *pptr) {
  list_ptr head = *pptr;
  while (head != NULL) {
    free(head->pelem);
    head = head->next;
    free(*pptr);
    *pptr = head;
  }
}
