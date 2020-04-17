#ifndef _LIST_H_
#define _LIST_H_

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} List;


List *make_list();
int list_find(List *l, void *val);
void list_insert(List *, void *);
void free_list_contents(List *l);
void **list_to_array(List *l);
void free_list(List *l);

#endif
