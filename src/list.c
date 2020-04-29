#include <stdlib.h>
#include <string.h>
#include "list.h"



/*
����:�����б�
*/
List *make_list()
{
	List *l = (List*)malloc(sizeof(List));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}


/*
����:�������е���Ԫ��
*/
void *list_pop(List *l)
{
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;
    
    return val;
}


/*
����:������β������Ԫ��
*/
void list_insert(List *l, void *val)
{
	node *newNode = (node*)malloc(sizeof(node));
	newNode->val = val;
	newNode->next = 0;

	if(!l->back)
	{
		l->front = newNode;
		newNode->prev = 0;
	}
	else
	{
		l->back->next = newNode;
		newNode->prev = l->back;
	}
	l->back = newNode;
	++l->size;
}


/*
����:�ͷŽڵ�
*/
void free_node(node *n)
{
	node *next;
	while(n) 
	{
		next = n->next;
		free(n);
		n = next;
	}
}


/*
����:�ͷ�����
*/
void free_list(List *l)
{
	free_node(l->front);
	free(l);
}


/*
���ܣ��ͷŽڵ�����
*/
void free_list_contents(List *l)
{
	node *n = l->front;
	while(n)
	{
		free(n->val);
		n = n->next;
	}
}


/*
���ܣ�����תΪ����
*/
void **list_to_array(List *l)
{
    void **a = (void**)calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n)
    {
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}
