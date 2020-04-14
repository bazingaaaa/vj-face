#include <stdlib.h>
#include <string.h>
#include "list.h"



/*
功能:生成列表
*/
list *make_list()
{
	list *l = (list*)malloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}


/*
功能:从链表中弹出元素
*/
void *list_pop(list *l){
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
功能:在链表尾部插入元素
*/
void list_insert(list *l, void *val)
{
	node *newNode = (node*)malloc(sizeof(node));
	newNode->val = val;
	newNode->next = 0;

	if(!l->back){
		l->front = newNode;
		newNode->prev = 0;
	}else{
		l->back->next = newNode;
		newNode->prev = l->back;
	}
	l->back = newNode;
	++l->size;
}


/*
功能:释放节点
*/
void free_node(node *n)
{
	node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}


/*
功能:释放链表
*/
void free_list(list *l)
{
	free_node(l->front);
	free(l);
}


/*
功能：释放节点内容
*/
void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}


/*
功能：链表转为数组
*/
void **list_to_array(list *l)
{
    void **a = (void**)calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}
