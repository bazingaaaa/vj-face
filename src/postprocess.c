#include <stdio.h>
#include <vector>
#include "type.h"
#include "const.h"
#include "image.h"
#include "feature.h"
#include "classifier.h"
#include "model.h"
#include "data.h"
#include "list.h"
#include "proto.h"

using namespace std;


static bool cmp(Sub_wnd wnd1, Sub_wnd wnd2);

// int *node; //每个节点 
// int *rank; //树的高度 
 
// //初始化n个节点 
// void init(int n)
// { 
// 	node = 
// 	for(int i = 0; i < n; i++)
// 	{ 
// 	 	node[i] = i; 
// 	 	rank[i] = 0; 
// 	} 
// }


// //查找当前元素所在树的根节点(代表元素) 
// int find(int x)
// { 
// 	if(x == node[x]) 
//  		return x; 
// 	return node[x] = find(node[x]); //在第一次查找时，将节点直连到根节点 
// } 


// //合并元素x， y所处的集合 
// void unite(int x, int y)
// { 
// 	//查找到x，y的根节点 
// 	x = find(x); 
// 	y = find(y); 
// 	if(x == y) 
//  		return ; 
// 	//判断两棵树的高度，然后在决定谁为子树 
// 	if(rank[x] < rank[y])
// 	{ 
//  		node[x] = y; 
// 	}
// 	else
// 	{ 
//  		node[y] = x; 
//  		if(rank[x] == rank[y]) 
//  			rank[x]++: 
// 	} 
// } 


// //判断x，y是属于同一个集合 
// bool same(int x, int y)
// { 
// 	return find(x) == find(y); 
// } 


// void free_data()
// {
// 	free(node);
// 	free(rank);
// }

/*
function:
param:
return:
*/
void postprocess(vector<Sub_wnd> &candidate)  
{
	i32 i, j, n = candidate.size();

	vector<Sub_wnd> temp(candidate);
	vector<bool> flags;

	sort(temp.begin(), temp.end(), cmp);


	for(i = 0; i < n; i++)
	{
		flags.push_back(true);
	}
	for(i = 0; i < n; i++)
	{
		for(j = i + 1; j < n; j++)
		{
			if(flags[i] && is_inside(temp[j], temp[i]))
			{
				flags[j] = false;
			}
		}
	}

	candidate.resize(0);
	for(i = 0; i < n; i++)
	{
		if(flags[i])
		{
			candidate.push_back(temp[i]);
		}
	}
}


static bool cmp(Sub_wnd wnd1, Sub_wnd wnd2)
{
	return wnd1.size > wnd2.size;
}