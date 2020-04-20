#include "const.h"
#include "image.h"
#include "feature.h"
#include "classifier.h"
#include "model.h"
#include "data.h"
#include <vector>
#include "list.h"
#include "proto.h"
#include <math.h>



typedef float(*Feat_func)(image, i32, i32, i32, i32); 


static int compare(const void *p1, const void *p2);


/*
功能：寻找决策桩
参数：array-训练样本实例信息，已经按照特征值由低到高排列
备注：对应论文An Analysis of the Viola-Jones Face Detection Algorithm中的算法4
*/
Stump search_decision_stump(Feat_info *parallel_examples, i32 example_num)
{
	float thresh_opt = parallel_examples[0].feat_val - 1;/*初始门限*/
	i32 sign_opt;
	float M_opt = 0;
	float error_opt = 2;
	i32 i, j;
	float W_1_pos = 0, W_1_neg = 0, W_n1_pos = 0, W_n1_neg = 0;
	float err_pos, err_neg;	

	/*初始化W_1_pos, W_1_neg, W_n1_pos, W_n1_neg*/
	for(i = 0; i < example_num; i++)
	{
		if(parallel_examples[i].label == 1)
		{
			W_1_pos += parallel_examples[i].weight;
		}
		else
		{
			W_n1_pos += parallel_examples[i].weight;
		}
	}
	
	float M, error, thresh, sign;
	j = 0;
	thresh = thresh_opt;
	M = M_opt;

	while(1)
	{
		err_pos = W_1_neg + W_n1_pos;
		err_neg = W_1_pos + W_n1_neg;

		/*确定分类器的极性，即判断大于门限值的样本应该判为正样本还是负样本*/
		if(err_pos < err_neg)
		{
			error = err_pos;
			sign = 1;
		}
		else
		{
			error = err_neg;
			sign = -1;
		}

		/*与最优的错误率和冗余进行比较（优先考虑错误率）*/
		if(error < error_opt || (error == error_opt && M > M_opt))
		{
			error_opt = error;
			thresh_opt = thresh;
			M_opt = M;
			sign_opt = sign;
		}

		/*处理完所有样本*/
		if(j == example_num)
			break;
		j++;
		while(1)
		{
			if(-1 == parallel_examples[j - 1].label)
			{
				W_n1_neg += parallel_examples[j - 1].weight;
				W_n1_pos -= parallel_examples[j - 1].weight;
			}
			else
			{
				W_1_neg += parallel_examples[j - 1].weight;
				W_1_pos -= parallel_examples[j - 1].weight;
			}

			if(j == example_num || (parallel_examples[j].feat_val != parallel_examples[j - 1].feat_val))
			{
				break;
			}
			else
			{
				j++;
			}
		}
		if(j == example_num)
		{
			thresh = parallel_examples[example_num - 1].feat_val + 1;
			M = 0;
		}
		else
		{
			thresh = (parallel_examples[j - 1].feat_val + parallel_examples[j].feat_val) / 2;
			M = parallel_examples[j].feat_val - parallel_examples[j - 1].feat_val;
		}
	}
	Stump ret;
	ret.thresh = thresh_opt;
	ret.error = error_opt;
	ret.margin = M_opt;
	ret.sign = sign_opt;
	return ret;
}


/*
功能：找到最佳决策桩
备注：对应论文An Analysis of the Viola-Jones Face Detection Algorithm中的算法5
*/
Stump *find_best_stump(Feat_info **parallel_examples, Train_example *examples, i32 example_num, Haar_feat *feat_array, i32 feat_num)
{
	Stump *best_stump = (Stump*)malloc(sizeof(Stump));
	Stump *candidate = (Stump*)malloc(sizeof(Stump) * feat_num);
	i32 i;

	best_stump->error = 2;
	best_stump->feat = feat_array[0];

	/*计算所有的特征值并排序*/
 	#pragma omp parallel for
	for(i = 0; i < feat_num; i++)
	{
		calc_example_feat_val(parallel_examples[i], examples, example_num, &feat_array[i]);
		qsort(parallel_examples[i], example_num, sizeof(Feat_info), compare);
		candidate[i] = search_decision_stump(parallel_examples[i], example_num);
	}

	/*找寻最优的决策桩*/
	for(i = 0; i < feat_num; i++)
	{
		/*搜索决策桩*/
		if(candidate[i].error < best_stump->error 
			|| (candidate[i].error == best_stump->error && candidate[i].margin > best_stump->margin))
		{
			*best_stump = candidate[i];/*找到了一个更好的决策桩*/
			best_stump->feat = feat_array[i];
		}
	}

	free(candidate);
	
	return best_stump;
}


/*
功能：计算训练样本对应的特征值
*/
void calc_example_feat_val(Feat_info *parallel_examples, Train_example *examples, i32 example_num, Haar_feat *pFeat)
{
	i32 i;
	static Feat_func func[5] = {calc_featA_val, calc_featB_val, calc_featC_val, calc_featD_val, calc_featE_val}; 
	Feat_func f = func[pFeat->type];
	
	for(i = 0; i < example_num; i++)
	{
		parallel_examples[i].feat_val = f(examples[i].integ, pFeat->i, pFeat->j, pFeat->w, pFeat->h);
		parallel_examples[i].weight = examples[i].weight;
		parallel_examples[i].label = examples[i].label;
	}
}



/*
功能：比较操作
备注：注意用法，很容易导致错误
*/
static int compare(const void *p1, const void *p2)
{
	Feat_info *s1 = (Feat_info*)p1;
	Feat_info *s2 = (Feat_info*)p2;
	
	//return s1->feat_val > s2->feat_val ? 1 : -1;
	if(s1->feat_val > s2->feat_val)
	{
		return 1;
	}
	else
	{
		return -1;
	}
}


/*
功能：测试单个树桩的识别性能
备注：测试程序正确性代码，可删除
*/
double test_stump(Stump *stmp, Train_example *examples, i32 example_num)
{
	i32 i;
	i32 count = 0;
	
	/*统计正确判断的样本数*/
	for(i = 0; i < example_num; i++)
	{
		if(examples[i].label == stump_func(stmp, examples[i].integ, 0))
		{
			count++;
		}
	}
	return count * 1.0 / example_num;
}


/*
功能：训练adaboost分类器
备注：对应论文An Analysis of the Viola-Jones Face Detection Algorithm中的算法7
*/
Stage *adaboost(Feat_info** parallel_examples, Train_example *examples, i32 example_num, i32 pos_num, i32 neg_num, Haar_feat *feat_array, i32 feat_num, i32 depth)
{
	Stage *s = (Stage*)malloc(sizeof(Stage));
	Stump *tail = NULL;
	i32 t, i;
	float err_t, a_t;
	i8 predict_label;
	
	s->stump_num = 0;
	
	/*权重初始化*/
	for(i = 0; i < example_num; i++)
	{
		if(examples[i].label == 1)
		{
			examples[i].weight = 1.0 / pos_num / 2;
		}
		else
		{
			examples[i].weight = 1.0 / neg_num / 2;
		}
	}

	for(t = 0; t < depth; t++)
	{
		Stump *stmp = find_best_stump(parallel_examples, examples, example_num, feat_array, feat_num);
		//times("add one stump\n");
		err_t = stmp->error;

		/*增加一个决策桩*/
		s->stump_num = s->stump_num + 1;
		if(NULL == tail)
		{
			s->head_stump = stmp;
			tail = stmp;
		}
		else
		{
			tail->next_stump = stmp;
			tail = stmp;
		}
		if(err_t == 0 && t == 0)
		{
			/*此时只有一个决策桩，权重值没有任何意义（不改变结果正负号）*/
			break;
		}

		a_t = 1.0 / 2 * log((1 - err_t) / err_t);
		stmp->weight = a_t;

		/*更新权重值*/
    	#pragma omp parallel for
		for(i = 0; i < example_num; i++)
		{
			predict_label = stump_func(stmp, examples[i].integ, 1);
			if(predict_label == examples[i].label)/*判断正确的样本*/
			{
				examples[i].weight = examples[i].weight / 2 / (1 - err_t);
			}
			else/*判断错误的样本*/
			{
				examples[i].weight = examples[i].weight / 2 / err_t;
			}
		}
		
	}
	if(tail != NULL)
		tail->next_stump = NULL;
	s->tail_stump = tail;

	return s;
}


/*
功能：决策桩决策函数
备注：区分训练和非训练
*/
i8 stump_func(Stump *stmp, image integ, i32 train_flag)
{
	i32 i;
	static Feat_func func[5] = {calc_featA_val, calc_featB_val, calc_featC_val, calc_featD_val, calc_featE_val}; 
	Feat_func f = func[stmp->feat.type];
	float feat_val;

	if(1 == train_flag)/*训练过程中用到的图片大小均为24x24*/
	{
		feat_val = f(integ, stmp->feat.i, stmp->feat.j, stmp->feat.w, stmp->feat.h);
	}
	else/*非训练过程计算特征值时需要scale*/
	{
		feat_val = calc_haar_feat_val(integ, &stmp->feat);
	}

	return stmp->sign * (feat_val > stmp->thresh ? 1 : -1);
}


/*
功能：决策阶段决策函数
*/
i8 stage_func(Stage *s, image integ, i32 train_flag)
{
	Stump *stmp = s->head_stump;
	float weight_rslt = 0;
	float shift = train_flag == 1 ? 0 : s->shift;

	while(stmp != NULL)
	{
		weight_rslt += (stmp->weight * (stump_func(stmp, integ, train_flag) + shift));
		stmp = stmp->next_stump;
	}

	return weight_rslt > 0 ? 1 : -1;
}


/*
功能：测试单个树桩的识别性能
备注：测试程序正确性代码，可删除
*/
double test_stage(Stage *s, Train_example *examples, i32 example_num)
{
	i32 i;
	i32 count = 0;
	
	/*统计正确判断的样本数*/
	for(i = 0; i < example_num; i++)
	{
		if(examples[i].label == stage_func(s, examples[i].integ, 0))
		{
			count++;
		}
	}
	return count * 1.0 / example_num;
}


/*
功能：在决策阶段中增加一个决策桩
*/
void add_stump_2_stage(Stage *s, Feat_info** parallel_examples, Train_example *examples, i32 example_num, Haar_feat *feat_array, i32 feat_num)
{
	i32 t, i;
	float err_t, a_t;
	i8 predict_label;

	Stump *stmp = find_best_stump(parallel_examples, examples, example_num, feat_array, feat_num);
	err_t = stmp->error;

	/*增加一个决策桩*/
	s->stump_num = s->stump_num + 1;
	
	/*计算该决策桩权重*/
	a_t = 1.0 / 2 * log((1 - err_t) / err_t);
	stmp->weight = a_t;

	/*更新所有样本的权重值*/
	#pragma omp parallel for
	for(i = 0; i < example_num; i++)
	{
		predict_label = stump_func(stmp, examples[i].integ, 1);
		if(predict_label == examples[i].label)/*判断正确的样本*/
		{
			examples[i].weight = examples[i].weight / 2 / (1 - err_t);
		}
		else/*判断错误的样本*/
		{
			examples[i].weight = examples[i].weight / 2 / err_t;
		}
	}

	s->tail_stump->next_stump = stmp;
	s->tail_stump = stmp;
	stmp->next_stump = NULL;
}
