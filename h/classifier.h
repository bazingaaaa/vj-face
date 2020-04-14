#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_


typedef struct decision_stump
{
	Haar_feat feat;
	float thresh;
	float error;/*考虑权重的误差*/
	float margin;
	i32 sign;

	float weight;
	struct decision_stump *next_stump;/*下一个决策桩*/
}Stump;


/*可能由多个Stump组成*/
typedef struct decision_stage
{
	i32 stump_num;
	float shift;
	Stump *head_stump;/*下一个决策桩*/
	Stump *tail_stump;/*尾部决策桩*/
	struct decision_stage *next_stage;/*下一个阶段*/
}Stage;


#endif