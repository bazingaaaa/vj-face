#ifndef _MODEL_H_
#define _MODEL_H_


typedef struct model
{
	i32 stage_num;
	double fpr;/**训练所得的fpr，用于继续训练*/
	Stage *head_stage;
}Model;


#endif