#ifndef _MODEL_H_
#define _MODEL_H_


typedef struct model
{
	i32 stage_num;
	double fpr;/*ѵ�����õ�fpr�����ڼ���ѵ��*/
	Stage *head_stage;
}Model;


#endif