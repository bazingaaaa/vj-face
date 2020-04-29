#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_


typedef struct decision_stump
{
	Haar_feat feat;
	float thresh;
	float error;/*����Ȩ�ص����*/
	float margin;
	i32 sign;

	float weight;
	struct decision_stump *next_stump;/*��һ������׮*/
}Stump;


/*�����ɶ��Stump���*/
typedef struct decision_stage
{
	i32 stump_num;
	float shift;
	Stump *head_stump;/*��һ������׮*/
	Stump *tail_stump;/*β������׮*/
	struct decision_stage *next_stage;/*��һ���׶�*/
}Stage;


#endif