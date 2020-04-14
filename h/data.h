#ifndef _DATA_H_
#define _DATA_H_

typedef struct training_example
{
	image integ;
	i32 label;
	i32 predict_label;
	float feat_val;
	float weight;
}Train_example;


typedef struct
{
	float feat_val;
	float weight;
	i32 label;
}Feat_info;


typedef struct data
{
	image *im_array;
	i32 im_num;
}Data;


#endif