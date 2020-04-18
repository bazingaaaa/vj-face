#ifndef _DATA_H_
#define _DATA_H_

typedef struct training_example
{
	image integ;
	i32 label;
	i32 predict_label;
	float feat_val;
	float weight;

	u16 size;/*窗口大小*/
    u16 x;/*X轴是竖直方向*/
    u16 y;/*Y轴是水平方向*/
	image src_img;
}Train_example;


typedef struct
{
	float confidence;
	i32 cc_id;
	i32 pos_i;
	i32 pos_j;
	i32 size;
	image integ;
}Sub_wnd;


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