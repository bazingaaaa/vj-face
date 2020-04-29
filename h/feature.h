#ifndef _FEATURE_H_
#define _FEATURE_H_

typedef struct haar_feature
{
	u16 i;
	u16 j;

	u16 w;/*单个矩形的宽度*/
	u16 h;/*单个矩形的高度*/

	u16 src_wnd_size;/*计算该特征的源窗口大小，用于缩放*/
	u8 type;

}Haar_feat;

#endif