#ifndef _FEATURE_H_
#define _FEATURE_H_

typedef struct haar_feature
{
	u16 i;
	u16 j;

	u16 w;/*�������εĿ��*/
	u16 h;/*�������εĸ߶�*/

	u16 src_wnd_size;/*�����������Դ���ڴ�С����������*/
	u8 type;

}Haar_feat;

#endif