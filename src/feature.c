#include <math.h>
#include "const.h"
#include "image.h"
#include "feature.h"

/*
功能：计算指定窗口内的haar特征数量
参数：wndSize-窗口大小（正方形）
	 featW-特征矩形宽度
	 featH-特征矩形高度
*/
u32 calc_haar_feat_num(u16 wndSize, u16 w, u16 h)
{
	/*计算竖轴*/
	u16 ch = floor(wndSize * 1.0 / h);
	u16 nh = ch * (wndSize + 1) - ((1 + ch) * ch) / 2 * h;
	/*计算横轴*/
	u16 cw = floor(wndSize * 1.0 / w);
	u16 nw = cw * (wndSize + 1) - ((1 + cw) * cw) / 2 * w;
	return nh * nw;
}


/*
功能：填写haar特征参数，记录特征信息
备注：x为垂直方向，y为水平方向
*/
void make_haar_feat(Haar_feat *pFeat, u8 type, u16 i, u16 j, u16 w, u16 h, u16 srcWndSize)
{
	pFeat->type = type;
	pFeat->i = i;
	pFeat->j = j;
	pFeat->w = w;
	pFeat->h = h;
	pFeat->srcWndSize = srcWndSize;
}


/*
功能：计算A类特征值
*/	
float calc_featA_val(image integ, i32 i, i32 j, i32 w, i32 h)
{
 	return calc_im_sum(integ, i, i + h - 1, j, j + w - 1)
 		   -calc_im_sum(integ, i, i + h - 1, j + w, j + 2 * w - 1);
}


/*
功能：计算B类特征值
*/
float calc_featB_val(image integ, i32 i, i32 j, i32 w, i32 h)
{
 	return calc_im_sum(integ, i, i + h - 1, j, j + w - 1)
 			-calc_im_sum(integ, i, i + h - 1, j + w, j + 2 * w - 1)
			+calc_im_sum(integ, i, i + h - 1, j + 2 * w, j + 3 * w - 1);
}


/*
功能：计算C类特征值
*/
float calc_featC_val(image integ, i32 i, i32 j, i32 w, i32 h)
{
 	return calc_im_sum(integ, i, i + h - 1, j, j + w - 1)
 			-calc_im_sum(integ, i + h, i + 2 * h - 1, j, j + w - 1);
}


/*
功能：计算D类特征值
*/
float calc_featD_val(image integ, i32 i, i32 j, i32 w, i32 h)
{
 	return calc_im_sum(integ, i, i + h - 1, j, j + w - 1)
 			-calc_im_sum(integ, i + h, i + 2 * h - 1, j, j + w - 1)
			+calc_im_sum(integ, i + 2 * h, i + 3 * h - 1, j, j + w - 1);
}


/*
功能：计算E类特征值
*/
float calc_featE_val(image integ, i32 i, i32 j, i32 w, i32 h)
{
 	return calc_im_sum(integ, i, i + h - 1, j, j + w - 1)
 			-calc_im_sum(integ, i + h, i + 2 * h - 1, j, j + w - 1)
			-calc_im_sum(integ, i, i + h - 1, j + w, j + 2 * w - 1)
			+calc_im_sum(integ, i + h, i + 2 * h - 1, j + w, j + 2 * w - 1);
}


/*
功能：计算haar特征值(考虑了scaling的情况)
参数：pWnd-指向子窗的指针
	 pFeat-指向特征的指针
备注：对应论文An Analysis of the Viola-Jones Face Detection Algorithm中的算法3
*/
float calc_haar_feat_val(image integ, Haar_feat *pFeat)
{
	float ret = 0;
	assert(pFeat->srcWndSize != 0 && pFeat->srcWndSize <= integ.w);
	i32 i = pFeat->i, j = pFeat->j, w = pFeat->w, h = pFeat->h, a;
	i32 e = integ.w;
	i32 s = pFeat->srcWndSize;

	switch(pFeat->type)
	{
		case FEAT_A:
			a = 2 * w * h;
 			i = NEAREST_INTEGER(i * e / s);
 			j = NEAREST_INTEGER(j * e / s);
 			h = NEAREST_INTEGER(h * e / s);
 			w = MIN((1 + 2 * w * e / s) / 2, (e - j + 1) / 2);
			ret = calc_featA_val(integ, i, j, w, h) * a / (2 * w * h);
			break;
	 	case FEAT_B:
	 		a = 3 * w * h;
 			i = NEAREST_INTEGER(i * e / s);
 			j = NEAREST_INTEGER(j * e / s);
 			h = NEAREST_INTEGER(h * e / s);
 			w = MIN((1 + 3 * w * e / s) / 3, (e - j + 1) / 3);
			ret = calc_featB_val(integ, i, j, w, h) * a / (3 * w * h);
	 		break;
	 	case FEAT_C:
	 		a = 2 * w * h;
 			i = NEAREST_INTEGER(i * e / s);
 			j = NEAREST_INTEGER(j * e / s);
 			w = NEAREST_INTEGER(w * e / s);
 			h = MIN((1 + 2 * h * e / s) / 2, (e - i + 1) / 2);
			ret = calc_featC_val(integ, i, j, w, h) * a / (2 * w * h);
	 		break;
		case FEAT_D:
			a = 3 * w * h;
 			i = NEAREST_INTEGER(i * e / s);
 			j = NEAREST_INTEGER(j * e / s);
 			w = NEAREST_INTEGER(w * e / s);
 			h = MIN((1 + 3 * h * e / s) / 3, (e - i + 1) / 3);
			ret = calc_featD_val(integ, i, j, w, h) * a / (3 * w * h);
	 		break;
		case FEAT_E:
			a = 4 * w * h;
 			i = NEAREST_INTEGER(i * e / s);
 			j = NEAREST_INTEGER(j * e / s);
 			w = MIN((1 + 2 * w * e / s) / 2, (e - j + 1) / 2);
 			h = MIN((1 + 2 * h * e / s) / 2, (e - i + 1) / 2);
			ret = calc_featE_val(integ, i, j, w, h) * a / (4 * w * h);
	 		break;
		default:
	 		break;
	}
	return ret;
}


/*
功能：生成窗口内指定类型的特征
返回值：总特征数量
*/
i32 make_specific_feat(u8 type, u16 wndSize, Haar_feat *feat_array)
{
	i32 count = 0;
	u16 w_array[] = {2, 3, 1, 1, 2};/*对应各类型haar特征最小宽度*/
	u16 h_array[] = {1, 1, 2, 3, 2};/*对应各类型haar特征最小高度*/
	u16 w = w_array[type];/*宽度初始值*/
	u16 h = h_array[type];/*高度初始值*/
	u16 ch = floor(wndSize * 1.0 / h);
	u16 cw = floor(wndSize * 1.0 / w);
	u16 i, j, m, n;

	for(m = 1; m <= ch; m++)
	{
		for(n = 1; n <= cw;n++)
		{
			for(i = 0; i <= (wndSize - m * h); i++)
			{
				for(j = 0; j <= (wndSize - n * w); j++)
				{
					make_haar_feat(&feat_array[count], type, i, j, n, m, wndSize);
					count++;
				}
			}
		}
	}
	return count;
}


/*
功能：生成指定大小窗口的所有haar特征
*/
Haar_feat *make_haar_features(u16 wndSize, i32 *num)
{
	i32 feat_num = calc_haar_feat_num(wndSize, 2, 1)/*a类特征*/
				 + calc_haar_feat_num(wndSize, 3, 1)/*b类特征*/
				 + calc_haar_feat_num(wndSize, 1, 2)/*c类特征*/
				 + calc_haar_feat_num(wndSize, 1, 3)/*d类特征*/
				 + calc_haar_feat_num(wndSize, 2, 2);/*e类特征*/
	Haar_feat *feat_array = (Haar_feat*)malloc(sizeof(Haar_feat) * feat_num);
	i32 count = 0;
	
	/*生成各类特征*/
	count += make_specific_feat(FEAT_A, wndSize, &feat_array[count]);
	count += make_specific_feat(FEAT_B, wndSize, &feat_array[count]);
	count += make_specific_feat(FEAT_C, wndSize, &feat_array[count]);
	count += make_specific_feat(FEAT_D, wndSize, &feat_array[count]);
	count += make_specific_feat(FEAT_E, wndSize, &feat_array[count]);
	
	assert(feat_num == count);
	*num = feat_num;
	return feat_array;
}	

