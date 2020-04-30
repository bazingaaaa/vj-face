#include <math.h>
#include "const.h"
#include "image.h"
#include "feature.h"
#include "stdlib.h"


/*
??????????????ÈÑ???????haar????????
??????wndSize-?????ÈÑ???????Ò·?
	 featW-???????Ñò???
	 featH-???????Ä²??
???????Haar-like????????
*/
u32 calc_haar_feat_num(u16 wndSize, u16 w, u16 h)
{
	/*????????*/
	u16 ch = floor(wndSize * 1.0 / h);
	u16 nh = ch * (wndSize + 1) - ((1 + ch) * ch) / 2 * h;
	/*???????*/
	u16 cw = floor(wndSize * 1.0 / w);
	u16 nw = cw * (wndSize + 1) - ((1 + cw) * cw) / 2 * w;
	return nh * nw;
}


/*
???????³Àhaar????????????????????
?????x????????y???????
*/
void make_haar_feat(Haar_feat *pFeat, u8 type, u16 i, u16 j, u16 w, u16 h, u16 src_wnd_size)
{
	pFeat->type = type;
	pFeat->i = i;
	pFeat->j = j;
	pFeat->w = w;
	pFeat->h = h;
	pFeat->src_wnd_size = src_wnd_size;
}


/*
?????????A???????
*/	
float calc_featA_val(image integ, i32 i, i32 j, i32 w, i32 h)
{
 	return calc_im_sum(integ, i, i + h - 1, j, j + w - 1)
 		   -calc_im_sum(integ, i, i + h - 1, j + w, j + 2 * w - 1);
}


/*
?????????B???????
*/
float calc_featB_val(image integ, i32 i, i32 j, i32 w, i32 h)
{
 	return calc_im_sum(integ, i, i + h - 1, j, j + w - 1)
 			-calc_im_sum(integ, i, i + h - 1, j + w, j + 2 * w - 1)
			+calc_im_sum(integ, i, i + h - 1, j + 2 * w, j + 3 * w - 1);
}


/*
?????????C???????
*/
float calc_featC_val(image integ, i32 i, i32 j, i32 w, i32 h)
{
 	return calc_im_sum(integ, i, i + h - 1, j, j + w - 1)
 			-calc_im_sum(integ, i + h, i + 2 * h - 1, j, j + w - 1);
}


/*
?????????D???????
*/
float calc_featD_val(image integ, i32 i, i32 j, i32 w, i32 h)
{
 	return calc_im_sum(integ, i, i + h - 1, j, j + w - 1)
 			-calc_im_sum(integ, i + h, i + 2 * h - 1, j, j + w - 1)
			+calc_im_sum(integ, i + 2 * h, i + 3 * h - 1, j, j + w - 1);
}


/*
?????????E???????
*/
float calc_featE_val(image integ, i32 i, i32 j, i32 w, i32 h)
{
 	return calc_im_sum(integ, i, i + h - 1, j, j + w - 1)
 			-calc_im_sum(integ, i + h, i + 2 * h - 1, j, j + w - 1)
			-calc_im_sum(integ, i, i + h - 1, j + w, j + 2 * w - 1)
			+calc_im_sum(integ, i + h, i + 2 * h - 1, j + w, j + 2 * w - 1);
}


/*
?????????haar?????(??????scaling?????)
??????pWnd-???????????
	 pFeat-????????????
????????????An Analysis of the Viola-Jones Face Detection Algorithm?¿ì???3
*/
float calc_haar_feat_val(image integ, Haar_feat *pFeat)
{
	float ret = 0;
	if(!(pFeat->src_wnd_size != 0 && pFeat->src_wnd_size <= integ.w))
	{
		printf("pFeat->src_wnd_size:%d\n w:%d h:%d\n", pFeat->src_wnd_size, integ.w, integ.h);
		assert(0);
	}
	//assert(pFeat->src_wnd_size != 0 && pFeat->src_wnd_size <= integ.w);
	i32 i = pFeat->i, j = pFeat->j, w = pFeat->w, h = pFeat->h, a;
	i32 e = integ.w;
	i32 s = pFeat->src_wnd_size;

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
??????????????????????????
?????????????????
*/
i32 make_specific_feat(u8 type, u16 wndSize, Haar_feat *feat_array)
{
	i32 count = 0;
	u16 w_array[] = {2, 3, 1, 1, 2};/*?????????haar??????ÈÑ????*/
	u16 h_array[] = {1, 1, 2, 3, 2};/*?????????haar??????ÈÑ???*/
	u16 w = w_array[type];/*???????*/
	u16 h = h_array[type];/*??????*/
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
??????????????ÈÑ?????????haar????
*/
Haar_feat *make_haar_features(u16 wndSize, i32 *num)
{
	i32 feat_num = calc_haar_feat_num(wndSize, 2, 1)/*a??????*/
				 + calc_haar_feat_num(wndSize, 3, 1)/*b??????*/
				 + calc_haar_feat_num(wndSize, 1, 2)/*c??????*/
				 + calc_haar_feat_num(wndSize, 1, 3)/*d??????*/
				 + calc_haar_feat_num(wndSize, 2, 2);/*e??????*/
	Haar_feat *feat_array = (Haar_feat*)malloc(sizeof(Haar_feat) * feat_num);
	i32 count = 0;
	
	/*???????????*/
	count += make_specific_feat(FEAT_A, wndSize, &feat_array[count]);
	count += make_specific_feat(FEAT_B, wndSize, &feat_array[count]);
	count += make_specific_feat(FEAT_C, wndSize, &feat_array[count]);
	count += make_specific_feat(FEAT_D, wndSize, &feat_array[count]);
	count += make_specific_feat(FEAT_E, wndSize, &feat_array[count]);
	
	assert(feat_num == count);
	*num = feat_num;
	return feat_array;
}	

