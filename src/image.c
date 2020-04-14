#include "const.h"
#include "image.h"
#include "feature.h"
#include "classifier.h"
#include "model.h"
#include "data.h"
#include "proto.h"



/*
功能：生成积分图像
备注：对应论文An Analysis of the Viola-Jones Face Detection Algorithm中的算法2
*/
image make_intergral_image(image im)
{
    i32 i, j, k;
    image integ = make_image(im.w, im.h, im.c);

    for(i = 0; i < im.w; i++)
    {
        for(j = 0; j < im.h; j++)
        {
            float pixel_val = get_pixel_extend(im, i, j)
                            + get_pixel_extend(integ, i, j - 1)
                            + get_pixel_extend(integ, i - 1, j)
                            - get_pixel_extend(integ, i - 1, j - 1);
            set_pixel_extend(integ, i, j, pixel_val);
        }
    }
    return integ;
}


/*
功能：利用积分图像求指定区域内的像素之和
备注：i和j的方向与get_pixel里的x/y方向有所不同
*/
float calc_im_sum(image integ, i32 iBeg, i32 iEnd, i32 jBeg, i32 jEnd)
{
    float sum = get_pixel_extend(integ, jEnd, iEnd)
                    + get_pixel_extend(integ, jBeg - 1, iBeg - 1)
                    - get_pixel_extend(integ, jEnd, iBeg - 1)
                    - get_pixel_extend(integ, jBeg - 1, iEnd);
	return sum;
}


/*
功能：计算图像均值
*/
float calc_im_mean(image im)
{
	i32 i, j, k;
	float ret = 0;

	for(i = 0; i < im.w * im.h * im.c; i++)
	{
		ret += im.data[i];
	}
	return ret / (im.w * im.h * im.c);
}


/*
功能：计算图像方差
*/
float calc_im_var(image im, float mean)
{
	i32 i, j, k;
	float ret = 0;

	for(i = 0; i < im.w * im.h * im.c; i++)
	{
		float diff = im.data[i] - mean;
		ret += (diff * diff);
	}
	return ret / (im.w * im.h * im.c);
}



/*
功能：图像归一化（in-place）
*/
void normalize_image(image im, float mean, float var)
{
	i32 i, j, k;

	for(i = 0; i < im.w * im.h * im.c; i++)
	{
		im.data[i] = (im.data[i] - mean) / sqrt(var);
	}
}


/*
功能：图像归一化和积分图像
备注：用一次遍历完成图像归一化和积分图像求解过程
*/
image normalize_integral_image(image im, float mean, float var)
{
    i32 i, j, k;
    image integ = make_image(im.w, im.h, im.c);
    
    for(i = 0; i < im.w; i++)
    {
        for(j = 0; j < im.h; j++)
        {
            i32 pixel_idx = j * im.w + i;
            im.data[pixel_idx] = (im.data[pixel_idx] - mean) / sqrt(var);
            float pixel_val = im.data[pixel_idx]
                            + get_pixel_extend(integ, i, j - 1)
                            + get_pixel_extend(integ, i - 1, j)
                            - get_pixel_extend(integ, i - 1, j - 1);
            set_pixel_extend(integ, i, j, pixel_val);
        }
    }
    return integ;
}


/*
功能：拷贝图像
*/
image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    memcpy((char*)copy.data, (char*)im.data, im.w * im.h * im.c * sizeof(float));
    return copy;
}


/*
功能：限制图像像素值
*/
void constrain_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if(im.data[i] < 0) im.data[i] = 0;
        if(im.data[i] > 1) im.data[i] = 1;
    }
}


/*
功能:获取像素值
*/
float get_pixel_extend(image m, int x, int y)
{
    if(x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
    return m.data[y*m.w + x];
}


/*
功能:设置像素值
*/
void set_pixel_extend(image im, int x, int y, float v)
{
    im.data[y * im.w + x] = v;
}


/*
功能:获取像素值
*/
float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}


/*
功能:设置像素值
*/
void set_pixel(image im, int x, int y, int c, float v)
{
    if(x >= im.w || x < 0 || y >= im.h || y < 0 || c >= im.c || c < 0)/*invald input*/
    {
        return;
    }
    im.data[c * im.w * im.h + y * im.w + x] = v;
}


/*
功能：裁减图片
*/
image crop_image(image im, int dx, int dy, int w, int h)
{
    image cropped = make_image(w, h, im.c);
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int r = j + dy;
                int c = i + dx;
                float val = 0;
                r = constrain_int(r, 0, im.h-1);
                c = constrain_int(c, 0, im.w-1);
                val = get_pixel(im, c, r, k);
                set_pixel(cropped, i, j, k, val);
            }
        }
    }
    return cropped;
}


/*
功能：裁减图片
*/
image crop_image_extend(image im, int dx, int dy, int w, int h)
{
    image cropped = make_image(w, h, 1);
    int i, j, k;
    for(j = 0; j < h; ++j){
        for(i = 0; i < w; ++i){
            int r = j + dy;
            int c = i + dx;
            float val = 0;
            val = get_pixel_extend(im, c, r);
            set_pixel_extend(cropped, i, j, val);
        }
    }
    return cropped;
}