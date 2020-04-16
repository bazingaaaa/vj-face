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
    image integ = make_image(im.w, im.h, 1);

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
    int i, j;
    #pragma omp parallel for private(j, i)
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


/*
功能：二维高斯函数
*/
float gaissian2d(int x, int y, float sigma)
{
    return 1 / (TWOPI * sigma * sigma) * exp(-(x * x + y * y)/(2 * sigma * sigma)); 
}


/*
功能：生成高斯滤波器
*/
image make_gaussian_filter(float sigma)
{
    int filter_size = ceil(sigma * 6);
    if(filter_size % 2 == 0)
        filter_size++;
    int i, j;
    int center_x = filter_size / 2;
    int center_y = filter_size / 2;
    image im = make_image(filter_size, filter_size, 1);
    for(i = 0;i < filter_size;i++)
    {
        for(j = 0;j < filter_size;j++)
        {
            im.data[j * filter_size + i] = gaissian2d(i - center_x, j -center_y, sigma);
        }
    }
    return im;
}


/*
功能：对灰度图像进行卷进
备注：处理图像边界时用最邻近的元素填充
*/
image convolve_image(image im, image filter)
{
    assert(filter.c == 1);
    int w_i = im.w;
    int h_i = im.h;
    int w_f = filter.w;
    int h_f = filter.h;
    int w_f_dist = w_f / 2;
    int h_f_dist = h_f / 2;
    int i, j;
    int m, n;

    image out = make_image(w_i, h_i, 1);
    for(i = 0;i < w_i;i++)
    {
        for(j = 0;j < h_i;j++)
        {
            float val = 0;
            for(m = 0;m < w_f;m++)
            {
                for(n = 0;n < h_f;n++)
                {
                    int x = i - w_f_dist + m;
                    int y = j - h_f_dist + n;
                    SET_RANGE(x, 0, w_i -1);
                    SET_RANGE(y, 0, h_i -1);
                    val += (filter.data[n * w_f + m] * im.data[y * w_i + x]);
                }
            }
            out.data[j * w_i + i] = val;
        }
    }
    return out;
}


/*
功能：对灰度图像进行高斯滤波
*/
image gaussian_filter_image(image im, float sigma)
{
    image gaussian_filter = make_gaussian_filter(sigma);
    image out = convolve_image(im, gaussian_filter);
    free_image(gaussian_filter);
    return out;
}


/*
功能：将输入图像降采样只指定大小
备注：对应论文An Analysis of the Viola-Jones Face Detection Algorithm中的算法8 
     非in-place 原图像被释放
*/
image down_sample(image im, i32 wnd_size)
{
    i32 i, j;
    i32 e = im.w;
    float i_new, j_new;
    i32 i_new_min, i_new_max, j_new_min, j_new_max;
    float pixel_val;
    assert(e > wnd_size);
    float sigma = 0.6 * sqrt((e * 1.0 / wnd_size) * (e * 1.0 / wnd_size) - 1);/*论文中给出的公式*/
    image im_smooth = gaussian_filter_image(im, sigma);
    image out = make_image(wnd_size, wnd_size, 1);
    for(i = 0; i < wnd_size; i++)
    {
        for(j = 0; j < wnd_size; j++)
        {
            i_new = (e - 1) * 1.0 / (wnd_size + 1) * (i + 1);
            j_new = (e - 1) * 1.0 / (wnd_size + 1) * (j + 1);
            i_new_max = MIN(NEAREST_INTEGER(i_new) + 1, e - 1);
            i_new_min = MAX(NEAREST_INTEGER(i_new), 0);
            j_new_max = MIN(NEAREST_INTEGER(j_new) + 1, e - 1);
            j_new_min = MAX(NEAREST_INTEGER(j_new), 0);
            pixel_val = 1.0 / 4 * (get_pixel_extend(im_smooth, i_new_max, j_new_max)
                                 + get_pixel_extend(im_smooth, i_new_min, j_new_max)
                                 + get_pixel_extend(im_smooth, i_new_min, j_new_min)
                                 + get_pixel_extend(im_smooth, i_new_max, j_new_min));
            set_pixel_extend(out, i, j, pixel_val);
        }
    }
    free_image(im_smooth);
    return out;
}


/*
功能：在被检测图像上画出检测窗（in-place）
备注：x为纵轴，y为横轴
*/
void draw_box(image im, i32 x, i32 y, i32 w, i32 h, float r, float g, float b)
{
    i32 i;
    i32 x1 = y, x2 = y + w - 1;
    i32 y1 = x, y2 = x + h - 1;

    if(x1 < 0) x1 = 0;
    if(x1 >= im.w) x1 = im.w-1;
    if(x2 < 0) x2 = 0;
    if(x2 >= im.w) x2 = im.w-1;

    if(y1 < 0) y1 = 0;
    if(y1 >= im.h) y1 = im.h-1;
    if(y2 < 0) y2 = 0;
    if(y2 >= im.h) y2 = im.h-1;

    for(i = x1; i <= x2; ++i){
        im.data[i + y1*im.w + 0*im.w*im.h] = r;
        im.data[i + y2*im.w + 0*im.w*im.h] = r;

        im.data[i + y1*im.w + 1*im.w*im.h] = g;
        im.data[i + y2*im.w + 1*im.w*im.h] = g;

        im.data[i + y1*im.w + 2*im.w*im.h] = b;
        im.data[i + y2*im.w + 2*im.w*im.h] = b;
    }
    for(i = y1; i <= y2; ++i){
        im.data[x1 + i*im.w + 0*im.w*im.h] = r;
        im.data[x2 + i*im.w + 0*im.w*im.h] = r;

        im.data[x1 + i*im.w + 1*im.w*im.h] = g;
        im.data[x2 + i*im.w + 1*im.w*im.h] = g;

        im.data[x1 + i*im.w + 2*im.w*im.h] = b;
        im.data[x2 + i*im.w + 2*im.w*im.h] = b;
    }
}


/*
功能：将rgb图像转换为灰度图像
*/
image rgb_to_grayscale(image im)
{
    int i, j;
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    for(i = 0;i < im.w;i++)
    {
        for(j = 0;j < im.h;j++)
        {
            int pixel_index = i + j * im.w;
            gray.data[pixel_index] = 0.299 * im.data[pixel_index] + 0.587 * im.data[pixel_index + im.w *im.h] + 0.114 * im.data[pixel_index + 2 * im.w *im.h];
        }
    }
    return gray;
}


/*
功能：图像像素乘以特定值
*/
void scale_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] *= s;
}
