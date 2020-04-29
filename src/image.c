#include "const.h"
#include "image.h"
#include "feature.h"
#include "classifier.h"
#include "model.h"
#include "data.h"
#include <vector>
#include "math.h"
#include "list.h"
#include "proto.h"


static void add_pixel(image m, int x, int y, int c, float val);


/*
���ܣ����ɻ���ͼ��
��ע����Ӧ����An Analysis of the Viola-Jones Face Detection Algorithm�е��㷨2
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
���ܣ����û���ͼ����ָ�������ڵ�����֮��
��ע��i��j�ķ�����get_pixel���x/y����������ͬ
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
���ܣ�����ͼ���ֵ
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
���ܣ�����ͼ�񷽲�
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
���ܣ�ͼ���һ����in-place��
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
���ܣ�ͼ���һ���ͻ���ͼ��
��ע����һ�α������ͼ���һ���ͻ���ͼ��������
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
���ܣ�����ͼ��
*/
image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    memcpy((char*)copy.data, (char*)im.data, im.w * im.h * im.c * sizeof(float));
    return copy;
}


/*
���ܣ�����ͼ������ֵ
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
����:��ȡ����ֵ
*/
float get_pixel_extend(image m, int x, int y)
{
    if(x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
    return m.data[y*m.w + x];
}


/*
����:��������ֵ
*/
void set_pixel_extend(image im, int x, int y, float v)
{
    im.data[y * im.w + x] = v;
}


/*
����:��ȡ����ֵ
*/
float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}


/*
����:��������ֵ
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
���ܣ��ü�ͼƬ
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
���ܣ��ü�ͼƬ
*/
image crop_image_extend(image im, int dx, int dy, int w, int h)
{
    image cropped = make_image(w, h, 1);
    int i, j;
    #pragma omp parallel for private(j, i) num_threads(32)
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
���ܣ���ά��˹����
*/
float gaissian2d(int x, int y, float sigma)
{
    return 1 / (TWOPI * sigma * sigma) * exp(-(x * x + y * y)/(2 * sigma * sigma)); 
}


/*
���ܣ����ɸ�˹�˲���
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
���ܣ��ԻҶ�ͼ����о��
��ע������ͼ��߽�ʱ�����ڽ���Ԫ�����
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
���ܣ��ԻҶ�ͼ����и�˹�˲�
*/
image gaussian_filter_image(image im, float sigma)
{
    image gaussian_filter = make_gaussian_filter(sigma);
    image out = convolve_image(im, gaussian_filter);
    free_image(gaussian_filter);
    return out;
}


/*
���ܣ�������ͼ�񽵲���ָֻ����С
��ע����Ӧ����An Analysis of the Viola-Jones Face Detection Algorithm�е��㷨8 
     ��in-place ԭͼ���ͷ�
*/
image down_sample(image im, i32 wnd_size)
{
    i32 i, j, ij;
    i32 e = im.w;
    float i_new, j_new;
    i32 i_new_min, i_new_max, j_new_min, j_new_max;
    float pixel_val;
    assert(e > wnd_size);
    //float sigma = 0.6 * sqrt((e * 1.0 / wnd_size) * (e * 1.0 / wnd_size) - 1);/*�����и����Ĺ�ʽ*/
    //image im_smooth = gaussian_filter_image(im, sigma);
    image im_smooth = im;
    image out = make_image(wnd_size, wnd_size, 1);

    #pragma omp parallel for num_threads(16) schedule(static)
    for(ij = 0; ij < wnd_size * wnd_size; ij++)
    {
        i = ij / wnd_size;
        j = ij % wnd_size;
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
    //free_image(im_smooth);
    return out;
}


/*
���ܣ��ڱ����ͼ���ϻ�����ⴰ��in-place��
��ע��xΪ���ᣬyΪ����
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
���ܣ���rgbͼ��ת��Ϊ�Ҷ�ͼ��
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
���ܣ�ͼ�����س����ض�ֵ
*/
void scale_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] *= s;
}


/*
���ܣ�����ͼƬ��С
��ע��ʹ��billinear��ֵ��
*/
image constrain_image_size(image im, i32 size_limit)
{
    if(im.w <= size_limit && im.h <= size_limit)
    {
        return im;
    }
    else if(im.w > im.h)
    {
        image resized = resize_image(im, size_limit, (float)im.h * size_limit / im.w);
        free_image(im);
        return resized;
    }
    else
    {
        image resized = resize_image(im, (float)im.w * size_limit / im.h, size_limit);
        free_image(im);
        return resized;
    }
}



/*
���ܣ�����ͼ���С
*/
image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);   
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}


static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}
