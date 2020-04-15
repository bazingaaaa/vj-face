#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <stdio.h>
#include "type.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


typedef struct{
    int w,h,c;
    float *data;
} image;


typedef struct sub_wnd
{
    u16 size;/*窗口大小*/
    u16 startPosX;/*X轴是竖直方向*/
    u16 startPosY;/*Y轴是水平方向*/
    
    image imIntegral;/*窗口一阶积分图像*/
}Sub_wnd;


#ifdef OPENCV
    #include "opencv2/opencv.hpp"
    #include "opencv2/highgui/highgui_c.h"
    #include "opencv2/imgproc/imgproc_c.h"
    #include "opencv2/core/version.hpp"
    #include "opencv2/videoio/videoio_c.h"
    using namespace cv;
    //#include "opencv2/imgcodecs/imgcodecs_c.h"
    //image get_image_from_stream(CvCapture *cap);
    image get_image_from_stream(VideoCapture *cap);
    IplImage *image_to_ipl(image im);
    int show_image(image im, const char *name, int ms);
#endif
 

// Loading and saving
image make_image(int w, int h, int c);
image load_image(char *filename, int norm_flag);
void save_image(image im, const char *name);
void save_png(image im, const char *name);
void free_image(image im);
image copy_image(image im);
void constrain_image(image im);
void rgbgr_image(image im);
float get_pixel(image im, int x, int y, int c);
float get_pixel_extend(image m, int x, int y);
void set_pixel(image im, int x, int y, int c, float v);
void set_pixel_extend(image im, int x, int y, float v);
image make_intergral_image(image im);
float calc_im_mean(image im);
float calc_im_var(image im, float mean);
void normalize_image(image im, float mean, float var);
float calc_im_sum(image integ, i32 iBeg, i32 iEnd, i32 jBeg, i32 jEnd);
image crop_image(image im, int dx, int dy, int w, int h);
image crop_image_extend(image im, int dx, int dy, int w, int h);
image normalize_integral_image(image im, float mean, float var);
image down_sample(image im, i32 wnd_size);
void draw_box(image im, i32 x, i32 y, i32 w, i32 h, float r, float g, float b);


#endif

