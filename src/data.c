#include "const.h"
#include "type.h"
#include "feature.h"
#include "image.h"
#include "classifier.h"
#include "model.h"
#include "data.h"
#include "proto.h"
#include "list.h"
#include <time.h>



char *fgetl(FILE *fp);


/*
功能：创建一个链表，链表中的每个元素代表文件中的每一列
*/
list *get_lines(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) {
        fprintf(stderr, "Couldn't open file %s\n", filename);
        exit(0);
    }
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}


/*
功能：装载图像数据
参数：images-图像数据目录文件
*/
Data load_image_data(char *images)
{
    Data im_data;
    list *image_list = get_lines(images);
    image *im_array = (image*)malloc(sizeof(image) * image_list->size);
    node *nd = image_list->front;
    i32 count = 0;

    while(nd){
        im_array[count] = load_image_extend((char*)nd->val);

        nd = nd->next;
        count++;
    }

    im_data.im_array = im_array;
    im_data.im_num = image_list->size;

    free_list_contents(image_list);
    free_list(image_list);
    
    return im_data;
}


/*
功能：从文件中获取一行
*/
char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char *line = (char*)malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = (char*)realloc(line, size*sizeof(char));
            if(!line) {
                fprintf(stderr, "malloc failed %ld\n", size);
                exit(0);
            }
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if(line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}


/*
功能：准备正样本数据，和负样本不同，正样本在训练过程中一直持有
*/
void prepare_pos_examples(Data data)
{
    i32 i;
    float mean;
    float var;
    
    for(i = 0; i < data.im_num; i++)
    {
        mean = calc_im_mean(data.im_array[i]);
        var = calc_im_var(data.im_array[i], mean);
        image tmp =  normalize_integral_image(data.im_array[i], mean, var);
        free_image(data.im_array[i]);
        data.im_array[i] = tmp;
    }
}


/*
功能：对正样本进行预处理获得训练样本
备注：对正样本进行归一化处理，并计算积分图像
    正样本和负样本处理不同，负样本需要裁减出子窗后，在子窗内进行归一化和积分图像处理
    而正样本可以直接对整个图像进行（无需裁减）
*/
Train_example * make_pos_example(Data data)
{
    i32 i;
    
    Train_example *examples = (Train_example*)malloc(sizeof(Train_example) * data.im_num);
    for(i = 0; i < data.im_num; i++)
    {
        examples[i].integ = data.im_array[i];
        examples[i].label = 1;
    }
    return examples;
}



/*
功能：对负样本进行预处理获得训练样本
参数：data-图像信息
     init_flag-是否是第一次获取样本
     example_num-需要的样本数量
     wnd_size-样本图像大小
     m-级联模型,用于筛选假阳性样本
备注：初始时随机取出指定数量的负样本数据，在训练过程中需要根据模型挑选出指定数量的假阳性样本
     当fpr（假阳性率）极小时，假阳性样本很难获取，此时采用遍历所有图片的方式寻找
*/
Train_example *make_neg_example(Data data, i32 init_flag, i32 example_num, i32 wnd_size, Model *m, float fpr)
{
    i32 count = 0;
    i32 im_size = data.im_num;
    float mean;
    float var;
    i32 i, j, k;
    image cropped, candidate;
    static i32 im_idx_recorder;/*记录上一次抽取样本的图片索引*/

    Train_example *examples = (Train_example*)malloc(sizeof(Train_example) * example_num);
    //srand(1);
    if(init_flag == 0 || fpr > 10e-4)
    {
        while(count < example_num)
        {
            i32 im_idx = rand() % im_size;
            i32 w = data.im_array[im_idx].w;
            i32 h = data.im_array[im_idx].h;
            if(w < wnd_size || h < wnd_size)/*图片太小*/
            {   
                continue;
           }
            i32 dx = rand()%(w - wnd_size);
            i32 dy = rand()%(h - wnd_size);
            cropped = crop_image_extend(data.im_array[im_idx], dx, dy, wnd_size, wnd_size);
            mean = calc_im_mean(cropped);
            var = calc_im_var(cropped, mean);
            candidate = normalize_integral_image(cropped, mean, var);
            free_image(cropped);
            if(var < 1 || (init_flag == 0 && 1 != model_func(m, candidate)))/*方差太小，数据对模型没有帮助*/
            {
                //free_image(cropped);
                free_image(candidate);
                continue;
            }
            examples[count].integ = candidate;
            examples[count].label = -1;
            
            count++;
        }
    }
    else
    {

    }
   
    return examples;
}


/*
功能：将正负样本合并
*/
Train_example *merge_pos_neg(Train_example *pos, i32 pos_num, Train_example *neg, i32 neg_num)
{
    Train_example *array = (Train_example*)malloc(sizeof(Train_example) * (pos_num + neg_num));
    memcpy(array, pos, sizeof(Train_example) * pos_num);
    memcpy(array + pos_num, neg, sizeof(Train_example) * neg_num);
    return array;
}


/*
功能：创建可并行处理的样本内存空间
备注：该内存空间可用于寻找最佳决策桩时的并行处理过程
*/
Feat_info **make_parallel_examples(i32 example_num, i32 feat_num)
{
    i32 i;
    Feat_info **parrel_examples = (Feat_info**)malloc(sizeof(Feat_info*) * feat_num);/*用于并行计算，提高速度*/

    for(i = 0; i < feat_num; i++)
    {
        parrel_examples[i] = (Feat_info*)malloc(sizeof(Feat_info) * example_num);
    }
    return parrel_examples;
}


/*
功能：释放并行处理的样本内存空间
*/
void free_parallel_examples(Feat_info **parrel_examples, i32 feat_num)
{
    i32 i;

    for(i = 0; i < feat_num; i++)
    {
        free(parrel_examples[i]);
    }
    free(parrel_examples);
}


/*
功能：释放图片数据
*/
void free_image_data(Data data)
{
    i32 i;

    for(i = 0; i < data.im_num; i++)
    {
        free_image(data.im_array[i]);
    }
}