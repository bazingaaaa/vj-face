#include <stdio.h>
#include <vector>
#include "type.h"
#include "const.h"
#include "image.h"
#include "feature.h"
#include "classifier.h"
#include "model.h"
#include "data.h"
#include "list.h"
#include "proto.h"



void train_model(char *train_pos_path, char *train_neg_path, char *vali_pos_path, char *vali_neg_path,  char *modelfile, char *sava_path,
		i32 wnd_size, double fnr_perstage, double fpr_perstage, double fpr_overall);


/*
功能：主函数
*/
int main(int argc, char *argv[])
{
	if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }

    if(0 == strcmp(argv[1], "train"))/*训练模型*/
    {
    	char *datacfg = argv[2];
    	list *options = read_data_cfg(datacfg);
    	if(NULL == options)
    	{
    		return -1;
    	}
    	char *train_pos = option_find_str(options, "train_positive", "./train_pos.list");
    	char *train_neg = option_find_str(options, "train_negative", "./train_neg.list");
    	char *vali_pos = option_find_str(options, "validation_positive", "./vali_pos.list");
    	char *vali_neg = option_find_str(options, "validation_negative", "./vali_neg.list");
    	char *save_path = option_find_str(options, "backup", "./backup");
    	i32 wnd_size = option_find_int(options, "window_size", 24);
    	double fnr_perstage = option_find_float(options, "fnr_perstage", 0.005);
    	double fpr_perstage = option_find_float(options, "fpr_perstage", 0.5);
    	double fpr_overall = option_find_float(options, "fpr_overall", 0.0000001);
    	char *modelfile = find_char_arg(argc, argv, "-model", 0);
        printf("using parameter:\n");
        printf("window_size:%d fnr_perstage:%lf fpr_perstage:%lf fpr_overall:%.10lf\n",
                wnd_size, fnr_perstage, fpr_perstage, fpr_overall);
    	train_model(train_pos, train_neg, vali_pos, vali_neg, modelfile, save_path, wnd_size, fnr_perstage, fpr_perstage, fpr_overall);
    }
    else if(0 == strcmp(argv[1], "test"))/*测试模型*/
    {
    	char *infile = NULL;
		char *modelfile = NULL;
		i32 skin_test_flag = 0;
		char *outfile = NULL;
		Model *model = NULL;
    	infile = argv[2];

    	modelfile = find_char_arg(argc, argv, "-model", "./backup/attention_cascade_def.cfg");
    	outfile = find_char_arg(argc, argv, "-outfile", 0);
    	skin_test_flag = find_int_arg(argc, argv, "-skintest", 0);
    	model = load_model(modelfile);
  		if(NULL == model)
  		{
			fprintf(stderr, "Load file failed\n");		
			return -1;
  		}
		image im = load_image_extend(infile);
		run_detection(im, model, skin_test_flag, outfile);
		free_model(model, 1);
		return 0;
    }
    else/*错误选项*/
    {
        fprintf(stderr, "Not an action: %s. \naction: train or test)\n", argv[0]);
    }
	
	return 0;
}


/*
功能：训练模型
*/
void train_model(char *train_pos_path, char *train_neg_path, char *vali_pos_path, char *vali_neg_path, char *modelfile, char *save_path,
		i32 wnd_size, double fnr_perstage, double fpr_perstage, double fpr_overall)
{
	char buf[100];
	/*加载数据，并对正样本进行预处理，负样本需要在训练过程中对截取的窗口进行处理，无法提前进行预处理*/
	printf("Loading data....\n");
	Data t_pos_data = load_image_data(train_pos_path);
	Data v_pos_data = load_image_data(vali_pos_path);
	Data t_neg_data = load_image_data(train_neg_path);
	Data v_neg_data = load_image_data(vali_neg_path);
    prepare_pos_examples(t_pos_data);
    prepare_pos_examples(v_pos_data);

	printf("train positive size:%d\n", t_pos_data.im_num);
	printf("validation positive size:%d\n", v_pos_data.im_num);
	printf("train negative size:%d\n", t_neg_data.im_num);
	printf("validation negative size:%d\n", v_neg_data.im_num);
	printf("Finish loading data....\n");
    

    /*模型加载和训练*/
	Model *model = load_model(modelfile);
	model = attentional_cascade(save_path, model, t_pos_data, v_pos_data, t_neg_data, v_neg_data, wnd_size, fpr_overall, fpr_perstage, fnr_perstage);
	
	/*最终模型保存*/
	i32 len = sprintf(buf, "%s/attentional_cascade_final.cfg", save_path);
	buf[len] = 0;
	save_model(model, buf);

    /*释放图片数据*/
    free_image_data(t_pos_data);
    free_image_data(t_neg_data);
    free_image_data(v_pos_data);
    free_image_data(v_neg_data);
}