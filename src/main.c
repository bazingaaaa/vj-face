#include <stdio.h>
#include "type.h"
#include "const.h"
#include "image.h"
#include "feature.h"
#include "classifier.h"
#include "model.h"
#include "data.h"
#include "proto.h"


int main(int argc, char *argv[])
{
	/*加载数据*/
	times("Loading data...\n");
	Data t_pos_data = load_image_data("./face.train");
	Data v_pos_data = load_image_data("./face.validation");
	Data t_neg_data = load_image_data("./non-face.train");
	Data v_neg_data = load_image_data("./non-face.validation");
	
	printf("training pos size:%d\n", t_pos_data.im_num);
	printf("validation pos size:%d\n", v_pos_data.im_num);
	printf("training neg size:%d\n", t_neg_data.im_num);
	printf("validation neg size:%d\n", v_neg_data.im_num);

	/*训练模型*/
	printf("Training begin...\n");
	Model *model = attentional_cascade(t_pos_data, v_pos_data, t_neg_data, v_neg_data, 24, FPR_GOAL, 0.5, 0.005);
	printf("Training end...\n");

	// save_model(model, "./backup/attentional_cascade_final.cfg");
	
	//Model *model = load_model("./backup/attentional_cascade_6.cfg");

	// prepare_pos_examples(t_pos_data);
	// prepare_pos_examples(v_pos_data);
	// Train_example *t_pos = make_pos_example(t_pos_data);
	// printf("make train pos example finish\n");	
	// Train_example *v_pos = make_pos_example(v_pos_data);
	// printf("make valid pos example finish\n");

	// float rate1 = test_model(model, t_pos, t_pos_data.im_num);
	// float rate2 = test_model(model, v_pos, v_pos_data.im_num);
	// printf("rate1:%f rate1:%f\n", rate1, rate2);
	// image im = load_image_extend("/Users/bazinga/Desktop/lena512.jpg");
	// image im_small = down_sample(im, 100);
	// scale_image(im_small, 1.0/255);
	// show_image(im_small, "test", 500000);
	//detect(im, model);


	return 0;
}