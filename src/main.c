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
	Model *model = attentional_cascade(t_pos_data, v_pos_data, t_neg_data, v_neg_data, 19, 0.00001, 0.5, 0.005);
	

	printf("Training end...\n");
	
	//Model *m = load_model("./backup/test_model.cfg");

	return 0;
}