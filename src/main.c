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
	Data train_pos = load_image_data("./face.train");
	Data valid_pos = load_image_data("./face.validation");
	Data train_neg = load_image_data("./non-face.train");
	Data valid_neg = load_image_data("./non-face.validation");
	
	printf("training face size:%d\n", train_pos.im_num);
	printf("validation face size:%d\n", valid_pos.im_num);
	printf("training non-face size:%d\n", train_neg.im_num);
	printf("validation non-face size:%d\n", valid_neg.im_num);

	/*训练模型*/
	printf("Training begin...\n");
	Model *model = train_model(train_pos, valid_pos, train_neg, valid_neg);

	//save_model(model, "./backup/test_model.cfg");
	
	//Model *m = load_model("./backup/test_model.cfg");

	return 0;
}