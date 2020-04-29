#ifndef _CONST_H_
#define _CONST_H_


#define ASSERT
#ifdef ASSERT
void assertion_failure(char *exp, char *file, int line);
#define assert(exp)  if (exp) ; \
        else assertion_failure(#exp, __FILE__, __LINE__)
#else
#define assert(exp)
#endif
      
#define PRIVATE static 
#define PUBLIC
#define NEAREST_INTEGER(a) ((int)(a + 0.5))  	


#define MAG_V 0xCC

#define TWOPI 6.2831853

#define SET_RANGE(a, min, max) {\
                                    if(a<min)\
                                        a=min;\
                                    if(a>max)\
                                    a=max;\
                                }


#define MAX_DEPTH(l) (MIN((10 * l) + 10, 200))
//#define MAX_DEPTH(l) (0)

#define UNIT_DECAY_RATE (0.8)
#define FPR_GOAL (10e-7)


/*Ŀǰ��ʵ��һ������ģʽ*/
typedef enum haar_feature_mode
{
	HAAR_BASIC,/*��������ֱ��������5������ģ��*/
	//HAAR_ALL/*������Ե�����ԣ�������������14������ģ��*/
}Feat_mode;



/*haar�������,������,����
a��c������������ɣ��ֱ�Ϊˮƽ�ʹ�ֱ�ṹ
b��d������������ɣ��ֱ�Ϊˮƽ�ʹ�ֱ�ṹ
e���ĸ�������ɣ�ˮƽ�ʹ�ֱ�����������
*/
typedef enum haar_feature_type
{
	FEAT_A,
	FEAT_B,
	FEAT_C,
	FEAT_D,
	FEAT_E
}Feat_type;


#endif