#include "const.h"
#include "image.h"
#include <time.h>
#include <sys/time.h>
#include <string.h>


/*
功能：断言失败
*/
void assertion_failure(char *exp, char *file, char *base_file, int line)
{
	printf("assert(%s) failed:file:%s,base_file:%s,ln:%d", exp, file, base_file, line);

	while(1)
	{
		;
	}
}


/*
功能：对整形范围进行限制
*/
int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}


/*
功能：时间函数
*/
void times(const char * which)
{
	/* If which is not empty, print the times since the previous call. */
	static double last_wall = 0.0, last_cpu = 0.0;
	double wall, cpu;
	struct timeval tv;
	clock_t stamp = 0;

	wall = last_wall;
	cpu = last_cpu;
	if (gettimeofday(&tv,NULL) != 0 || (stamp = clock()) == (clock_t)-1)
	{
		printf("Unable to get times\n");
	}
	last_wall = tv.tv_sec+1.0e-6*tv.tv_usec;
	last_cpu = stamp/(double)CLOCKS_PER_SEC;
	if (strlen(which) > 0)
	{
	    wall = last_wall-wall;
	    cpu = last_cpu-cpu;
	    printf("\n%s", which);
	    printf("time = %lf seconds, CPU = %lf  seconds\n\n", wall, cpu);
	}
}