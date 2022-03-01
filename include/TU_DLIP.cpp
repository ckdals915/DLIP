#include "TU_DLIP.h"

void proj_A::myNum::print() {
	printf("val1	= %d\n", val1);
	std::cout << "val2	= " << val2 << std::endl;
	std::cout << "sum	= " << sum() << std::endl;
	std::cout << "dsize	= " << sizeof(sum()) << std::endl;
}

int proj_A::myNum::sum() {
	return val1 + val2;
}

proj_A::myNum::myNum(int in1, int in2)		//option 2
{
	val1 = in1;
	val2 = in2;
}

proj_A::myNum::myNum() {};



















