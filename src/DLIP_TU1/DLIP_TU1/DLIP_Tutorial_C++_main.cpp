#include "..//..//..//include//TU_DLIP.h"

namespace proj_B {
	class myNum {
	public:
		int val1, val2, val3;
		myNum() {};
		myNum(int in1, int in2, int in3) {
			val1 = in1;
			val2 = in2;
			val3 = in3;
		}
		int sum() {
			return val1 + val2 + val3;
		}
		void print() {
			std::cout << "val1	= " << val2 << std::endl;
			std::cout << "val2	= " << val2 << std::endl;
			std::cout << "val3	= " << val2 << std::endl;
			std::cout << "sum	= " << sum() << std::endl;
			std::cout << "dsize	= " << sizeof(sum()) << std::endl;
		}

	};
}

void main() {

	/*myNum mynum;
	mynum.val1 = 10;
	mynum.val2 = 20;*/

	proj_A::myNum mynum(11, 1100);

	mynum.print();

	system("pause");
}