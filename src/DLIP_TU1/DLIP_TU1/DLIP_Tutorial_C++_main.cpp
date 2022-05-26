#include "..//..//..//include//TU_DLIP.h"


namespace proj_B {
	//template <typename T>
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
		int sum(int in1) {
			return sum() + in1;
		}
		int sum(int in1, int in2 = 0, int in3 = 0) {
			return sum() + in1 + in2 + in3;
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

using namespace std;

void main() {
	// Data Type designate
	proj_B::myNum mynum(11, 1100, 11);

	mynum.print();

	cout << mynum.sum(10000, 10000) << endl;

	system("pause");
}