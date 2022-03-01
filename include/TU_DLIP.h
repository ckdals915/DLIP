#ifndef _TU_DLIP_H		// same as "#if !define _TU_DLIP_H" (or #pragma once) 
#define _TU_DLIP_H

#include <iostream>
namespace proj_A {
	class myNum {
	public:
		// variable
		int val1;
		int val2;

		// constructor
		myNum();					//option 1
		myNum(int in1, int in2);

		// function
		int sum();
		void print();
	};
}


#endif // !_TU_DLIP_H