///*------------------------------------------------------------------------------------------
//@ Deep Learning Image Processing  by Young-Keun Kim - Handong Global University
//
//
// Description     : Exercise:  C++ reference exercise
//------------------------------------------------------------------------------------------*/

#include <iostream>
using namespace std;

double adder(double numIn1, double numIn2)
{
    return numIn1 + numIn2;
}

double adder(double numIn1, double numIn2, double numIn3)
{
    return numIn1 + numIn2 + numIn3;
}

void add(double numIn1, double numIn2, double& numOut) 
{
    numOut = adder(numIn1, numIn2);
}

class MyNum {
public:
    MyNum() {};  	// Option 1 
    MyNum(int x); 	// Option 2     
    int num;
    double numinv();
};


// Class Constructor Definition 
MyNum::MyNum(int x)
{
    num = x;
}

double MyNum::numinv()
{
    return 1.0 / num;
}


int main()
{

    // Option 1
    MyNum mynum;
    mynum.num = 10;

    // Option 2
    //MyNum mynum(10);
    double out = 0;

    add(10, 20, out);

    cout << mynum.num << endl;
    // print its inverse value using  numinv() 
    cout << mynum.numinv() << endl;

    cout << adder(2.5, 5.9, 100) << endl;
    cout << out << endl;

    return 0;
}